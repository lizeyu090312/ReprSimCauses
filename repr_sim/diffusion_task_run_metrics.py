import ffcv 

import sys
sys.path.append("../")
from train_diffusion_unet.task_overlap_train import make_dataloaders as make_dataloaders_task, make_small_cond_unet, DATASETS

import platonic
from metrics import AlignmentMetrics

import torch, torchvision, os, tqdm, argparse, csv
import numpy as np

from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data import Subset
import torchvision.transforms.v2 as v2
from torchvision.models.feature_extraction import create_feature_extractor

from diffusers import DDPMScheduler

def get_args_parser():
    parser = argparse.ArgumentParser("platonic", add_help=False)

    parser.add_argument("--log_base_dir", type=str, required=True)
    parser.add_argument("--dsname", type=str, required=True)
    
    parser.add_argument("--metrics", nargs="+", type=str, default=AlignmentMetrics.SUPPORTED_METRICS, help="List of metrics to compute")
    return parser.parse_args()

def attach_mid_hook(unet):
    feats = {"mid": []}
    h = unet.mid_block.register_forward_hook(
        lambda m, i, o, feats=feats: feats["mid"].append(o if not isinstance(o, (tuple, list)) else o[0])
    )
    return feats, [h]

@torch.no_grad()
def mid_feat(unet, x_noisy, t, y):
    feats, hs = attach_mid_hook(unet)
    _ = unet(x_noisy, t, class_labels=y, return_dict=True)
    for h in hs: h.remove()
    return feats["mid"][-1].mean(dim=(2,3))  # [B, d]


def attach_conv_act_hook(unet):
    feats = {"conv_act": []}
    h = unet.conv_act.register_forward_hook(
        lambda m, i, o, feats=feats: feats["conv_act"].append(o)
    )
    return feats, [h]

@torch.no_grad()
def conv_act_feat(unet, x_noisy, t, y):
    feats, hs = attach_conv_act_hook(unet)
    _ = unet(x_noisy, t, class_labels=y, return_dict=True)
    for h in hs:
        h.remove()

    x = feats["conv_act"][-1]           # [B, C, H, W]
    mu = x.mean(dim=(2, 3))
    sd = x.std(dim=(2, 3), unbiased=False)
    # print(torch.cat([mu, sd], dim=1).shape)  # [bsz, 192] for cifar10
    return torch.cat([mu, sd], dim=1)   # [B, 2C]

@torch.no_grad()
def compute_metric(dl, device, metrics, ddpm_sched:DDPMScheduler, model1, model2, ood, map1, map2):
    
    def map_label(inp_tensor: torch.Tensor, m: dict):
        keys = torch.tensor(list(m.keys()), device=inp_tensor.device)
        values = torch.tensor(list(m.values()), device=inp_tensor.device)

        out = torch.full_like(inp_tensor, fill_value=UNCOND_ID)

        mask = (inp_tensor.unsqueeze(-1) == keys)  # shape [N, K]
        idx = mask.float().argmax(dim=-1)          # index of first match (or 0 if none)
        matched = mask.any(dim=-1)

        out[matched] = values[idx[matched]]
        return out
    
    if dl is None: 
        return {metric:-1 for metric in metrics}
    platonic_metric = platonic.Alignment(
        dataset="ds", # <--- this is the dataset 
        subset=None,    # <--- this is the subset
        # models=["openllama_7b", "llama_65b"], 
        models=["A", "B"], transform=None, device=device, dtype=torch.float32
        ) # you can also pass in device and dtype as arguments

    lvm_feats1, lvm_feats2 = [], []
    for i, (ims, lbl) in enumerate(dl):
        ims, lbl = ims.to(device), lbl.long().to(device)
        t = torch.randint(0, ddpm_sched.config.num_train_timesteps, (ims.shape[0],), device=ims.device, dtype=torch.long)
        noise = torch.randn_like(ims)
        noisy = ddpm_sched.add_noise(ims, noise, t)
        with torch.no_grad():
            # lvm_output1 = mid_feat(model1, noisy, t, lbl)
            # lvm_output2 = mid_feat(model2, noisy, t, lbl)
            lvm_output1 = conv_act_feat(model1, noisy, t, torch.full(lbl.shape, UNCOND_ID, device=lbl.device, dtype=lbl.dtype))#map_label(lbl, map1) if ood else lbl)
            lvm_output2 = conv_act_feat(model2, noisy, t, torch.full(lbl.shape, UNCOND_ID, device=lbl.device, dtype=lbl.dtype))#map_label(lbl, map2) if ood else lbl)
        
        lvm_feats1.append(torch.stack([lvm_output1]).permute(1, 0, 2))
        lvm_feats2.append(torch.stack([lvm_output2]).permute(1, 0, 2))
    # compute score 
    lvm_feats1, lvm_feats2 = torch.cat(lvm_feats1), torch.cat(lvm_feats2)
    metric_to_score = dict()
    for metric in metrics:
        metric_to_score[metric] = platonic_metric.score(lvm_feats1, lvm_feats2, metric=metric, topk=10, normalize=True)['A'][0]
    return metric_to_score


def load_dl_task_run_metrics_ffcv_train_ds(dsname, test_indices_ood, test_indices_common,
                                              orig_to_new_class_dict_common):
    dirname = f"../train_diffusion_unet/{dsname}_ffcv"
    batch_size, num_workers = 512, 6
    tl_ood, tl_common = None, None
    if len(test_indices_ood) > 0:
        tl_ood = make_dataloaders_task(
            None, test_indices_ood, {i:i for i in range(10)}, 
            always_rand_order=True, torch_float32=True, 
            train_dataset=os.path.join(dirname, f'{dsname}_ffcv_data/train.beton'), 
            val_dataset=os.path.join(dirname, f'{dsname}_ffcv_data/val.beton'), 
            batch_size=batch_size, num_workers=num_workers)[0]['test']
    if len(orig_to_new_class_dict_common) > 0:
        tl_common = make_dataloaders_task(
            None, test_indices_common, orig_to_new_class_dict_common, 
            always_rand_order=True, torch_float32=True, 
            train_dataset=os.path.join(dirname, f'{dsname}_ffcv_data/train.beton'), 
            val_dataset=os.path.join(dirname, f'{dsname}_ffcv_data/val.beton'), 
            batch_size=batch_size, num_workers=num_workers)[0]['test']
    return {"tl_ood":tl_ood,"tl_common":tl_common}

def get_common_class_mapping(file1_path, file2_path):
    def read_mapping(file_path):
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            return {int(row['orig_cls']): int(row['new_cls']) for row in reader}
    
    map1 = read_mapping(file1_path)
    map2 = read_mapping(file2_path)
    common_keys = set(map1) & set(map2)
    for k in common_keys:
        assert map1[k] == map2[k]
    return {k: map1[k] for k in common_keys}, map1, map2


if __name__ == "__main__":
    args = get_args_parser()
    device = torch.device("cuda:0")

    dsname = args.dsname
    UNCOND_ID   = DATASETS[dsname]['num_classes']
    img_size = 32 if dsname == 'cifar10' else 64
    
    path_to_cka_res = os.path.join(args.log_base_dir, f"platonic_ffcv.csv")
    f = open(path_to_cka_res, "a")
    f.write("seed,metric,frac_overlap,metric_overlap,metric_ood_one\n")
    f.close()
    ddpm_sched = DDPMScheduler(num_train_timesteps=1000)
    
    for seed in tqdm.tqdm(range(4)):
        all_dirs = os.listdir(os.path.join(args.log_base_dir, f"seed_{seed:d}"))
        for p in all_dirs:
            if "slurm" not in p:
                path_to_splits = os.path.join(args.log_base_dir, f"seed_{seed:d}", p, "split_indices")
                
                # construct datasets for overlapping data and non-overlapping data
                # note that there are two datasets: one containing classes with which both models were trained (overlap)
                # and another containing classes that were used to train only one model (ood)
                
                orig_to_new_class_dict_common, map1, map2 = \
                    get_common_class_mapping(os.path.join(path_to_splits, "ret_orig_to_new_class_dict_split_train_0.txt"), 
                                             os.path.join(path_to_splits, "ret_orig_to_new_class_dict_split_train_1.txt"))
                n_classes = len(np.load(os.path.join(path_to_splits, "split_classes_train_0.npy")))
                s0_test = set(list(np.load(os.path.join(path_to_splits, "split_idx_test_0.npy"))))
                s1_test = set(list(np.load(os.path.join(path_to_splits, "split_idx_test_1.npy"))))
                s_common = np.array(list(s1_test & s0_test), dtype=int)
                s_ood_for_one_model = np.array(list(s1_test ^ s0_test), dtype=int)
                
                
                tl_dict = load_dl_task_run_metrics_ffcv_train_ds(dsname=dsname,
                    test_indices_ood=s_ood_for_one_model, 
                    test_indices_common=s_common,
                    orig_to_new_class_dict_common=orig_to_new_class_dict_common)
                tl_common = tl_dict["tl_common"]
                tl_ood = tl_dict["tl_ood"]
                
                # independently calculate frac overlap
                s0 = set(list(np.load(os.path.join(path_to_splits, "split_idx_train_0.npy"))))
                s1 = set(list(np.load(os.path.join(path_to_splits, "split_idx_train_1.npy"))))
                frac_overlap = len(s1 & s0) / len(s0)

                # load models
                path_to_0 = os.path.join(args.log_base_dir, f"seed_{seed:d}", p, "split_0/ckpt.pth")
                path_to_1 = os.path.join(args.log_base_dir, f"seed_{seed:d}", p, "split_1/ckpt.pth")
                first = make_small_cond_unet(img_size, DATASETS[dsname]['num_classes']).eval()
                first.load_state_dict(torch.load(path_to_0, map_location=device))
                second = make_small_cond_unet(img_size, DATASETS[dsname]['num_classes']).eval()
                second.load_state_dict(torch.load(path_to_1, map_location=device))
                
                # for metric in args.metrics:
                metric_overlap = compute_metric(tl_common, device, args.metrics, ddpm_sched, first, second, ood=False, map1=map1, map2=map2)
                metric_ood_one = compute_metric(tl_ood, device, args.metrics, ddpm_sched, first, second, ood=True, map1=map1, map2=map2)
                
                f = open(path_to_cka_res, "a")
                for metric in args.metrics:
                    f.write(f"{seed:d},{metric},{frac_overlap},{metric_overlap[metric]:.6f},{metric_ood_one[metric]:.6f}\n")
                f.close()
