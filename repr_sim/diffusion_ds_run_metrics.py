import ffcv 

import sys
sys.path.append("../")
from train_diffusion_unet.ds_overlap_train import make_dataloaders as make_dataloaders_ds, make_small_cond_unet, DATASETS

import platonic
from metrics import AlignmentMetrics

import torch, torchvision, os, tqdm, argparse
import numpy as np

from diffusers import DDPMScheduler

def load_dl_ds_run_metrics_ffcv(dsname):
    batch_size, num_workers = 512, 6
    dirname = f"../train_diffusion_unet/{dsname}_ffcv"
    train_ds_dl = make_dataloaders_ds(None, always_rand_order=True, torch_float32=True, 
            train_dataset=os.path.join(dirname, f'{dsname}_ffcv_data/train.beton'), 
            val_dataset=os.path.join(dirname, f'{dsname}_ffcv_data/val.beton'), 
            batch_size=batch_size, num_workers=num_workers)[0]['test']
    return {"train_ds_dl":train_ds_dl}

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
def compute_metric(dl, device, metrics, ddpm_sched:DDPMScheduler, model1, model2):
    if dl is None: 
        return {metric:-1 for metric in metrics}
    platonic_metric = platonic.Alignment(
        dataset="cifar10", # <--- this is the dataset 
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
            lvm_output1 = conv_act_feat(model1, noisy, t, torch.full(lbl.shape, UNCOND_ID, device=lbl.device, dtype=lbl.dtype))#lbl)
            lvm_output2 = conv_act_feat(model2, noisy, t, torch.full(lbl.shape, UNCOND_ID, device=lbl.device, dtype=lbl.dtype))#lbl)
        
        lvm_feats1.append(torch.stack([lvm_output1]).permute(1, 0, 2))
        lvm_feats2.append(torch.stack([lvm_output2]).permute(1, 0, 2))
    # compute score 
    lvm_feats1, lvm_feats2 = torch.cat(lvm_feats1), torch.cat(lvm_feats2)
    metric_to_score = dict()
    for metric in metrics:
        metric_to_score[metric] = platonic_metric.score(lvm_feats1, lvm_feats2, metric=metric, topk=10, normalize=True)['A'][0]
    return metric_to_score


if __name__ == "__main__":
    args = get_args_parser()
    dsname = args.dsname
    UNCOND_ID = DATASETS[dsname]['num_classes']
    img_size = 32 if dsname == 'cifar10' else 64
    device = torch.device("cuda:0")
    dl_dict = load_dl_ds_run_metrics_ffcv(dsname)
    tl_train_ds = dl_dict["train_ds_dl"]
    
    path_to_cka_res = os.path.join(args.log_base_dir, f"platonic_ffcv.csv")
    f = open(path_to_cka_res, "a")
    f.write("seed,metric,frac_overlap,metric_train_ds\n")
    f.close()
    ddpm_sched = DDPMScheduler(num_train_timesteps=1000)
    
    for seed in tqdm.tqdm(range(4)):
        all_dirs = os.listdir(os.path.join(args.log_base_dir, f"seed_{seed:d}"))
        for p in all_dirs:
            if "slurm" not in p:
                path_to_splits = os.path.join(args.log_base_dir, f"seed_{seed:d}", p, "split_indices")
                
                common = set(list(np.load(os.path.join(path_to_splits, "common.npy"))))
                s0 = set(list(np.load(os.path.join(path_to_splits, "split_0.npy"))))
                s1 = set(list(np.load(os.path.join(path_to_splits, "split_1.npy"))))
                frac_overlap = len(s1 & s0) / len(s0)
                
                # load models
                path_to_0 = os.path.join(args.log_base_dir, f"seed_{seed:d}", p, "split_0/ckpt.pth")
                path_to_1 = os.path.join(args.log_base_dir, f"seed_{seed:d}", p, "split_1/ckpt.pth")
                first = make_small_cond_unet(img_size, DATASETS[dsname]['num_classes']).eval()
                first.load_state_dict(torch.load(path_to_0, map_location=device))
                second = make_small_cond_unet(img_size, DATASETS[dsname]['num_classes']).eval()
                second.load_state_dict(torch.load(path_to_1, map_location=device))
                
                metric_train_ds = compute_metric(tl_train_ds, device, args.metrics, ddpm_sched, first, second)
                
                f = open(path_to_cka_res, "a")
                for metric in args.metrics:
                    f.write(f"{seed:d},{metric},{frac_overlap},{metric_train_ds[metric]:.6f}\n")
                f.close()
