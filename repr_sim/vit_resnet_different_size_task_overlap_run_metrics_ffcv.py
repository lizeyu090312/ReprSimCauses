import ffcv 

import sys, timm
sys.path.append("../")
from vit_resnet_different_model_size.train_task_overlap import make_dataloaders as make_dataloaders_vertical

import platonic
from metrics import AlignmentMetrics

import torch, torchvision, os, tqdm, argparse, csv
import numpy as np

from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data import Subset
import torchvision.transforms.v2 as v2
from torchvision.models.feature_extraction import create_feature_extractor


def get_args_parser():
    parser = argparse.ArgumentParser("platonic", add_help=False)

    parser.add_argument("--log_base_dir", type=str, default="../vit_resnet_different_model_size/log_dir/task_overlap")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--metrics", nargs="+", type=str, default=AlignmentMetrics.SUPPORTED_METRICS, help="List of metrics to compute")
    return parser.parse_args()


def build_model(model_name: str, num_classes: int):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    return model.cuda()


@torch.no_grad()
def compute_metric(dl, device, metrics):
    if dl is None: 
        return {metric:-1 for metric in metrics}
    platonic_metric = platonic.Alignment(
        dataset="tinyimagenet", # <--- this is the dataset 
        subset=None,    # <--- this is the subset
        # models=["openllama_7b", "llama_65b"], 
        models=["A", "B"], transform=None, device=device, dtype=torch.float32
        ) # you can also pass in device and dtype as arguments

    lvm_feats1, lvm_feats2 = [], []
    for i, (ims, _) in enumerate(dl):
        ims = ims.to(device)
        with torch.no_grad():
            lvm_output1 = vision_model1(ims).detach()
            lvm_output2 = vision_model2(ims).detach()
        
        lvm_feats1.append(torch.stack([lvm_output1]).permute(1, 0, 2))
        lvm_feats2.append(torch.stack([lvm_output2]).permute(1, 0, 2))
    # compute score 
    lvm_feats1, lvm_feats2 = torch.cat(lvm_feats1), torch.cat(lvm_feats2)
    metric_to_score = dict()
    for metric in metrics:
        metric_to_score[metric] = platonic_metric.score(lvm_feats1, lvm_feats2, metric=metric, topk=10, normalize=True)['A'][0]
    return metric_to_score


def load_dl_vertical_run_metrics_ffcv(ds_name, test_indices_ood, test_indices_common,
                                              orig_to_new_class_dict_common):
    ds_name_to_path = {"tinyimagenet":"../vit_resnet_different_model_size/tinyimagenet_ffcv_data/",
                       "cifar100":"../vit_resnet_different_model_size/cifar100_ffcv_data/",
                       "cifar10":"../vit_resnet_different_model_size/cifar10_ffcv_data/"}
    batch_size, num_workers = 512, 8
    tl_ood, tl_common = None, None
    if len(test_indices_ood) > 0:
        tl_ood = make_dataloaders_vertical(
            None, test_indices_ood, {i:i for i in range(200)}, 
            always_rand_order=True, torch_float32=True, 
            train_dataset=f"{ds_name_to_path[ds_name]}train.beton", 
            val_dataset=f"{ds_name_to_path[ds_name]}val.beton", 
            batch_size=batch_size, num_workers=num_workers)[0]['test']
    if len(orig_to_new_class_dict_common) > 0:
        tl_common = make_dataloaders_vertical(
            None, test_indices_common, orig_to_new_class_dict_common, 
            always_rand_order=True, torch_float32=True, 
            train_dataset=f"{ds_name_to_path[ds_name]}train.beton", 
            val_dataset=f"{ds_name_to_path[ds_name]}val.beton", 
            batch_size=batch_size, num_workers=num_workers)[0]['test']
    return {"tl_ood":tl_ood,"tl_common":tl_common}

'''
def load_dl_vertical_run_metrics_ffcv_ood():
    batch_size, num_workers = 512, 8
    tl_mnist = make_dataloaders_vertical(
        None, None, {i:i for i in range(10)}, 
        always_rand_order=True, torch_float32=True, 
        train_dataset="../tinyimagenet_ffcv_michael/mnist_ffcv_data/train.beton", 
        val_dataset="../tinyimagenet_ffcv_michael/mnist_ffcv_data/val.beton", 
        batch_size=batch_size, num_workers=num_workers)[0]['test']
    tl_svhn = make_dataloaders_vertical(
        None, None, {i:i for i in range(10)}, 
        always_rand_order=True, torch_float32=True, 
        train_dataset="../tinyimagenet_ffcv_michael/svhn_ffcv_data/train.beton", 
        val_dataset="../tinyimagenet_ffcv_michael/svhn_ffcv_data/val.beton", 
        batch_size=batch_size, num_workers=num_workers)[0]['test']
    return {"tl_mnist":tl_mnist,"tl_svhn":tl_svhn}
'''

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
    return {k: map1[k] for k in common_keys}


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_correct, total_num = 0., 0.
    for ims, labs in loader:
        with torch.autocast('cuda'):
            out = model(ims)
            total_correct += out.argmax(1).eq(labs).sum().cpu().item()
            total_num += ims.shape[0]
    return total_correct / total_num


if __name__ == "__main__":
    args = get_args_parser()
    device = torch.device("cuda:0")
    
    args.log_base_dir = os.path.join(args.log_base_dir, f"{args.dataset}_{args.model}")
    ds_name = args.dataset
    
    print(f"Starting with {args.log_base_dir}...")
    path_to_acc = os.path.join(args.log_base_dir, f"platonic_acc_ffcv.csv")
    f = open(path_to_acc, "a")
    f.write("seed,frac_overlap,model_idx,acc\n")  # model_idx is 1 or 2 (one pair of model is trained for each frac_overlap & seed)
    f.close()
    
    path_to_cka_res = os.path.join(args.log_base_dir, f"platonic_ffcv.csv")
    f = open(path_to_cka_res, "a")
    f.write("seed,metric,frac_overlap,metric_overlap,metric_ood_one\n")
    f.close()
    for seed in tqdm.tqdm(range(4)):
        all_dirs = os.listdir(os.path.join(args.log_base_dir, f"seed_{seed:d}"))
        for p in all_dirs:
            if "slurm" not in p:
                path_to_splits = os.path.join(args.log_base_dir, f"seed_{seed:d}", p, "split_indices")
                
                # construct datasets for overlapping data and non-overlapping data
                # note that there are two datasets: one containing classes with which both models were trained (overlap)
                # and another containing classes that were used to train only one model (ood)
                
                orig_to_new_class_dict_common = \
                    get_common_class_mapping(os.path.join(path_to_splits, "ret_orig_to_new_class_dict_split_train_0.txt"), 
                                             os.path.join(path_to_splits, "ret_orig_to_new_class_dict_split_train_1.txt"))
                n_classes = len(np.load(os.path.join(path_to_splits, "split_classes_train_0.npy")))
                s0_test = set(list(np.load(os.path.join(path_to_splits, "split_idx_test_0.npy"))))
                s1_test = set(list(np.load(os.path.join(path_to_splits, "split_idx_test_1.npy"))))
                s_common = np.array(list(s1_test & s0_test), dtype=int)
                s_ood_for_one_model = np.array(list(s1_test ^ s0_test), dtype=int)
                
                
                tl_tinyimagenet_dict = load_dl_vertical_run_metrics_ffcv(
                    ds_name, test_indices_ood=s_ood_for_one_model, 
                    test_indices_common=s_common,
                    orig_to_new_class_dict_common=orig_to_new_class_dict_common)
                tl_common = tl_tinyimagenet_dict["tl_common"]
                tl_ood = tl_tinyimagenet_dict["tl_ood"]
                
                # independently calculate frac overlap
                s0 = set(list(np.load(os.path.join(path_to_splits, "split_idx_train_0.npy"))))
                s1 = set(list(np.load(os.path.join(path_to_splits, "split_idx_train_1.npy"))))
                frac_overlap = len(s1 & s0) / len(s0)

                # load models
                path_to_0 = os.path.join(args.log_base_dir, f"seed_{seed:d}", p, "split_0/checkpoint.pth")
                path_to_1 = os.path.join(args.log_base_dir, f"seed_{seed:d}", p, "split_1/checkpoint.pth")
                first = build_model(args.model, n_classes).eval()
                first.load_state_dict(torch.load(path_to_0, map_location=device))
                second = build_model(args.model, n_classes).eval()
                second.load_state_dict(torch.load(path_to_1, map_location=device))
                
                acc1, acc2 = -1, -1
                if tl_common is not None:
                    acc1 = evaluate(first, tl_common)
                    acc2 = evaluate(second, tl_common)
                f = open(path_to_acc, "a")
                f.write(f"{seed},{frac_overlap},1,{acc1}\n")
                f.write(f"{seed},{frac_overlap},2,{acc2}\n")
                f.close()
                
                vision_model1 = lambda x: first.forward_head(first.forward_features(x), pre_logits=True)
                vision_model2 = lambda x: second.forward_head(second.forward_features(x), pre_logits=True)
                # for metric in args.metrics:
                metric_overlap = compute_metric(tl_common, device, args.metrics)
                metric_ood_one = compute_metric(tl_ood, device, args.metrics)
                # metric_ood_all_mnist = compute_metric(tl_mnist, device, args.metrics)
                # metric_ood_all_svhn = compute_metric(tl_svhn, device, args.metrics)
                
                f = open(path_to_cka_res, "a")
                for metric in args.metrics:
                    f.write(f"{seed:d},{metric},{frac_overlap},{metric_overlap[metric]:.6f},{metric_ood_one[metric]:.6f}\n")
                f.close()
    print(f"Done with {args.log_base_dir}")