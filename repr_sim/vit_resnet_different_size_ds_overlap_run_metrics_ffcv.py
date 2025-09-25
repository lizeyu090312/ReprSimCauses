import ffcv 

import sys, timm
sys.path.append("../")
from vit_resnet_different_model_size.train_ds_overlap import make_dataloaders as make_dataloaders_ds_overlap

import platonic
from metrics import AlignmentMetrics

import torch, torchvision, os, tqdm, argparse
import numpy as np

from torch.utils.data import DataLoader
from torch import nn
import torchvision.transforms.v2 as v2
from torchvision.models.feature_extraction import create_feature_extractor


def load_dl_horizontal_run_metrics_ffcv():
    batch_size, num_workers = 512, 8
    tinyimagenet_dl = make_dataloaders_ds_overlap(None, always_rand_order=True, torch_float32=True, 
            train_dataset="../vit_resnet_different_model_size/tinyimagenet_ffcv_data/train.beton", 
            val_dataset="../vit_resnet_different_model_size/tinyimagenet_ffcv_data/val.beton", 
            batch_size=batch_size, num_workers=num_workers)[0]['test']
    cifar100_dl = make_dataloaders_ds_overlap(None, always_rand_order=True, torch_float32=True, 
            train_dataset="../vit_resnet_different_model_size/cifar100_ffcv_data/train.beton", 
            val_dataset="../vit_resnet_different_model_size/cifar100_ffcv_data/val.beton", 
            batch_size=batch_size, num_workers=num_workers)[0]['test']
    cifar10_dl = make_dataloaders_ds_overlap(None, always_rand_order=True, torch_float32=True, 
            train_dataset="../vit_resnet_different_model_size/cifar10_ffcv_data/train.beton", 
            val_dataset="../vit_resnet_different_model_size/cifar10_ffcv_data/val.beton", 
            batch_size=batch_size, num_workers=num_workers)[0]['test']
    return {"tinyimagenet_dl":tinyimagenet_dl, "cifar10_dl":cifar10_dl, "cifar100_dl":cifar100_dl}


def get_args_parser():
    parser = argparse.ArgumentParser("platonic", add_help=False)

    parser.add_argument("--log_base_dir", type=str, default="../vit_resnet_different_model_size/log_dir/ds_overlap")
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
    dl_dict = load_dl_horizontal_run_metrics_ffcv()
    tl_cifar10 = dl_dict["cifar10_dl"]
    tl_cifar100 = dl_dict["cifar100_dl"]
    tl_tinyimagenet = dl_dict["tinyimagenet_dl"]
    
    args.log_base_dir = os.path.join(args.log_base_dir, f"{args.dataset}_{args.model}")
    ds_name = args.dataset
    if 'cifar100' == ds_name:
        num_classes = 100
        eval_dl = tl_cifar100
    elif 'cifar10' == ds_name:
        num_classes = 10
        eval_dl = tl_cifar10
    elif 'tinyimagenet' == ds_name:
        num_classes = 200
        eval_dl = tl_tinyimagenet

    print(f"Starting with {args.log_base_dir}...")
    path_to_acc = os.path.join(args.log_base_dir, f"platonic_acc_ffcv.csv")
    f = open(path_to_acc, "a")
    f.write("seed,frac_overlap,model_idx,acc\n")  # model_idx is 1 or 2 (one pair of model is trained for each frac_overlap & seed)
    f.close()

    path_to_cka_res = os.path.join(args.log_base_dir, f"platonic_ffcv.csv")
    f = open(path_to_cka_res, "a")
    f.write("seed,metric,frac_overlap,metric_cifar100,metric_tinyimagenet,metric_cifar10\n")
    f.close()
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
                path_to_0 = os.path.join(args.log_base_dir, f"seed_{seed:d}", p, "split_0/checkpoint.pth")
                path_to_1 = os.path.join(args.log_base_dir, f"seed_{seed:d}", p, "split_1/checkpoint.pth")
                first = build_model(args.model, num_classes).eval()
                first.load_state_dict(torch.load(path_to_0, map_location=device))
                second = build_model(args.model, num_classes).eval()
                second.load_state_dict(torch.load(path_to_1, map_location=device))
                
                acc1 = evaluate(first, eval_dl)
                acc2 = evaluate(second, eval_dl)
                f = open(path_to_acc, "a")
                f.write(f"{seed},{frac_overlap},1,{acc1}\n")
                f.write(f"{seed},{frac_overlap},2,{acc2}\n")
                f.close()
                
                vision_model1 = lambda x: first.forward_head(first.forward_features(x), pre_logits=True)
                vision_model2 = lambda x: second.forward_head(second.forward_features(x), pre_logits=True)
                # for metric in args.metrics:
                metric_tinyimagenet = compute_metric(tl_tinyimagenet, device, args.metrics)
                metric_cifar10 = compute_metric(tl_cifar10, device, args.metrics)
                metric_cifar100 = compute_metric(tl_cifar100, device, args.metrics)
                
                f = open(path_to_cka_res, "a")
                for metric in args.metrics:
                    # "seed,metric,frac_overlap,metric_overlap,metric_ood_all_mnist,metric_ood_all_svhn
                    f.write(f"{seed:d},{metric},{frac_overlap},{metric_cifar100[metric]:.6f},{metric_tinyimagenet[metric]:.6f},{metric_cifar10[metric]:.6f}\n")
                f.close()
    print(f"Done with {args.log_base_dir}")