import ffcv 

import sys
sys.path.append("../")
from train_cifar_ffcv.ffcv_cifar10_train_ds_overlap import make_dataloaders as make_dataloaders_horizontal

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
    cifar10_dl = make_dataloaders_horizontal(None, always_rand_order=True, torch_float32=True, 
            train_dataset="../train_cifar_ffcv/train_cifar_ffcv_data/train.beton", 
            val_dataset="../train_cifar_ffcv/train_cifar_ffcv_data/val.beton", 
            batch_size=batch_size, num_workers=num_workers)[0]['test']
    mnist_dl = make_dataloaders_horizontal(None, always_rand_order=True, torch_float32=True, 
            train_dataset="../train_cifar_ffcv/mnist_ffcv_data/train.beton", 
            val_dataset="../train_cifar_ffcv/mnist_ffcv_data/val.beton", 
            batch_size=batch_size, num_workers=num_workers)[0]['test']
    svhn_dl = make_dataloaders_horizontal(None, always_rand_order=True, torch_float32=True, 
            train_dataset="../train_cifar_ffcv/svhn_ffcv_data/train.beton", 
            val_dataset="../train_cifar_ffcv/svhn_ffcv_data/val.beton", 
            batch_size=batch_size, num_workers=num_workers)[0]['test']
    return {"cifar10_dl":cifar10_dl, "mnist_dl":mnist_dl, "svhn_dl":svhn_dl}


def get_args_parser():
    parser = argparse.ArgumentParser("platonic", add_help=False)

    parser.add_argument("--log_base_dir", type=str, required=True)
    
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size")
    parser.add_argument("--metrics", nargs="+", type=str, default=AlignmentMetrics.SUPPORTED_METRICS, help="List of metrics to compute")
    return parser.parse_args()


def construct_resnet(device, n_classes):
    model = torchvision.models.resnet18()
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.eval()
    return model.to(device)


@torch.no_grad()
def compute_metric(dl, device, metrics):
    if dl is None: 
        return {metric:-1 for metric in metrics}
    platonic_metric = platonic.Alignment(
        dataset="cifar10", # <--- this is the dataset 
        subset=None,    # <--- this is the subset
        # models=["openllama_7b", "llama_65b"], 
        models=["A", "B"], transform=None, device=device, dtype=torch.float32
        ) # you can also pass in device and dtype as arguments

    lvm_feats1, lvm_feats2 = [], []
    for i, (ims, _) in enumerate(dl):
        ims = ims.to(device)
        with torch.no_grad():
            lvm_output1 = vision_model1(ims)
            lvm_output2 = vision_model2(ims)  # {key1: torch.tensor of size (bsz, ...), etc}
        
        lvm_feats1.append(torch.stack([v for v in lvm_output1.values()]).permute(1, 0, 2))
        lvm_feats2.append(torch.stack([v for v in lvm_output2.values()]).permute(1, 0, 2))
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
    # {"cifar10_dl":cifar10_dl, "mnist_dl":mnist_dl, "svhn_dl":svhn_dl}
    tl_mnist = dl_dict["mnist_dl"]
    tl_svhn = dl_dict["svhn_dl"]
    tl_cifar10 = dl_dict["cifar10_dl"]
        
    path_to_acc = os.path.join(args.log_base_dir, f"platonic_acc_ffcv.csv")
    f = open(path_to_acc, "a")
    f.write("seed,frac_overlap,model_idx,acc\n")  # model_idx is 1 or 2 (one pair of model is trained for each frac_overlap & seed)
    f.close()

    path_to_cka_res = os.path.join(args.log_base_dir, f"platonic_ffcv.csv")
    f = open(path_to_cka_res, "a")
    f.write("seed,metric,frac_overlap,metric_overlap,metric_ood_all_mnist,metric_ood_all_svhn\n")
    f.close()
    for seed in tqdm.tqdm(range(10)):
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
                first = construct_resnet(device, 10).eval()
                first.load_state_dict(torch.load(path_to_0, map_location=device))
                second = construct_resnet(device, 10).eval()
                second.load_state_dict(torch.load(path_to_1, map_location=device))
                
                acc1 = evaluate(first, tl_cifar10)
                acc2 = evaluate(second, tl_cifar10)
                f = open(path_to_acc, "a")
                # f.write("seed,seed,frac_overlap,model_idx,acc\n")
                f.write(f"{seed},{frac_overlap},1,{acc1}\n")
                f.write(f"{seed},{frac_overlap},2,{acc2}\n")
                f.close()

                return_nodes = ["flatten"]
                vision_model1 = create_feature_extractor(first, return_nodes=return_nodes)
                vision_model2 = create_feature_extractor(second, return_nodes=return_nodes)
                # for metric in args.metrics:
                metric_overlap = compute_metric(tl_cifar10, device, args.metrics)
                metric_ood_all_mnist = compute_metric(tl_mnist, device, args.metrics)
                metric_ood_all_svhn = compute_metric(tl_svhn, device, args.metrics)
                
                f = open(path_to_cka_res, "a")
                for metric in args.metrics:
                    # "seed,metric,frac_overlap,metric_overlap,metric_ood_all_mnist,metric_ood_all_svhn
                    f.write(f"{seed:d},{metric},{frac_overlap},{metric_overlap[metric]:.6f},{metric_ood_all_mnist[metric]:.6f},{metric_ood_all_svhn[metric]:.6f}\n")
                f.close()
