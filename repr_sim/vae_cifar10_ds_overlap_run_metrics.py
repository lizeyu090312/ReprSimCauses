# import ffcv 

import platonic
from metrics import AlignmentMetrics

import torch, torchvision, os, tqdm, argparse
import numpy as np

from torch.utils.data import DataLoader
from torch import nn
import torchvision.transforms.v2 as v2

from torchvision.datasets import CIFAR10, MNIST, SVHN
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import sys
sys.path.append("../")
from train_vae import vae

def get_args_parser():
    parser = argparse.ArgumentParser("platonic", add_help=False)

    parser.add_argument("--log_base_dir", type=str, required=True)
    
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size")
    parser.add_argument("--metrics", nargs="+", type=str, default=AlignmentMetrics.SUPPORTED_METRICS, help="List of metrics to compute")
    return parser.parse_args()

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
            lvm_output1 = vision_model1.extract_features(ims)
            lvm_output2 = vision_model2.extract_features(ims)  # {key1: torch.tensor of size (bsz, ...), etc}
        
        lvm_feats1.append(torch.stack([v for v in lvm_output1.values()]).permute(1, 0, 2))  
        lvm_feats2.append(torch.stack([v for v in lvm_output2.values()]).permute(1, 0, 2))
    # compute score 
    lvm_feats1, lvm_feats2 = torch.cat(lvm_feats1), torch.cat(lvm_feats2)
    metric_to_score = dict()
    for metric in metrics:
        metric_to_score[metric] = platonic_metric.score(lvm_feats1, lvm_feats2, metric=metric, topk=10, normalize=True)['A'][0]
    return metric_to_score

if __name__ == "__main__":
    args = get_args_parser()
    
    tf_cifar10_svhn = transforms.Compose([
        transforms.Resize(28),  # due to VAE dimensions
        transforms.ToTensor(),
        transforms.Grayscale(),
    ])
    tf = transforms.Compose([
        transforms.ToTensor(),
    ])
    num_workers = 4
    
    cifar10 = CIFAR10(root='/tmp', download=True, train=False, transform=tf_cifar10_svhn)
    mnist = MNIST(root='/tmp', train=False, transform=tf)
    svhn = SVHN(root='/tmp', split='test', transform=tf_cifar10_svhn)
    
    device = torch.device("cuda:0")

    tl_mnist = DataLoader(mnist, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    tl_svhn = DataLoader(svhn, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    tl_cifar10 = DataLoader(cifar10, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

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
                first = vae.VariationalAutoencoder().eval()
                first.load_state_dict(torch.load(path_to_0, map_location=device))
                second = vae.VariationalAutoencoder().eval()
                second.load_state_dict(torch.load(path_to_1, map_location=device))
                
                vision_model1 = first
                vision_model2 = second
                # for metric in args.metrics:
                metric_overlap = compute_metric(tl_cifar10, device, args.metrics)
                metric_ood_all_mnist = compute_metric(tl_mnist, device, args.metrics)
                metric_ood_all_svhn = compute_metric(tl_svhn, device, args.metrics)
                
                f = open(path_to_cka_res, "a")
                for metric in args.metrics:
                    # "seed,metric,frac_overlap,metric_overlap,metric_ood_all_mnist,metric_ood_all_svhn
                    f.write(f"{seed:d},{metric},{frac_overlap},{metric_overlap[metric]:.6f},{metric_ood_all_mnist[metric]:.6f},{metric_ood_all_svhn[metric]:.6f}\n")
                f.close()
