"""
Fast training script for CIFAR-10 using FFCV.
For tutorial, see https://docs.ffcv.io/ffcv_examples/cifar10.html.

First, from the same directory, run:

    `python write_datasets.py --data.train_dataset [TRAIN_PATH] \
                              --data.val_dataset [VAL_PATH]`

to generate the FFCV-formatted versions of CIFAR.

Then, simply run this to train models with default hyperparameters:

    `python train_cifar.py --config-file default_config.yaml`

You can override arguments as follows:

    `python train_cifar.py --config-file default_config.yaml \
                           --training.lr 0.2 --training.num_workers 4 ... [etc]`

or by using a different config file.
"""
from typing import List
import time, random, copy, os, json, argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision.transforms as transforms

import sys, torch
sys.path.append("../data_splitting")
from data_splitting import split_dataset
from vae import VariationalAutoencoder, train_vae_gen


def make_indices_ds(ds_size, log_base_path, class_frac_overlap=0.5, num_classes=10, variable_ds_size=False) -> dict[str, list]:
    """Computes a dictionary of split to indices mappings for each split so that every split has the same
    prescribed fraction of data that's shared with other splits. For example, ret_dict[0] provides
    the indices of datapoints in split 0. Every class is represented in all splits (hence horizontal). 
    Important assumption: the dataset is ordered sequentially
    based on the class label with ds_size/num_classes data points per class.
    Args:
        ds_size (int): number of images in the dataset
        log_base_path (str): path to base log directory
        class_frac_overlap (float): fraction of data is shared between any two split. 
            Computation: Assume 2 splits, s1 = split 1 indices, s2 = split 2 indices, 
            common = s1 & s2 (aka indices that's found in both s1 and s2). len(common) / len(s1) 
            should equal len(common) / len(s2), and len(common) / len(s_whatever) 
            is the class_frac_overlap
        num_classes (int, optional): number of classes in the dataset
    Returns:
        dict[str, list] 
    """
    path_to_log_dir = os.path.join(log_base_path, "split_indices")
    os.makedirs(path_to_log_dir, exist_ok=True)
    with open(os.path.join(path_to_log_dir, "info.json"), "w") as f:
        # writing input args
        all_params = locals()
        serializable_data = {k: v for k, v in all_params.items() if isinstance(v, (int, float, str, list, dict, tuple, bool, type(None)))}
        json.dump(serializable_data, f, indent=4, default=str)
    dataset = torchvision.datasets.CIFAR10("/tmp", train=True, download=True)
    ret_dict = split_dataset(dataset, class_frac_overlap, split_method="horizontal", 
                             mode="unconstrained" if variable_ds_size else "constrained")
    rd = dict()
    rd[0], rd[1] = ret_dict["s1_indices"], ret_dict["s2_indices"]
    common_indices = np.array(list(set(rd[0]) & set(rd[1])))
    for split in rd.keys():
        np.save(os.path.join(path_to_log_dir, f"split_{split:d}.npy"), 
                np.array(rd[split]), allow_pickle=False)
    
    np.save(os.path.join(path_to_log_dir, f"common.npy"), 
            common_indices, allow_pickle=False)
    return rd

def get_args_parser():
    parser = argparse.ArgumentParser("PreferenceLFM", add_help=False)
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--output_root", type=str, default="./log_dir", help="path to logs")
    parser.add_argument("--frac_overlap", type=float, required=True, help="Dataset overlap")
    
    parser.add_argument("--batch_size", type=int, default=16, help="input batch size")
    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate g")

    
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    device = torch.device('cuda:0')
    
    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_base_path = os.path.join(args.output_root, f"seed_{args.seed}", 
                                formatted_datetime + "_desired_" + f"{args.frac_overlap:g}")
    os.makedirs(log_base_path, exist_ok=True)
    
    tf = transforms.Compose([
        transforms.Resize(28),  # due to VAE dimensions
        transforms.ToTensor(),
        transforms.Grayscale(),
    ])
    
    cifar10 = CIFAR10(root='/tmp', download=True, train=True, transform=tf)
    
    indices_dict = make_indices_ds(ds_size=50_000, 
                                log_base_path=log_base_path, 
                                class_frac_overlap=args.frac_overlap, 
                                num_classes=10, variable_ds_size=False)
    for split in indices_dict.keys():
        split_log_path = os.path.join(log_base_path, f"split_{split}")
        os.makedirs(split_log_path, exist_ok=True)
        ds_split = Subset(cifar10, indices_dict[split])
        dl = DataLoader(ds_split, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, )
        vae = VariationalAutoencoder()
        train_vae_gen(vae, dl, split_log_path, num_epochs=args.num_epoch, learning_rate=args.lr, 
                      device=device)
        