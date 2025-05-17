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

import sys, torch, csv
sys.path.append("../data_splitting")
from data_splitting import split_dataset
from vae import VariationalAutoencoder, train_vae_gen


class RemappedCIFAR10(torch.utils.data.Dataset):
    def __init__(self, root, class_map, train=True, transform=None, target_transform=None, download=False):
        """
        Custom CIFAR10 dataset with remapped labels.

        Args:
            root (str): Root directory of dataset.
            class_map (dict): Mapping from original class index to new class index.
            train (bool): If True, creates dataset from training set, otherwise from test set.
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            download (bool): If True, downloads the dataset if it is not already available.
        """
        self.dataset = CIFAR10(root=root, train=train, transform=transform,
                               target_transform=target_transform, download=download)
        self.class_map = class_map

    def __getitem__(self, index):
        img, target = self.dataset[index]
        # Apply class mapping
        target = self.class_map.get(target, target)  # Use original target if not found in map
        return img, target

    def __len__(self):
        return len(self.dataset)

def generate_subsets(total_num_classes, desired_overlap):
    """
    Given the total number of classes and a desired overlap (defined as |s1 ∩ s2| / |s1|),
    finds the best combination of n (classes per subset) and k (overlapping classes)
    such that 2*n - k <= total_num_classes, and |k/n - desired_overlap| is minimized.
    
    Returns the number of classes per set (n) and the number of overlapping classes (k)
    """
    best_diff = float('inf')
    best_n, best_k = None, None
    # Try all possible n (number of classes in each subset) and k (overlap count)
    for n in range(1, total_num_classes + 1):
        for k in range(0, n + 1):
            if 2 * n - k > total_num_classes:
                continue  # cannot use more than available classes
            diff = abs((k / n) - desired_overlap)
            if diff < best_diff:
                best_diff = diff
                best_n, best_k = n, k
    if best_n is None:
        raise ValueError("No valid (n, k) combination found.")
    return best_n, best_k


def make_indices_task_train(ds_size, log_base_path, vertical_frac_overlap=0.4, num_classes=10, variable_ds_size=False):
    """Computes a dictionary of split to indices mappings for each split so that every split has the same
    prescribed fraction of data that's shared with other splits. For example, split_to_idx_dict[0] provides
    the indices of datapoints in split 0. This is class-wise overlap, so if vertical_frac_overlap < 1
    not every class is represented in each dataset. Important assumption: the dataset is ordered sequentially
    based on the class label with ds_size/num_classes data points per class.
    Args:
        ds_size (int): number of images in the dataset
        log_base_path (str): path to base log directory
        vertical_frac_overlap (float): fraction of classes is shared between any two splits. 
            Ensure that an all data points in a non-overlapping class belong in one and only one split
        num_classes (int, optional): number of classes in the dataset
    Returns:
        split_to_idx_dict: provides the indices in each split
        class_dict: provides the classes used in each split
        ret_orig_to_new_class_dict: provides the old to new class mappings for each split. The common classes
        should have identical mapping. 
    """
    path_to_log_dir = os.path.join(log_base_path, "split_indices")
    os.makedirs(path_to_log_dir, exist_ok=True)
    with open(os.path.join(path_to_log_dir, "info.json"), "w") as f:
        # writing input args
        all_params = locals()
        serializable_data = {k: v for k, v in all_params.items() if isinstance(v, (int, float, str, list, dict, tuple, bool, type(None)))}
        json.dump(serializable_data, f, indent=4, default=str)
    
    dataset = torchvision.datasets.CIFAR10("/tmp", train=True, download=True)
    ret_dict = split_dataset(dataset, vertical_frac_overlap, split_method="vertical", 
                             mode="unconstrained" if variable_ds_size else "constrained")
    s1, s2 = ret_dict["s1_indices"], ret_dict["s2_indices"]
    s1_classes, s2_classes = ret_dict["s1_classes"], ret_dict["s2_classes"]
    split_to_idx_dict, class_dict, ret_orig_to_new_class_dict = dict(), dict(), defaultdict(dict)
    split_to_idx_dict[0], split_to_idx_dict[1] = list(s1), list(s2)
    class_dict[0], class_dict[1] = list(s1_classes), list(s2_classes)
    common_classes = s1_classes & s2_classes
    for split in class_dict.keys():
        idxx = 0
        for c in common_classes:
            ret_orig_to_new_class_dict[split][c] = idxx
            idxx += 1
        for c in class_dict[split]:
            if c not in common_classes:
                ret_orig_to_new_class_dict[split][c] = idxx
                idxx += 1
    print(f"Updated task overlap (original={vertical_frac_overlap}): {len(set(s1)&set(s2))/len(s2)}")
    
    for split in class_dict.keys():
        np.save(os.path.join(path_to_log_dir, f"split_idx_train_{split:d}.npy"), 
                np.array(split_to_idx_dict[split]), allow_pickle=False)
        np.save(os.path.join(path_to_log_dir, f"split_classes_train_{split:d}.npy"), 
                np.array(class_dict[split]), allow_pickle=False)
        with open(os.path.join(path_to_log_dir, f"ret_orig_to_new_class_dict_split_train_{split:d}.txt"), "w") as f:
            f.write(f"orig_cls,new_cls\n")
            writer = csv.writer(f)
            for k, v in ret_orig_to_new_class_dict[split].items():
                writer.writerow([k, v])
    
    return split_to_idx_dict, class_dict, ret_orig_to_new_class_dict


def make_indices_task_test(ds_size, log_base_path, class_dict, num_classes):
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(torchvision.datasets.CIFAR10("/tmp", train=False, download=True)):
        class_to_indices[label].append(idx)
        
    path_to_log_dir = os.path.join(log_base_path, "split_indices")
    num_img_per_class = int(ds_size / num_classes)
    split_to_idx_dict = defaultdict(list)
    for split in class_dict.keys():
        for c in class_dict[split]:
            split_to_idx_dict[split].extend(class_to_indices[c])
    for split in class_dict.keys():
        split_to_idx_dict[split] = [int(x) for x in split_to_idx_dict[split]]
    for split in class_dict.keys():
        np.save(os.path.join(path_to_log_dir, f"split_idx_test_{split:d}.npy"), 
                np.array(split_to_idx_dict[split]), allow_pickle=False)
    return split_to_idx_dict


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
    
    cifar10 = RemappedCIFAR10(root='/tmp', class_map=None, download=True, train=True, transform=tf)
    
    indices_dict_train, class_dict, ret_orig_to_new_class_dict = \
        make_indices_task_train(ds_size=50_000, log_base_path=log_base_path, 
                          vertical_frac_overlap=args.frac_overlap, 
                          num_classes=10, variable_ds_size=False)
    for split in indices_dict_train.keys():
        split_log_path = os.path.join(log_base_path, f"split_{split}")
        os.makedirs(split_log_path, exist_ok=True)
        cifar10.class_map = {i:class_dict[split][i] for i in range(len(class_dict[split]))}
        ds_split = Subset(cifar10, indices_dict_train[split])
        dl = DataLoader(ds_split, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, )
        vae = VariationalAutoencoder()
        train_vae_gen(vae, dl, split_log_path, num_epochs=args.num_epoch, learning_rate=args.lr, 
                      device=device)
        