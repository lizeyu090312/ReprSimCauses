from argparse import ArgumentParser
import time, random, copy, os, json, pickle
from datetime import datetime
import numpy as np

import torch as ch
import torchvision
from PIL import Image
from tinyimagenet_path import PATH_TO_TINYIMAGNET

split = "val"
def loader(path: str):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

ds_to_labels = {"cifar10":f"./log_dir/cifar10/labels_{split}.pkl", 
                "cifar100":f"./log_dir/cifar100/labels_{split}.pkl", 
                "tinyimagenet":f"./log_dir/tinyimagenet/labels_{split}.pkl"}

def write_labels(ds_name):
    train = not split == "val"
    if 'cifar100' == ds_name:
        dataset = torchvision.datasets.CIFAR100('/tmp', train=train, download=True)
    elif 'cifar10' == ds_name:
        dataset = torchvision.datasets.CIFAR10('/tmp', train=train, download=True)
    elif 'tinyimagenet' == ds_name:
        dataset = torchvision.datasets.DatasetFolder(os.path.join(PATH_TO_TINYIMAGNET, f"tiny_imagenet_200/{'train' if train else 'val/images'}"), 
                                                     transform=None, extensions="jpeg", loader=loader)
    os.makedirs(os.path.dirname(ds_to_labels[ds_name]), exist_ok=True)
    labels = [l for (_, l) in dataset]
    with open(ds_to_labels[ds_name], "wb") as f:
        pickle.dump({f"labels_{ds_name}":labels}, f)
    print(f"Written labels, saved to {ds_to_labels[ds_name]}")
    return

if __name__ == "__main__":
    for ds_name in ds_to_labels.keys():
        indices_dict = write_labels(ds_name)
        