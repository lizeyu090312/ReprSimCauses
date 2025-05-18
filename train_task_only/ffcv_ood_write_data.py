import ffcv
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField

from argparse import ArgumentParser
from typing import List
import time, os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

Section('data', 'arguments to give the writer').params(
    train_dataset=Param(str, 'Where to write the new dataset', default="./digit_shape_datasets_ffcv/cifar10/train.beton"),
    val_dataset=Param(str, 'Where to write the new dataset', default="./digit_shape_datasets_ffcv/cifar10/val.beton"),
)

# Custom dataset wrapper to transform MNIST images
class TransformedMNIST(Dataset):
    def __init__(self, train=True):
        self.mnist = torchvision.datasets.MNIST('/tmp', train=train, download=True)
        
        # Define the transformation: resize to 32x32 and convert to 3 channels
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Resize to 32x32
            transforms.Grayscale(num_output_channels=3),  # Duplicate grayscale into 3 channels
            transforms.ToTensor(),  # Convert to tensor
            transforms.Lambda(lambda x: (x * 255).byte())  # Convert to uint8 for FFCV compatibility
        ])

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        image = self.transform(image)  # Apply transformation
        image = image.permute(1, 2, 0).numpy()  # Convert to HWC format for FFCV
        return image, label


class TransformedCIFAR10(Dataset):
    def __init__(self, train=True):
        self.cifar10 = torchvision.datasets.CIFAR10('/tmp', train=train, download=True)
        
        # Define the transformation: keep 32x32, convert to tensor, then to uint8
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Resize to 32x32
            transforms.ToTensor(),  # Convert PIL image to [0,1] tensor
            transforms.Lambda(lambda x: (x * 255).byte())  # Convert to uint8
        ])

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        image, label = self.cifar10[idx]
        image = self.transform(image)  # Apply transformation
        image = image.permute(1, 2, 0).numpy()  # Convert to HWC format for FFCV
        return image, label


class TransformedSVHN(Dataset):
    def __init__(self, train=True):
        self.svhn = torchvision.datasets.SVHN('/tmp', split='train' if train else 'test', download=True)
        
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Resize to 28x28
            transforms.ToTensor(),  # Convert PIL or ndarray image to [0, 1] tensor
            transforms.Lambda(lambda x: (x * 255).byte())  # Convert to uint8
        ])

    def __len__(self):
        return len(self.svhn)

    def __getitem__(self, idx):
        image, label = self.svhn[idx]
        image = self.transform(image)  # Apply transformation
        image = image.permute(1, 2, 0).numpy()  # Convert to HWC format for FFCV
        return image, label


@param('data.train_dataset')
@param('data.val_dataset')
def main(train_dataset, val_dataset):
    os.makedirs(os.path.dirname(train_dataset), exist_ok=True)
    os.makedirs(os.path.dirname(val_dataset), exist_ok=True)
    # train_indices, test_indices = None, None
    if "mnist" in train_dataset:
        datasets = {
            'train': TransformedMNIST(train=True),
            'test': TransformedMNIST(train=False),
            }
    elif "svhn" in train_dataset:
        datasets = {
            'train': TransformedSVHN(train=True),
            'test': TransformedSVHN(train=False),
            }
    elif "cifar10" in train_dataset:  # 8 classes in total
        datasets = {
            'train': TransformedCIFAR10(train=True),
            'test': TransformedCIFAR10(train=False),
            }
 
    for (name, ds) in datasets.items():
        # indices = np.argsort(np.array(ds.targets))
        path = train_dataset if name == 'train' else val_dataset
        writer = DatasetWriter(path, {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)


if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    main()