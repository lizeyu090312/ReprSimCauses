import ffcv
import os

from argparse import ArgumentParser
import torchvision

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# Define the section for fastargs
Section('data', 'arguments to give the writer').params(
    train_dataset=Param(str, 'Where to write the new dataset', default="./mnist_ffcv_data/train.beton"),
    val_dataset=Param(str, 'Where to write the new dataset', default="./mnist_ffcv_data/val.beton"),
)

# Custom dataset wrapper to transform MNIST images
class TransformedMNIST(Dataset):
    def __init__(self, train=True):
        self.mnist = torchvision.datasets.MNIST('/tmp', train=train, download=True)
        
        # Define the transformation: resize to 32x32 and convert to 3 channels
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to 32x32
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

@param('data.train_dataset')
@param('data.val_dataset')
def main(train_dataset, val_dataset):
    os.makedirs(os.path.dirname(train_dataset), exist_ok=True)
    datasets = {
        'train': TransformedMNIST(train=True),
        'test': TransformedMNIST(train=False)
    }

    for name, ds in datasets.items():
        path = train_dataset if name == 'train' else val_dataset
        writer = DatasetWriter(path, {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    main()
