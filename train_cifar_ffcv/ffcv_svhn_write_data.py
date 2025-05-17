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

class TransformedSVHN(Dataset):
    def __init__(self, train=True):
        self.svhn = torchvision.datasets.SVHN('/tmp', split='train' if train else 'test', download=True)
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to 28x28
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


Section('data', 'arguments to give the writer').params(
    train_dataset=Param(str, 'Where to write the new dataset', default="./svhn_ffcv_data/train.beton"),
    val_dataset=Param(str, 'Where to write the new dataset', default="./svhn_ffcv_data/val.beton"),
)

@param('data.train_dataset')
@param('data.val_dataset')
def main(train_dataset, val_dataset):
    os.makedirs(os.path.dirname(train_dataset), exist_ok=True)
    datasets = {
        'train': TransformedSVHN(train=True),
        'test': TransformedSVHN(train=False),
        }

    for (name, ds) in datasets.items():
        # indices = np.argsort(np.array(ds.targets))
        path = train_dataset if name == 'train' else val_dataset
        writer = DatasetWriter(path, {
            'image': RGBImageField(),
            'label': IntField()
        })
        # writer.from_indexed_dataset(ds, indices=indices)
        writer.from_indexed_dataset(ds)


if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    main()