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

from PIL import Image
from tinyimagenet_path import PATH_TO_TINYIMAGNET

Section('data', 'arguments to give the writer').params(
    train_dataset=Param(str, 'Where to write the new dataset', default="./tinyimagenet_ffcv_data/train.beton"),
    val_dataset=Param(str, 'Where to write the new dataset', default="./tinyimagenet_ffcv_data/val.beton"),
)


def loader(path: str):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
        
@param('data.train_dataset')
@param('data.val_dataset')
def main(train_dataset, val_dataset):
    os.makedirs(os.path.dirname(train_dataset), exist_ok=True)
    datasets = {
        'train': torchvision.datasets.DatasetFolder(f"{PATH_TO_TINYIMAGNET}/train", 
                                                        transform=None, extensions="jpeg", 
                                                        loader=loader),
        'test': torchvision.datasets.DatasetFolder(f"{PATH_TO_TINYIMAGNET}/val/images", 
                                                        transform=None, extensions="jpeg", 
                                                        loader=loader),
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
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    main()