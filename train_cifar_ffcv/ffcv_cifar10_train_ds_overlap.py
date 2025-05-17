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
import ffcv
from argparse import ArgumentParser
from typing import List
import time, random, copy, os, json
from datetime import datetime
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision

from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage, RandomBrightness, RandomContrast, RandomSaturation
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

import sys
sys.path.append("../data_splitting")
from data_splitting import split_dataset

Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', default=0.5),
    epochs=Param(int, 'Number of epochs to run for', default=50),
    lr_peak_epoch=Param(int, 'Peak epoch for cyclic lr', default=10),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.1),
    num_workers=Param(int, 'The number of workers', default=8),
    lr_tta=Param(bool, 'Test time augmentation by averaging with horizontally flipped version', default=False), 
    seed=Param(int, 'Random seed', default=0)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', default="./cifar10_ffcv_data/train.beton"),
    val_dataset=Param(str, '.dat file to use for validation', default="./cifar10_ffcv_data/val.beton"),
    # n_split=Param(int, 'Number of datasets', default=2),
    class_frac_overlap=Param(float, 'Frac of data shared by all datasets', required=True),
    variable_ds_size=Param(bool, 'Whether the training dataset size can be allowed to vary', default=False), 
)

Section('output', 'Output related').params(
    output_root=Param(str, 'Root for saving', required=True),
)


@param('data.train_dataset')
@param('data.val_dataset')
@param('training.batch_size')
@param('training.num_workers')
def make_dataloaders(train_indices, always_rand_order=False, torch_float32=False, train_dataset=None, val_dataset=None, batch_size=None, num_workers=None):
    paths = {
        'train': train_dataset,
        'test': val_dataset

    }

    start_time = time.time()
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    loaders = {}

    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(ch.device('cuda:0')), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomBrightness(0.2, 0.3), 
                RandomContrast(0.2, 0.3), 
                RandomSaturation(0.3, 0.3),
                RandomTranslate(padding=8, fill=tuple(map(int, CIFAR_MEAN))),
                Cutout(10, tuple(map(int, CIFAR_MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice(ch.device('cuda:0'), non_blocking=True),
            ToTorchImage(),
            Convert(ch.float32 if torch_float32 else ch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        
        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL
        if always_rand_order:
            ordering = OrderOption.RANDOM
        loaders[name] = Loader(paths[name], batch_size=batch_size, num_workers=num_workers,
                               order=ordering, drop_last=(name == 'train'), indices=train_indices if name == "train" else None,
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loaders, start_time


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


def construct_resnet():
    num_classes = 10
    model = torchvision.models.resnet18()
    model.conv1 = ch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = ch.nn.Identity()
    model.fc = ch.nn.Linear(model.fc.in_features, num_classes)
    return model.to(memory_format=ch.channels_last).cuda()


@param('training.lr')
@param('training.epochs')
@param('training.momentum')
@param('training.weight_decay')
@param('training.label_smoothing')
@param('training.lr_peak_epoch')
def train(model, loaders, lr=None, epochs=None, label_smoothing=None,
          momentum=None, weight_decay=None, lr_peak_epoch=None):
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loaders['train'])
    # Cyclic LR with single triangle
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    for _ in tqdm(range(epochs)):
        for ims, labs in loaders['train']:
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
    return


@param('training.lr_tta')
def evaluate(model, loaders, log_dir, lr_tta=False):
    model.eval()
    with ch.no_grad():
        for name in ['train', 'test']:
            total_correct, total_num = 0., 0.
            for ims, labs in tqdm(loaders[name]):
                with autocast():
                    out = model(ims)
                    if lr_tta:
                        out += model(ims.flip(-1))
                    total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                    total_num += ims.shape[0]
            print(f'{name} accuracy: {total_correct / total_num * 100:.1f}%')
            with open(os.path.join(log_dir, "log.txt"), "a") as f:
                f.write(f'{name} accuracy: {total_correct / total_num * 100:.1f}%\n')
    return


if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    ch.manual_seed(config['training.seed'])
    random.seed(config['training.seed'])
    np.random.seed(config['training.seed'])
    # Also loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # log_base_path = os.path.join(f"./log_dir/same_dataset_len/seed_{config['training.seed']:d}", 
    #                              formatted_datetime + "_" + f"{config['data.class_frac_overlap']:g}")
    log_base_path = os.path.join(config['output.output_root'], f"seed_{config['training.seed']:d}", 
                                 formatted_datetime + "_desired_" + f"{config['data.class_frac_overlap']:g}")
    os.makedirs(log_base_path, exist_ok=True)
    with open(os.path.join(log_base_path, "args.json"), "w") as f:
        config.summary(f)
    
    indices_dict = make_indices_ds(ds_size=50_000, 
                                log_base_path=log_base_path, 
                                class_frac_overlap=config["data.class_frac_overlap"], 
                                num_classes=10, variable_ds_size=config["data.variable_ds_size"])
    for split in indices_dict.keys():
        split_log_path = os.path.join(log_base_path, f"split_{split}")
        os.makedirs(split_log_path, exist_ok=True)
        loaders, start_time = make_dataloaders(indices_dict[split])
        model = construct_resnet()
        train(model, loaders)
        print(f'Total time: {time.time() - start_time:.5f}')
        evaluate(model, loaders, split_log_path)
        ch.save(model.state_dict(), os.path.join(split_log_path, "checkpoint.pth"))