import ffcv
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

from argparse import ArgumentParser
from typing import List
import time, random, copy, os, json
from datetime import datetime
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter

import torch as ch
from torch.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision

from fastargs import get_current_config, Param, Section
from fastargs.decorators import param

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
    seed=Param(int, 'Random seed', default=0),
)

Section('output', 'Output related').params(
    output_root=Param(str, 'Root for saving', default="./log_dir_partition/partition1"),
)

@param('training.batch_size')
@param('training.num_workers')
def make_dataloaders(dataset_base_dir, always_rand_order=False, torch_float32=False, batch_size=None, num_workers=None,):
    paths = {
        'train': os.path.join(dataset_base_dir, "train.beton"),
        'test': os.path.join(dataset_base_dir, "val.beton"),
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
                RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                Cutout(4, tuple(map(int, CIFAR_MEAN))),
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
                               order=ordering, drop_last=False, indices=None,
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loaders, start_time


def construct_resnet(num_classes):
    model = torchvision.models.resnet18()
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
    scaler = GradScaler("cuda")
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    for _ in tqdm(range(epochs)):
        for ims, labs in loaders['train']:
            opt.zero_grad(set_to_none=True)
            with autocast("cuda"):
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
                with autocast("cuda"):
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
    parser = ArgumentParser(description='ffcv')
    config.augment_argparse(parser)
    ch.manual_seed(config['training.seed'])
    random.seed(config['training.seed'])
    np.random.seed(config['training.seed'])
    # Also loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    ds_name_to_n_class = {"shape":8, "digit":10,}
    for ds_name in ds_name_to_n_class.keys():
        log_base_path = os.path.join(config['output.output_root'], f"seed_{config['training.seed']}", ds_name)    
        
        os.makedirs(log_base_path, exist_ok=True)
        with open(os.path.join(log_base_path, "args.json"), "w") as f:
            config.summary(f)
        
        loaders, start_time = make_dataloaders(dataset_base_dir=f"./ffcv_all_partitions/partition1/{ds_name}")
        
        for name, loader in loaders.items():
            lbl_list = list()
            for _, lbl in loader:
                lbl_list.extend(lbl.cpu().tolist())
            lbl_set = set(lbl_list)
            labels_counter = Counter(lbl_list)
            with open(os.path.join(log_base_path, "data_log.txt"), "a") as fptr:
                fptr.write(f"combined_{ds_name}_{name}_labels_counter\n")
                fptr.write(str(labels_counter) + "\n")
                fptr.write(f"combined_{ds_name}_{name}_unique_labels\n")
                fptr.write(f"{sorted(list(lbl_set))}\n")
        
        model = construct_resnet(ds_name_to_n_class[ds_name])
        train(model, loaders)
        print(f'Total time: {time.time() - start_time:.5f}')
        evaluate(model, loaders, log_base_path)
        ch.save(model.state_dict(), os.path.join(log_base_path, "checkpoint.pth"))
        