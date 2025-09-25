import ffcv
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage, RandomBrightness, RandomContrast, RandomSaturation
from ffcv.transforms.common import Squeeze

from argparse import ArgumentParser
from typing import List
import time, random, copy, os, json, timm, gc
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
    epochs=Param(int, 'Number of epochs to run for', default=10),
    lr_peak_epoch=Param(int, 'Peak epoch for cyclic lr', default=1),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.1),
    num_workers=Param(int, 'The number of workers', default=8),
    lr_tta=Param(bool, 'Test time augmentation by averaging with horizontally flipped version', default=False), 
    seed=Param(int, 'Random seed', default=0)
)

Section('output', 'Output related').params(
    output_root=Param(str, 'Root for saving', default="./log_dir_partition/partition3"),
)

Section('model', 'Model selection').params(
    model_name=Param(str, 'Some resnets/ViTs (from timm)', default='resnet18'),
)

@param('training.batch_size')
@param('training.num_workers')
def make_dataloaders(dataset_base_dir, always_rand_order=False, torch_float32=True, batch_size=None, num_workers=None,):
    paths = {
        'train': os.path.join(dataset_base_dir, "train.beton"),
        'test': os.path.join(dataset_base_dir, "val.beton"),
    }

    start_time = time.time()
    MEAN = [127, 127, 127]
    STD = [127, 127, 127]
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
                RandomTranslate(padding=8, fill=tuple(map(int, MEAN))),
                Cutout(10, tuple(map(int, MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice(ch.device('cuda:0'), non_blocking=True),
            ToTorchImage(),
            torchvision.transforms.Resize(224),
            Convert(ch.float32 if torch_float32 else ch.float16),
            torchvision.transforms.Normalize(MEAN, STD),
        ])

        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL
        if always_rand_order:
            ordering = OrderOption.RANDOM
        loaders[name] = Loader(paths[name], batch_size=batch_size, num_workers=num_workers,
                               order=ordering, drop_last=False, indices=None,
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loaders, start_time


def build_model(model_name: str, num_classes: int, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model.to(memory_format=ch.channels_last).cuda()


@param('training.lr')
@param('training.epochs')
@param('training.momentum')
@param('training.weight_decay')
@param('training.label_smoothing')
@param('training.lr_peak_epoch')
@param('model.model_name')
def train(model, loaders, lr=None, epochs=None, label_smoothing=None,
          momentum=None, weight_decay=None, lr_peak_epoch=None, model_name=None):
    if 'vit' in model_name:
        opt = ch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif 'resnet' in model_name:
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
    
    ds_name_to_n_class = {"all0":800, "all1":800}
    for ds_name in ds_name_to_n_class.keys():
        log_base_path = os.path.join(config['output.output_root'], config['model.model_name'], "partition3", 
                                     f"seed_{config['training.seed']}", ds_name)
        
        os.makedirs(log_base_path, exist_ok=True)
        with open(os.path.join(log_base_path, "args.json"), "w") as f:
            config.summary(f)
        
        loaders, start_time = make_dataloaders(dataset_base_dir=f"./ffcv_all_partitions/partition3/{ds_name[0:-1]}")  # remove digit at the end of ds_name
        
        # for name, loader in loaders.items():
        #     lbl_list = list()
        #     for _, lbl in loader:
        #         lbl_list.extend(lbl.cpu().tolist())
        #     lbl_set = set(lbl_list)
        #     labels_counter = Counter(lbl_list)
        #     with open(os.path.join(log_base_path, "data_log.txt"), "a") as fptr:
        #         fptr.write(f"combined_{ds_name}_{name}_labels_counter\n")
        #         fptr.write(str(labels_counter) + "\n")
        #         fptr.write(f"combined_{ds_name}_{name}_unique_labels\n")
        #         fptr.write(f"{sorted(list(lbl_set))}\n")
        
        model = build_model(config['model.model_name'], num_classes=ds_name_to_n_class[ds_name], pretrained=True)
        train(model, loaders)
        print(f'Total time: {time.time() - start_time:.5f}')
        evaluate(model, loaders, log_base_path)
        ch.save(model.state_dict(), os.path.join(log_base_path, "checkpoint.pth"))
        with open(os.path.join(log_base_path, "model_info.txt"), "w") as f:
            f.write(f"model_name:{config['model.model_name']}\n")
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            f.write(f"n_param_million:{n_params:.6f}\n")
        model = model.to('cpu')
        del model
        ch.cuda.empty_cache()
        gc.collect()