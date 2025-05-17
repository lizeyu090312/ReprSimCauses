import ffcv
from argparse import ArgumentParser
from typing import List, Tuple
import time, random, copy, os, json, csv
from datetime import datetime
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter

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
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage, RandomBrightness, RandomContrast, RandomSaturation
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
from PIL import Image 

from dataclasses import replace

import sys
sys.path.append("../data_splitting")
from data_splitting import split_dataset

from tinyimagenet_path import PATH_TO_TINYIMAGNET

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
    train_dataset=Param(str, '.dat file to use for training', default="./tinyimagenet_ffcv_data/train.beton"),
    val_dataset=Param(str, '.dat file to use for validation', default="./tinyimagenet_ffcv_data/val.beton"),
    # n_split=Param(int, 'Number of datasets', default=2),
    class_frac_overlap=Param(float, 'Frac of data shared by all datasets', required=True),
    variable_ds_size=Param(bool, 'Whether the training dataset size can be allowed to vary', default=False), 
)

Section('output', 'Output related').params(
    output_root=Param(str, 'Root for saving', required=True),
)

def loader(path: str):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class MapClass(Operation):
    def __init__(self, orig_to_new_class_dict):
        max_class = max(orig_to_new_class_dict.keys()) + 1  # Ensure array covers all indices
        self.lookup_table = np.zeros(max_class) - 1  # Default mapping (identity)
        for orig_class, new_class in orig_to_new_class_dict.items():
            self.lookup_table[orig_class] = new_class
        
    # Return the code to run this operation
    def generate_code(self):
        parallel_range = Compiler.get_iterator()
        mapping = self.lookup_table

        def map_class(labels, dst):
            for i in parallel_range(labels.shape[0]):
                labels[i] = mapping[labels[i]]
            return labels
        
        map_class.is_parallel = True
        return map_class
    
    def declare_state_and_memory(self, previous_state):
        new_shape = previous_state.shape
        return previous_state, AllocationQuery(new_shape, previous_state.dtype)

@param('data.train_dataset')
@param('data.val_dataset')
@param('training.batch_size')
@param('training.num_workers')
def make_dataloaders(train_indices, test_indices, orig_to_new_class_dict, always_rand_order=False, torch_float32=False, 
                     train_dataset=None, val_dataset=None, batch_size=None, num_workers=None):
    paths = {
        'train': train_dataset,
        'test': val_dataset

    }

    start_time = time.time()
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    loaders = {}

    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), MapClass(orig_to_new_class_dict), 
                                           ToTensor(), ToDevice(ch.device('cuda:0')), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomBrightness(0.2, 0.3), 
                RandomContrast(0.2, 0.3), 
                RandomSaturation(0.3, 0.3),
                RandomTranslate(padding=12, fill=tuple(map(int, CIFAR_MEAN))),
                Cutout(16, tuple(map(int, CIFAR_MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice(ch.device('cuda:0'), non_blocking=True),
            ToTorchImage(),
            Convert(ch.float32 if torch_float32 else ch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        if always_rand_order:
            ordering = OrderOption.RANDOM
        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name], batch_size=batch_size, num_workers=num_workers,
                               order=ordering, drop_last=(name == 'train'), indices=train_indices if name == "train" else test_indices,
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loaders, start_time


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
    
    dataset = torchvision.datasets.DatasetFolder(f"{PATH_TO_TINYIMAGNET}/train", 
                                                        transform=None, extensions="jpeg", 
                                                        loader=loader)
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
    for idx, (_, label) in enumerate(torchvision.datasets.DatasetFolder(f"{PATH_TO_TINYIMAGNET}/val/images", 
                                                        transform=None, extensions="jpeg", 
                                                        loader=loader)):
        class_to_indices[label].append(idx)
        
    path_to_log_dir = os.path.join(log_base_path, "split_indices")
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


def construct_resnet(num_classes_per_split):
    num_classes = int(num_classes_per_split)
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


def test_loaders_for_correct_split(log_base_path, loaders, orig_to_new_class, split):
    new_class_to_orig = {v:k for k,v in orig_to_new_class[split].items()}
    path_to_log = os.path.join(log_base_path, "indices_verification")
    os.makedirs(path_to_log, exist_ok=True)
    for train_or_test in ['train', 'test']:
        loader = loaders[train_or_test]
        labels_set = set()
        old_class_to_num_data = dict()
        for _, labs in loader:
            labs = [new_class_to_orig[int(l)] for l in labs.detach().cpu().flatten().tolist()]
            labels_set.update(labs)
            c = Counter(labs)
            for label, count in c.items():  # here, label is old label
                if label not in old_class_to_num_data.keys():
                    old_class_to_num_data[label] = count
                else:
                    old_class_to_num_data[label] += count
        labels_set_is_ok = len(set(new_class_to_orig.values()) ^ labels_set) == 0
        # ^ computes elements not in one of the sets, so labels found in loader 
        # should completely match labels in orig class of new_class_to_orig (which contains the mapping). 
        old_class_to_num_data_lab_is_ok = len(set(old_class_to_num_data.keys()) ^ set(new_class_to_orig.values())) == 0
        # Same as above. Labels in the frequency dict should overlap completely with orig classes
        counts_temp = np.array(list(old_class_to_num_data.values()))
        old_class_to_num_data_count_is_ok = np.allclose(counts_temp, counts_temp[0])
        # counts_temp is array of #datapoints for each class. np allclose checks if every value in 
        # counts_temp is very close.
        with open(os.path.join(path_to_log, f"split_{split}_{train_or_test}_labels_set.txt"), "w") as f:
            f.write(str(labels_set) + "\n")
            f.write(f"labels_set_is_ok,{labels_set_is_ok}\n")
        with open(os.path.join(path_to_log, f"split_{split}_{train_or_test}_old_class_to_num_data.txt"), "w") as f:
            f.write(str(old_class_to_num_data) + "\n")
            f.write(f"old_class_to_num_data_lab_is_ok,{old_class_to_num_data_lab_is_ok}\n")
            f.write(f"old_class_to_num_data_count_is_ok,{old_class_to_num_data_count_is_ok}\n")
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

    log_base_path = os.path.join(config['output.output_root'], f"seed_{config['training.seed']:d}", 
                                 formatted_datetime + "_desired_" + f"{config['data.class_frac_overlap']:g}")
    
    os.makedirs(log_base_path, exist_ok=True)
    with open(os.path.join(log_base_path, "args.json"), "w") as f:
        config.summary(f)
    indices_dict_train, class_dict, ret_orig_to_new_class_dict = \
        make_indices_task_train(ds_size=None, log_base_path=log_base_path, 
                          vertical_frac_overlap=config["data.class_frac_overlap"], 
                          num_classes=None, variable_ds_size=config["data.variable_ds_size"])
    indices_dict_test = make_indices_task_test(ds_size=None, log_base_path=log_base_path, class_dict=class_dict, num_classes=None)
    for split in indices_dict_train.keys():
        split_log_path = os.path.join(log_base_path, f"split_{split}")
        os.makedirs(split_log_path, exist_ok=True)
        loaders, start_time = make_dataloaders(indices_dict_train[split], indices_dict_test[split], ret_orig_to_new_class_dict[split])
        test_loaders_for_correct_split(log_base_path, loaders, orig_to_new_class=ret_orig_to_new_class_dict, split=split)
        model = construct_resnet(len(class_dict[split]))
        train(model, loaders)
        print(f'Total time: {time.time() - start_time:.5f}')
        evaluate(model, loaders, split_log_path)
        ch.save(model.state_dict(), os.path.join(split_log_path, "checkpoint.pth"))