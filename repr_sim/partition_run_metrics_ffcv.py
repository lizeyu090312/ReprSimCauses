import ffcv 

import sys
sys.path.append("../")
from train_task_only.ffcv_partition1_train import make_dataloaders as make_dataloaders_partiton_ffcv
## Only need the images, so we load from an arbitrary partition

import platonic
from metrics import AlignmentMetrics

import torch, torchvision, os, tqdm, argparse, sys, itertools, datetime, copy
from collections import Counter, defaultdict
import numpy as np

from torch.utils.data import DataLoader
from torch import nn
import torchvision.transforms.v2 as v2
from torchvision.models.feature_extraction import create_feature_extractor


def get_args_parser():
    parser = argparse.ArgumentParser("platonic", add_help=False)

    parser.add_argument("--log_base_dir", type=str, required=True)
    
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size")
    parser.add_argument("--metrics", nargs="+", type=str, default=AlignmentMetrics.SUPPORTED_METRICS, help="List of metrics to compute")
    return parser.parse_args()


def construct_resnet(device, n_classes):
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.eval()
    return model.to(device)


###### loading for combined_run_metrics_ffcv
def load_dl_combined_run_metrics_ffcv():
    batch_size = 512
    num_workers = 8

    partition3 = {
        "all0":
        make_dataloaders_partiton_ffcv(always_rand_order=True, torch_float32=True, 
            dataset_base_dir="../train_task_only/ffcv_all_partitions/partition3/all", 
            batch_size=batch_size, num_workers=num_workers, )[0]['test'], 
        "all1":
        make_dataloaders_partiton_ffcv(always_rand_order=True, torch_float32=True, 
            dataset_base_dir="../train_task_only/ffcv_all_partitions/partition3/all", 
            batch_size=batch_size, num_workers=num_workers, )[0]['test']}
    partition2 = {}
    ds_names_partition2 = ["shape_digit", "shape_color", "digit_color"]
    for ds_name in ds_names_partition2:
        partition2[ds_name] = make_dataloaders_partiton_ffcv(always_rand_order=True, torch_float32=True, 
            dataset_base_dir=f"../train_task_only/ffcv_all_partitions/partition2/{ds_name}", 
            batch_size=batch_size, num_workers=num_workers, )[0]['test']
    partition1 = {}
    ds_names_partition1 = ["shape", "digit"]
    for ds_name in ds_names_partition1:
        partition1[ds_name] = make_dataloaders_partiton_ffcv(always_rand_order=True, torch_float32=True, 
            dataset_base_dir=f"../train_task_only/ffcv_all_partitions/partition1/{ds_name}", 
            batch_size=batch_size, num_workers=num_workers, )[0]['test']
    
    partition_metric = make_dataloaders_partiton_ffcv(always_rand_order=True, torch_float32=True, 
            dataset_base_dir="../train_task_only/ffcv_all_partitions/partition3/all", 
            batch_size=batch_size, num_workers=num_workers, )[0]['test'] 
    
    digit_dl = make_dataloaders_partiton_ffcv(always_rand_order=True, torch_float32=True, 
        dataset_base_dir="../train_task_only/digit_shape_datasets_ffcv/digit_only", 
                     batch_size=batch_size, num_workers=num_workers, )[0]['test']
    shape_dl = make_dataloaders_partiton_ffcv(always_rand_order=True, torch_float32=True, 
        dataset_base_dir="../train_task_only/digit_shape_datasets_ffcv/shape_only", 
                     batch_size=batch_size, num_workers=num_workers, )[0]['test']
    
    cifar10_dl = make_dataloaders_partiton_ffcv(always_rand_order=True, torch_float32=True, 
        dataset_base_dir="../train_task_only/ood_dataset/cifar10",
                     batch_size=batch_size, num_workers=num_workers, )[0]['test']
    mnist_dl = make_dataloaders_partiton_ffcv(always_rand_order=True, torch_float32=True, 
        dataset_base_dir="../train_task_only/ood_dataset/mnist",
                     batch_size=batch_size, num_workers=num_workers, )[0]['test']
    svhn_dl = make_dataloaders_partiton_ffcv(always_rand_order=True, torch_float32=True, 
        dataset_base_dir="../train_task_only/ood_dataset/svhn",
                     batch_size=batch_size, num_workers=num_workers, )[0]['test']

    ret_partial = {
        "partition_metric": partition_metric,
        
        "mnist":            mnist_dl, 
        "svhn":             svhn_dl, 
        "cifar10":          cifar10_dl, 

        "shape":            shape_dl, 
        "digit":            digit_dl,
    }
    for kaaa, vaaa in partition1.items():
        ret_partial[f"partition1/{kaaa}"] = vaaa
    for kaaa, vaaa in partition2.items():
        ret_partial[f"partition2/{kaaa}"] = vaaa
    for kaaa, vaaa in partition3.items():
        ret_partial[f"partition3/{kaaa}"] = vaaa
    
    return ret_partial


@torch.no_grad()
def compute_metric(vision_model1, vision_model2, dl, device, metrics, max_num_batches=40):
    if dl is None: 
        return {metric:-1 for metric in metrics}
    platonic_metric = platonic.Alignment(
        dataset="cifar10", # <--- this is the dataset 
        subset=None,    # <--- this is the subset
        # models=["openllama_7b", "llama_65b"], 
        models=["A", "B"], transform=None, device=device, dtype=torch.float32
        ) # you can also pass in device and dtype as arguments

    lvm_feats1, lvm_feats2 = [], []
    for i, (ims, _) in enumerate(dl):
        ims = ims.to(device)
        with torch.no_grad():
            lvm_output1 = vision_model1(ims)
            lvm_output2 = vision_model2(ims)  # {key1: torch.tensor of size (bsz, ...), etc}
        
        lvm_feats1.append(torch.stack([v for v in lvm_output1.values()]).permute(1, 0, 2))
        lvm_feats2.append(torch.stack([v for v in lvm_output2.values()]).permute(1, 0, 2))
        if i > max_num_batches:
            break
    # compute score 
    lvm_feats1, lvm_feats2 = torch.cat(lvm_feats1), torch.cat(lvm_feats2)
    metric_to_score = dict()
    for metric in metrics:
        metric_to_score[metric] = platonic_metric.score(lvm_feats1, lvm_feats2, metric=metric, topk=10, normalize=True)['A'][0]
    return metric_to_score


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_correct, total_num = 0., 0.
    for ims, labs in loader:
        with torch.autocast('cuda'):
            out = model(ims)
            total_correct += out.argmax(1).eq(labs).sum().cpu().item()
            total_num += ims.shape[0]
    return total_correct / total_num


if __name__ == "__main__":
    args = get_args_parser()
    device = torch.device("cuda:0")
    
    print("Preparing datasets")
    data_loaders_dict = load_dl_combined_run_metrics_ffcv()
    print("Done preparing datasets")
    
    with open(os.path.join(args.log_base_dir, f"platonic_dataset_sanity_check.txt"), "a") as fptr:
        for ds_name, temp_dl in data_loaders_dict.items():
            lbl_list = list()
            for _, lbl in temp_dl:
                lbl_list.extend(lbl.cpu().tolist())
            lbl_set = set(lbl_list)
            labels_counter = Counter(lbl_list)
            
            fptr.write(f"{ds_name}_labels_counter\n")
            fptr.write(str(labels_counter) + "\n")
            fptr.write(f"{ds_name}_unique_labels\n")
            fptr.write(f"{sorted(list(lbl_set))}\n\n")
    
    # all_train_sets = [k for k in data_loaders_dict.keys() if "partition" in k]
    
    path_to_acc = os.path.join(args.log_base_dir, f"acc_ffcv.csv")
    f = open(path_to_acc, "a")
    f.write("seed,partition,model,acc\n")
    print(f"Computing accuracy")
    partition1_train_ds_name_to_n_class = {"partition1/shape":8, "partition1/digit":10,}
    partition2_train_ds_name_to_n_class = {"partition2/shape_digit":80, "partition2/shape_color":80, "partition2/digit_color":100}
    partition3_train_ds_name_to_n_class = {"partition3/all0":800, "partition3/all1":800,}
    train_ds_name_to_n_class = {**partition1_train_ds_name_to_n_class, **partition2_train_ds_name_to_n_class, **partition3_train_ds_name_to_n_class}
    for seed in range(10):
        for partition_ds_name, n_class in train_ds_name_to_n_class.items():
            partition, ds_name = partition_ds_name.split("/")
            path_to_model = os.path.join(args.log_base_dir, partition, f"seed_{seed:d}/{ds_name}", "checkpoint.pth")
            model = construct_resnet(device, n_class).eval()
            model.load_state_dict(torch.load(path_to_model, map_location=device))
            acc = evaluate(model, data_loaders_dict[partition_ds_name])
            f.write(f"{seed},{partition},{ds_name},{acc:.6f}\n")
    f.close()
    print(f"Done computing accuracy")
    # only need to evaluate with one of the many partition datasets since they all contain same images
    # removes all partitionx/yyy datasets, keeps partition_metric
    data_loaders_dict_trunc = {k:copy.deepcopy(v) for k,v in data_loaders_dict.items() if "/" not in k}
    del data_loaders_dict
    
    partition_to_ds_list = defaultdict(list)
    for partition_ds_name in train_ds_name_to_n_class.keys():
        partition, ds_name = partition_ds_name.split("/")
        partition_to_ds_list[partition].append(ds_name)

    path_to_cka_res = os.path.join(args.log_base_dir, f"platonic_ffcv.csv")
    f = open(path_to_cka_res, "a")
    base_str = f"seed,metric,partition,model0,model1"
    for ds_name in sorted(list(data_loaders_dict_trunc.keys())):
        base_str += f",{ds_name}"
    base_str += "\n"
    f.write(base_str)
    f.close()
    for seed in tqdm.tqdm(range(10)):
        for partition, ds_names in partition_to_ds_list.items():
            all_pairs = [x for x in itertools.combinations(ds_names, 2)]
            for m0, m1 in all_pairs:
                path_to_0 = os.path.join(args.log_base_dir, partition, f"seed_{seed:d}/{m0}", "checkpoint.pth")
                path_to_1 = os.path.join(args.log_base_dir, partition, f"seed_{seed:d}/{m1}", "checkpoint.pth")

                first = construct_resnet(device, train_ds_name_to_n_class[f"{partition}/{m0}"]).eval()
                first.load_state_dict(torch.load(path_to_0, map_location=device))
                second = construct_resnet(device, train_ds_name_to_n_class[f"{partition}/{m1}"]).eval()
                second.load_state_dict(torch.load(path_to_1, map_location=device))
                
                return_nodes = ["flatten"]
                vision_model1 = create_feature_extractor(first, return_nodes=return_nodes)
                vision_model2 = create_feature_extractor(second, return_nodes=return_nodes)
                test_data_name_to_metric_dict = dict()
                
                for ds_name, dl in data_loaders_dict_trunc.items():
                    test_data_name_to_metric_dict[ds_name] = compute_metric(vision_model1, vision_model2, dl, device, args.metrics)
                
                f = open(path_to_cka_res, "a")
                for metric in args.metrics:
                    base_str = f"{seed:d},{metric},{partition},{m0},{m1}"  # only contains ds_name=partition_metric and other OOD datasets
                    for ds_name in sorted(list(data_loaders_dict_trunc.keys())):
                        base_str += f",{test_data_name_to_metric_dict[ds_name][metric]:.6f}"
                    base_str += "\n"
                    f.write(base_str)
                f.close()
                print(f"{datetime.datetime.now()} -- done with seed={seed}, partition={partition}, models=[{m0},{m1}]")
