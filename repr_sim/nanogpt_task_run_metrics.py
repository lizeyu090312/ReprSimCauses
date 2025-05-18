import platonic
from metrics import AlignmentMetrics

import torch, torchvision, os, tqdm, argparse, json, tiktoken, logging, sys, random
import numpy as np

from torch.utils.data import DataLoader
from torch import nn
import torchvision.transforms.v2 as v2
from torchvision.models.feature_extraction import create_feature_extractor

from datasets import load_dataset

from torch.utils.data import Dataset

sys.path.append("../")
from train_nanoGPT.model import GPTConfig, GPT

DATA_DIR_BASE = "/path/to/ReprSimCauses/train_nanoGPT/data"

def get_args_parser():
    parser = argparse.ArgumentParser("platonic_gpt2", add_help=False)

    parser.add_argument("--log_base_dir", type=str, required=True)
    
    parser.add_argument('--datasets', nargs='+', type=str, 
                        default=["tinycodes", "tinystories", "tinytextbooks", "tinywebtext", "wikitext", "shakespeare_all"], 
                        help='Which datasets to use when computing metrics')
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size")
    parser.add_argument("--metrics", nargs="+", type=str, default=AlignmentMetrics.SUPPORTED_METRICS, help="List of metrics to compute")
    return parser.parse_args()


class RandomBinDataset(Dataset):
    def __init__(self, avail_datasets, block_size, batch_size, device, total_len=10000000):
        # total_len=10000000 is a completely arbitrary choice
        self.total_len = total_len
        self.avail_datasets = avail_datasets
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
    
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        x, y = [], []
        for data_idx in range(self.batch_size):
            data = np.memmap(os.path.join(DATA_DIR_BASE, random.choice(self.avail_datasets), f'val.bin'), dtype=np.uint16, mode='r')
            i = torch.randint(len(data) - self.block_size, (1,))[0]
            x_this = torch.from_numpy((data[i:i+self.block_size]).astype(np.int64))
            y_this = torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64))
            x.append(x_this)
            y.append(y_this)
        x, y = torch.stack(x), torch.stack(y)
        return x.pin_memory().to(self.device, non_blocking=True), \
               y.pin_memory().to(self.device, non_blocking=True)


def construct_loaded_gpt2(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

@torch.no_grad()
def compute_metric(first, second, dl, device, metrics, max_num_batches=80):
    if dl is None: 
        return {metric:-1 for metric in metrics}
    platonic_metric = platonic.Alignment(
        dataset="gpt2_model",
        subset=None,
        models=["A", "B"], transform=None, device=device, dtype=torch.float32,
    )

    lvm_feats1, lvm_feats2 = [], []
    for i, (inp_txt, _) in enumerate(dl):
        inp_txt = inp_txt.to(device)
        with torch.no_grad():
            _ = first(inp_txt, output_last_hidden_state=True)
            _ = second(inp_txt, output_last_hidden_state=True)
            lvm_output1 = first.last_hidden_state
            lvm_output2 = second.last_hidden_state

        lvm_feats1.append(torch.stack([lvm_output1.mean(dim=1)]).permute(1, 0, 2))  # from layer, bsz, d to bsz, layer, d
        lvm_feats2.append(torch.stack([lvm_output2.mean(dim=1)]).permute(1, 0, 2))
        if i > max_num_batches:
            break
    # compute score 
    lvm_feats1, lvm_feats2 = torch.cat(lvm_feats1), torch.cat(lvm_feats2)
    metric_to_score = dict()
    for metric in metrics:
        metric_to_score[metric] = platonic_metric.score(lvm_feats1, lvm_feats2, metric=metric, topk=10, normalize=True)['A'][0]
    return metric_to_score

def analyze_file_strings(file1_path, file2_path, all_str):
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        set1 = set(line.strip() for line in f1)
        set2 = set(line.strip() for line in f2)
    all_set = set(all_str)
    # Strings common to both files
    common_strings = set1 & set2
    # Strings in only one of the two files
    exclusive_strings = (set1 ^ set2) 
    # Strings in all_str but in neither file
    outside_strings = all_set - (set1 | set2)
    return common_strings, exclusive_strings, outside_strings, set1, set2

if __name__ == "__main__":
    args = get_args_parser()
    device = torch.device("cuda:0")

    path_to_cka_res = os.path.join(args.log_base_dir, f"platonic_mean_bin_data.csv")
    # mean stands for: mean of all embeddings in last hidden state, so [bsz, n_tokens, n_embed] becomes [bsz, 1, n_embed]
    f = open(path_to_cka_res, "a")
    f.write("seed,metric,frac_overlap,metric_common,metric_OOD_for_one,metric_OOD_for_all\n")
    f.close()
    for seed in tqdm.tqdm(range(6)):
        all_dirs = os.listdir(os.path.join(args.log_base_dir, f"seed_{seed:d}"))
        for p in all_dirs:
            if "slurm" not in p:
                path_to_splits = os.path.join(args.log_base_dir, f"seed_{seed:d}", p, "split_indices")
                print(path_to_splits)
                path_to_train_args = os.path.join(args.log_base_dir, f"seed_{seed:d}", p, "args.json")
                with open(path_to_train_args, "r") as f:
                    args_file = f.read()
                    train_args_dict = json.loads(args_file)

                common, ood_for_one, ood_for_all, s0, s1 = \
                    analyze_file_strings(os.path.join(path_to_splits, "split_0.txt"), 
                                         os.path.join(path_to_splits, "split_1.txt"), 
                                         args.datasets)
                # (self, avail_datasets, total_len, block_size, batch_size, device)
                block_size = train_args_dict["block_size"]
                common_dl = RandomBinDataset(list(common), block_size, args.batch_size, device=device) if len(common) > 0 else None
                ood_for_one_dl = RandomBinDataset(list(ood_for_one), block_size, args.batch_size, device=device) if len(ood_for_one) > 0 else None
                ood_for_all_dl = RandomBinDataset(list(ood_for_all), block_size, args.batch_size, device=device) if len(ood_for_all) > 0 else None
                frac_overlap = len(s1 & s0) / len(s0)
                
                # load models
                path_to_0 = os.path.join(args.log_base_dir, f"seed_{seed:d}", p, "split_0/ckpt.pt")
                path_to_1 = os.path.join(args.log_base_dir, f"seed_{seed:d}", p, "split_1/ckpt.pt")
                first = construct_loaded_gpt2(path_to_0, device).eval()
                second = construct_loaded_gpt2(path_to_1, device).eval()

                metric_common = compute_metric(first, second, common_dl, device, args.metrics)
                metric_OOD_for_one = compute_metric(first, second, ood_for_one_dl, device, args.metrics)
                metric_OOD_for_all = compute_metric(first, second, ood_for_all_dl, device, args.metrics)
                
                f = open(path_to_cka_res, "a")
                for metric in args.metrics:
                    # "seed,metric,frac_overlap,metric_common,metric_OOD_for_one,metric_OOD_for_all"
                    f.write(f"{seed:d},{metric},{frac_overlap},{metric_common[metric]:.6f},{metric_OOD_for_one[metric]:.6f},{metric_OOD_for_all[metric]:.6f}\n")
                f.close()
