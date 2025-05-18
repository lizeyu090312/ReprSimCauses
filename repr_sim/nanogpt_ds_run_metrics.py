import platonic
from metrics import AlignmentMetrics

import torch, torchvision, os, tqdm, argparse, json, tiktoken, logging, sys
import numpy as np

from torch.utils.data import DataLoader
from torch import nn
import torchvision.transforms.v2 as v2
from torchvision.models.feature_extraction import create_feature_extractor

from datasets import load_dataset

from torch.utils.data import Dataset

sys.path.append("../")
from train_nanoGPT.model import GPTConfig, GPT


def get_args_parser():
    parser = argparse.ArgumentParser("platonic_gpt2", add_help=False)

    parser.add_argument("--log_base_dir", type=str, required=True)
    
    parser.add_argument("--batch_size", type=int, default=250, help="Batch size")
    parser.add_argument("--metrics", nargs="+", type=str, default=AlignmentMetrics.SUPPORTED_METRICS, help="List of metrics to compute")
    return parser.parse_args()

class HFDataLoader:
    def __init__(self, dataset, device, block_size, tokenizer_name="gpt2", batch_size=32):
        """
        Dataloader for HF datasets that returns tokenized batches for language modeling.
        Note: this dataset may repeat a data point before finishing with all data in one epoch due to random sampling
        with replacement
        Args:
            dataset (str): from load_dataset (e.g., "roneneldan/TinyStories" or "wikitext").
            tokenizer_name (str): The tokenizer to use (e.g., "gpt2").
            block_size (int): Number of tokens per sample.
            batch_size (int): Batch size.
            device (torch.device): The device to load batches onto.
        """
        self.dataset = dataset
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        
        # Concatenate all text fields into one long string.
        # Adjust the key ("text") if your dataset uses another name.
        texts = [example['text'] for example in self.dataset]
        self.text = "\n\n".join(texts)
        # Get the tokenizer from tiktoken (using GPT-2's BPE here)
        self.encoder = tiktoken.get_encoding(tokenizer_name)
        # Tokenize into token IDs.
        self.token_ids = np.array(self.encoder.encode_ordinary(self.text), dtype=np.uint16)
        self.N = len(self.token_ids)
        print(f"Loaded {len(self.dataset)} examples; total tokens: {self.N}")
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        """
        Samples a batch of tokenized sequences.
        
        Returns:
            (x, y): Two torch.LongTensor batches on the desired device,
                    where x is a tensor of input token IDs and y is x shifted one token ahead.
        """
        ix = torch.randint(0, self.N - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy(self.token_ids[i:i+self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.token_ids[i+1:i+1+self.block_size].astype(np.int64)) for i in ix])
        return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)


class BinDataset(Dataset):
    def __init__(self, data_path, block_size, batch_size, device):
        """
        PyTorch Dataset wrapper for a tokenized dataset stored in a binary (.bin) file.

        Args:
            data_path (str): path to dataset
            block_size (int): Length (in tokens) of each sample.
            batch_size (int): Number of samples to return per __getitem__.
            device (torch.device): Device to place the tensors.
            length (int): Artificial dataset length (number of batches).
        """
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data
        ix = torch.randint(0, len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+self.block_size].astype(np.int64)) for i in ix])
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

if __name__ == "__main__":
    args = get_args_parser()
    device = torch.device("cuda:0")

    dl_shakespeare_all = BinDataset("../train_nanoGPT/data/shakespeare_all/val.bin", block_size=None, # block_size set later
                                      batch_size=args.batch_size, device=device)
    dl_tinystories = BinDataset("../train_nanoGPT/data/tinystories/val.bin", block_size=None, # block_size set later
                                      batch_size=args.batch_size, device=device)
    dl_wikitext = HFDataLoader(dataset=load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="validation"), 
                                 device=device, tokenizer_name="gpt2", 
                                 batch_size=args.batch_size, block_size=None)
    
    path_to_cka_res = os.path.join(args.log_base_dir, f"platonic_mean_bin_data.csv")
    f = open(path_to_cka_res, "a")
    f.write("seed,metric,frac_overlap,metric_shakespeare,metric_tinystories,metric_wikitext\n")
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
                dl_shakespeare_all.block_size = train_args_dict["block_size"]
                dl_tinystories.block_size = train_args_dict["block_size"]
                dl_wikitext.block_size = train_args_dict["block_size"]
                
                common = set(list(np.load(os.path.join(path_to_splits, "common.npy"))))
                s0 = set(list(np.load(os.path.join(path_to_splits, "split_0.npy"))))
                s1 = set(list(np.load(os.path.join(path_to_splits, "split_1.npy"))))
                frac_overlap = len(s1 & s0) / len(s0)
                
                # load models
                path_to_0 = os.path.join(args.log_base_dir, f"seed_{seed:d}", p, "split_0/ckpt.pt")
                path_to_1 = os.path.join(args.log_base_dir, f"seed_{seed:d}", p, "split_1/ckpt.pt")
                first = construct_loaded_gpt2(path_to_0, device).eval()
                second = construct_loaded_gpt2(path_to_1, device).eval()

                metric_shakespeare_all = compute_metric(first, second, dl_shakespeare_all, device, args.metrics)
                metric_tinystories = compute_metric(first, second, dl_tinystories, device, args.metrics)
                metric_wikitext = compute_metric(first, second, dl_wikitext, device, args.metrics)
                
                f = open(path_to_cka_res, "a")
                for metric in args.metrics:
                    # "seed,metric,frac_overlap,metric_overlap,metric_ood_all_tinystories,metric_ood_all_wikitext"
                    f.write(f"{seed:d},{metric},{frac_overlap},{metric_shakespeare_all[metric]:.6f},{metric_tinystories[metric]:.6f},{metric_wikitext[metric]:.6f}\n")
                f.close()
