#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, json, re, glob, tqdm, torch, gc
import numpy as np

import platonic
from metrics import AlignmentMetrics

from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM
from peft import PeftModel

def get_args_parser():
    parser = argparse.ArgumentParser("platonic_llm_peft", add_help=True)
    parser.add_argument("--log_base_dir", type=str, required=True, help="e.g. path_to_log_dir/ds_overlap_bfloat16_tinycodes/meta-llama_Llama-3.2-3B-Instruct")
    parser.add_argument("--model_name", type=str, required=True, help="Base HF model name used during finetuning (same as --model_name in training).")
    parser.add_argument("--hf_dataset", type=str, required=True, help="HF dataset name used during finetuning (same as --hf_dataset).")
    parser.add_argument("--batch_size", type=int, default=256, help="Eval batch size (tokens sequences).")
    parser.add_argument("--metrics", nargs="+", type=str, default=AlignmentMetrics.SUPPORTED_METRICS, help="List of platonic metrics to compute.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hf_token_env", type=str, default="my_token", help="Env var holding HF token, if needed.")
    parser.add_argument("--hf_cache_env", type=str, default="HF_TRANSFORMER_CACHE", help="Env var for HF cache dir, if set.")
    parser.add_argument("--max_num_batches", type=int, default=80, help="Max number of batches per run (like the nanoGPT script).")
    parser.add_argument("--csv_name", type=str, default="platonic_llm.csv", help="Name of the output CSV saved inside --log_base_dir.")
    return parser.parse_args()

# ------------------------------
# Utilities
# ------------------------------
def _safe_str_dir(inp: str) -> str:
    return inp.replace("/", "_").replace(":", "_").replace("-", "")

def _family_from_model_name(model_name: str) -> str:
    name = model_name.lower()
    if "llama" in name or "meta-llama" in name:
        return "llama"
    if "gemma" in name or "google/gemma" in name:
        return "gemma"
    return "unknown"

def _token_dtype_from_meta(meta_dir: str):
    import pickle, os
    meta_path = os.path.join(meta_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab = int(meta.get("vocab_size", 50304))
    return np.uint16 if vocab <= 65535 else np.uint32

# ------------------------------
# Validation .bin dataset (memmap)
# ------------------------------
class BinDataset(Dataset):
    def __init__(self, val_bin_dir: str, block_size: int, batch_size: int, device: torch.device):
        """
        val_bin_dir: e.g. ./data/{safe_dataset}/family_{family}
        expects val.bin and meta.pkl inside
        """
        self.meta_dir = val_bin_dir
        self.data = np.memmap(os.path.join(val_bin_dir, "val.bin"),
                              dtype=_token_dtype_from_meta(val_bin_dir), mode='r')
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ix = torch.randint(0, len(self.data) - self.block_size - 1, (self.batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i+1:i+1+self.block_size].astype(np.int64)) for i in ix])
        return x.pin_memory().to(self.device, non_blocking=True), \
               y.pin_memory().to(self.device, non_blocking=True)

# ------------------------------
# Load base + adapter (PEFT)
# ------------------------------
def load_split_model(base_name: str, adapter_dir: str, device: torch.device,
                     hf_token: str = None, hf_cache: str = None):
    load_kwargs = dict(trust_remote_code=True, token=hf_token, cache_dir=hf_cache)
    base = AutoModelForCausalLM.from_pretrained(base_name, **load_kwargs).to(device)
    base.eval()
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    model.eval()
    return model

# ------------------------------
# Metric computation (keep structure & permute)
# ------------------------------
@torch.no_grad()
def compute_metric(first, second, dl: BinDataset, device, metrics, max_num_batches=80):
    if dl is None:
        return {metric: -1 for metric in metrics}

    platonic_metric = platonic.Alignment(
        dataset="gpt2_model", subset=None, models=["A", "B"], transform=None, device=device, dtype=torch.float32,)

    lvm_feats1, lvm_feats2 = [], []
    for i in tqdm.tqdm(range(max_num_batches), desc="forward pass"):
        inp_txt = dl[i][0].to(device)
        # HF models: request hidden states
        out1 = first(input_ids=inp_txt, output_hidden_states=True)
        out2 = second(input_ids=inp_txt, output_hidden_states=True)
        lvm_output1 = out1.hidden_states[-1]  # (bsz, seq, dim)
        lvm_output2 = out2.hidden_states[-1]

        # mean over sequence, keep shape: (bsz, dim)
        lvm_output1 = lvm_output1.mean(dim=1)
        lvm_output2 = lvm_output2.mean(dim=1)

        # stack a pseudo "layer" axis, then permute to (bsz, layer, dim)
        lvm_feats1.append(torch.stack([lvm_output1]).permute(1, 0, 2))
        lvm_feats2.append(torch.stack([lvm_output2]).permute(1, 0, 2))

    lvm_feats1 = torch.cat(lvm_feats1, dim=0)
    lvm_feats2 = torch.cat(lvm_feats2, dim=0)

    metric_to_score = {}
    for metric in metrics:
        metric_to_score[metric] = platonic_metric.score(lvm_feats1, lvm_feats2, metric=metric, topk=10, normalize=True)['A'][0]
    return metric_to_score

# ------------------------------
# Main
# ------------------------------
def main():
    args = get_args_parser()
    print("log_base_dir", args.log_base_dir)
    device = torch.device(args.device)

    hf_token = os.environ.get(args.hf_token_env) if args.hf_token_env else None
    hf_cache = os.environ.get(args.hf_cache_env) if args.hf_cache_env else None

    # Figure out validation bin dir built by your finetuner:
    # ./data/{safe_dataset}/family_{family}/val.bin
    safe_ds = args.hf_dataset.replace('/', '_')
    family = _family_from_model_name(args.model_name)
    val_bin_dir = os.path.join("../llm_finetune/data", safe_ds, f"family_{_safe_str_dir(family)}")
    if not (os.path.exists(os.path.join(val_bin_dir, "val.bin")) and
            os.path.exists(os.path.join(val_bin_dir, "meta.pkl"))):
        raise FileNotFoundError(f"Could not find val.bin/meta.pkl under {val_bin_dir}")

    # Prepare CSV
    ds_tag = args.hf_dataset.split("/")[-1].replace("-", "")
    path_to_cka_res = os.path.join(args.log_base_dir, args.csv_name)
    with open(path_to_cka_res, "a") as f:
        f.write(f"seed,metric,frac_overlap,metric_{ds_tag}\n")

    # Expected structure:
    # {log_base_dir}/seed_{i}/frac_overlap_{x}/split_{0,1}/adapter_model.safetensors
    seed_dirs = sorted(glob.glob(os.path.join(args.log_base_dir, "seed_*")))
    frac_re = re.compile(r"frac_overlap_([0-9]*\.?[0-9]+)")

    for seed_dir in tqdm.tqdm(seed_dirs, desc="Seeds"):
        seed_name = os.path.basename(seed_dir)
        seed_val = int(seed_name.split("_")[-1])
        frac_dirs = sorted(glob.glob(os.path.join(seed_dir, "frac_overlap_*")))
        
        for frac_dir in frac_dirs:
            m = frac_re.search(os.path.basename(frac_dir))
            if not m:
                continue
            frac_overlap = float(m.group(1))

            # read block_size from args.json saved by trainer at this level
            args_json = os.path.join(frac_dir, "args.json")
            with open(args_json, "r") as f:
                block_size = json.load(f)["block_size"]

            # Build dataset for this block size
            dl_val = BinDataset(val_bin_dir, block_size=block_size, batch_size=args.batch_size, device=device)

            # Load models for split_0 and split_1
            split0 = os.path.join(frac_dir, "split_0")
            split1 = os.path.join(frac_dir, "split_1")

            first = load_split_model(args.model_name, split0, device, hf_token, hf_cache)
            second = load_split_model(args.model_name, split1, device, hf_token, hf_cache)

            # Compute metrics on validation data
            metric_vals = compute_metric(first, second, dl_val, device, args.metrics, max_num_batches=args.max_num_batches)

            # Append to CSV
            with open(path_to_cka_res, "a") as f:
                for metric in args.metrics:
                    f.write(f"{seed_val},{metric},{frac_overlap},{metric_vals[metric]:.6f}\n")
            first.to('cpu')
            del first
            second.to('cpu')
            del second
            torch.cuda.empty_cache()
            gc.collect()
    print(f"Saved results to: {path_to_cka_res}")

if __name__ == "__main__":
    main()
