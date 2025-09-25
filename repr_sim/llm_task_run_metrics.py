#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, json, re, glob, tqdm, torch, gc, random, pickle
import numpy as np

import platonic
from metrics import AlignmentMetrics

from torch.utils.data import IterableDataset

from transformers import AutoModelForCausalLM
from peft import PeftModel

# ------------------------------
# Config / Constants
# ------------------------------
# Where finetuning stored tokenized data: ./data/{safe_dataset}/family_{family}/(train|val).bin
# This mirrors the trainer’s layout.
DATA_DIR_BASE = "../llm_finetune/data"

CANDIDATES = [
    "nampdn-ai/tiny-codes",
    "roneneldan/TinyStories",
    "nampdn-ai/tiny-textbooks",
    "nampdn-ai/tiny-webtext",
    "Salesforce/wikitext",      # uses subset wikitext-103-raw-v1 during tokenization
    "Trelis/tiny-shakespeare",
]

# ------------------------------
# CLI
# ------------------------------
def get_args_parser():
    parser = argparse.ArgumentParser("platonic_llm_peft", add_help=True)
    parser.add_argument("--log_base_dir", type=str, required=True, help="Root log dir, e.g. path_to_log_dir/task_overlap_bfloat16/meta-llama_Llama-3.2-1B-Instruct")
    parser.add_argument("--model_name", type=str, required=True, help="Base HF model name used during finetuning (same as --model_name in training).")
    parser.add_argument("--batch_size", type=int, default=256, help="Eval batch size (token sequences).")
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
    meta_path = os.path.join(meta_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab = int(meta.get("vocab_size", 50304))
    return np.uint16 if vocab <= 65535 else np.uint32

def _build_sources_for_category(dataset_names, model_name):
    """
    For each dataset name, locate ../llm_finetune/data/{safe_ds}/family_{family}/val.bin (+ meta.pkl).
    Return a list of sources [{"bin_path":..., "meta_dir":...}, ...] but only for those that exist.
    """
    family = _family_from_model_name(model_name)
    sources = []
    for ds in dataset_names:
        meta_dir = os.path.join(DATA_DIR_BASE, ds.replace('/', '_'), f"family_{_safe_str_dir(family)}")
        val_bin = os.path.join(meta_dir, "val.bin")
        meta_pkl = os.path.join(meta_dir, "meta.pkl")
        assert os.path.exists(val_bin) and os.path.exists(meta_pkl), f"meta_dir={meta_dir}"
        sources.append({"bin_path": val_bin, "meta_dir": meta_dir})
    return sources

def _read_split_list(txt_path):
    with open(txt_path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

# ------------------------------
# RandomMixtureDataset (validation)
# ------------------------------
class RandomMixtureDataset(IterableDataset):
    def __init__(self, sources, block_size, seed):
        super().__init__()
        self.sources = sources
        self.block_size = block_size
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed + int.from_bytes(os.urandom(2), "little"))
        while True:
            src = rng.choice(self.sources)
            dtype = _token_dtype_from_meta(src["meta_dir"])
            data = np.memmap(src["bin_path"], dtype=dtype, mode="r")
            i = rng.randint(0, len(data) - self.block_size - 1)
            x = torch.from_numpy((data[i : i + self.block_size]).astype(np.int64))
            yield {"input_ids": x, "labels": x.clone()}

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
# Metric computation
# ------------------------------
@torch.no_grad()
def _next_batch_from_iterable(dl_iter, batch_size, device):
    """
    Pull `batch_size` items from an iterable dataset iterator and stack to (B, T).
    """
    xs = []
    for _ in range(batch_size):
        sample = next(dl_iter)          # {"input_ids": tensor(T), "labels": tensor(T)}
        xs.append(sample["input_ids"])
    x = torch.stack(xs, dim=0).to(device)
    return x

@torch.no_grad()
def compute_metric(first, second, dl: RandomMixtureDataset, device, metrics,
                   batch_size=256, max_num_batches=80):
    """
    If dl is None (category empty), return metrics with value -1.
    Otherwise, run max_num_batches batches of size `batch_size` drawn from dl.
    """
    if dl is None:
        return {metric: -1 for metric in metrics}

    platonic_metric = platonic.Alignment(
        dataset="gpt2_model", subset=None, models=["A", "B"], transform=None, device=device, dtype=torch.float32,)

    lvm_feats1, lvm_feats2 = [], []
    dl = iter(dl)
    for _ in tqdm.tqdm(range(max_num_batches), desc="forward pass"):
        inp_batch = _next_batch_from_iterable(dl, batch_size, device)  # (B, T)
        out1 = first(input_ids=inp_batch, output_hidden_states=True)
        out2 = second(input_ids=inp_batch, output_hidden_states=True)
        lvm_output1 = out1.hidden_states[-1].mean(dim=1)  # (bsz, dim)
        lvm_output2 = out2.hidden_states[-1].mean(dim=1)
        # stack a pseudo "layer" axis, then permute to (bsz, layer, dim)
        lvm_feats1.append(torch.stack([lvm_output1]).permute(1, 0, 2))
        lvm_feats2.append(torch.stack([lvm_output2]).permute(1, 0, 2))

    lvm_feats1 = torch.cat(lvm_feats1, dim=0)  # (N*B, 1, D)
    lvm_feats2 = torch.cat(lvm_feats2, dim=0)  # (N*B, 1, D)

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

    # Prepare CSV (three categories now)
    path_to_cka_res = os.path.join(args.log_base_dir, args.csv_name)
    with open(path_to_cka_res, "a") as f:
        f.write("seed,metric,frac_overlap,metric_common,metric_OOD_for_one,metric_OOD_for_all\n")

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
                block_size = int(json.load(f)["block_size"])

            # Read split membership
            split_idx_dir = os.path.join(frac_dir, "split_indices")
            split0_list = _read_split_list(os.path.join(split_idx_dir, "split_0.txt"))
            split1_list = _read_split_list(os.path.join(split_idx_dir, "split_1.txt"))

            set0, set1 = set(split0_list), set(split1_list)
            all_used = set0 | set1
            overlap_set   = set0 & set1
            exclusive_set = (set0 ^ set1)  # in exactly one
            neither_set   = set(CANDIDATES) - all_used

            # Build sources for each category (only those with existing val.bin/meta.pkl)
            overlap_sources   = _build_sources_for_category(sorted(overlap_set), args.model_name)
            exclusive_sources = _build_sources_for_category(sorted(exclusive_set), args.model_name)
            neither_sources   = _build_sources_for_category(sorted(neither_set), args.model_name)

            # Wrap each category in a RandomMixtureDataset or set to None if empty/missing
            dl_overlap   = RandomMixtureDataset(overlap_sources,   block_size, seed_val) if len(overlap_sources)   > 0 else None
            dl_exclusive = RandomMixtureDataset(exclusive_sources, block_size, seed_val) if len(exclusive_sources) > 0 else None
            dl_neither   = RandomMixtureDataset(neither_sources,   block_size, seed_val) if len(neither_sources)   > 0 else None

            # Load models for split_0 and split_1
            split0 = os.path.join(frac_dir, "split_0")
            split1 = os.path.join(frac_dir, "split_1")

            first = load_split_model(args.model_name, split0, device, hf_token, hf_cache)
            second = load_split_model(args.model_name, split1, device, hf_token, hf_cache)

            # Compute metrics per category (dl=None is acceptable and returns -1s)
            scores_overlap = compute_metric(first, second, dl_overlap,   device, args.metrics,
                                            batch_size=args.batch_size, max_num_batches=args.max_num_batches)
            scores_excl    = compute_metric(first, second, dl_exclusive, device, args.metrics,
                                            batch_size=args.batch_size, max_num_batches=args.max_num_batches)
            scores_neither = compute_metric(first, second, dl_neither,   device, args.metrics,
                                            batch_size=args.batch_size, max_num_batches=args.max_num_batches)

            # Append to CSV: one row per metric with three columns
            with open(path_to_cka_res, "a") as f:
                for metric in args.metrics:
                    f.write(
                        f"{seed_val},{metric},{frac_overlap},"
                        f"{scores_overlap[metric]:.6f},"
                        f"{scores_excl[metric]:.6f},"
                        f"{scores_neither[metric]:.6f}\n"
                    )

            # cleanup VRAM
            first.to('cpu'); second.to('cpu')
            del first, second
            torch.cuda.empty_cache()
            gc.collect()
    print(f"Saved results to: {path_to_cka_res}")

if __name__ == "__main__":
    main()
