#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace/PEFT LoRA finetuning (two splits) + your dataset constructor.

- Data: tokenize once per (dataset, model_name) into ./data/{dataset}/{safe_model}/(train|val).bin
- Splits/chunks: exactly your code path (minimal edits)
- Training: HuggingFace Trainer for everything (two independent adapters)

Example:
python finetune_lora_two_splits_trainer.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --frac_overlap 0.5 \
  --hf_dataset nampdn-ai/tiny-webtext \
  --max_steps 2000 --per_device_train_batch_size 4 --logging_steps 20 --save_strategy no \
  --load_in_4bit
"""

import os, sys, math, gc, time, json, random, pickle, argparse, logging, tqdm, torch
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from torch.utils.data import IterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# HF stack
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    BitsAndBytesConfig, default_data_collator
)
from datasets import load_dataset
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except:
    pass

# PEFT
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# =========================
#      LOGGING
# =========================
def setup_logging(log_file, rank):
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file); fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(ch)


# =========================
#   TOKENIZATION / STORAGE
# =========================
TINY_PRESETS = {
    # tiny-webtext: take the 'human' field as the raw text
    "nampdn-ai/tiny-webtext": {
        "text_field": "human", "train_split": "train", "val_split": "validation", "config": None,
        "generate_prompt_field": "human",
    },
    # tiny-codes: special handling (prompt + SEP + response); text_field is ignored
    "nampdn-ai/tiny-codes": {
        "text_field": None, "train_split": "train", "val_split": "validation", "config": None,
        "generate_prompt_field": "prompt",
    },
    # tiny-shakespeare: splits train/test; some variants have 'Text' (capital T)
    "Trelis/tiny-shakespeare": {
        "text_field": "Text", "train_split": "train", "val_split": "test", "config": None,
        "generate_prompt_field": "Text",
    },
    # TinyStories: standard train/validation
    "roneneldan/TinyStories": {
        "text_field": "text", "train_split": "train", "val_split": "validation", "config": None,
        "generate_prompt_field": "text",
    },
    # WikiText (use subset wikitext-103-raw-v1); train/validation; ignore test
    "Salesforce/wikitext": {
        "text_field": "text", "train_split": "train", "val_split": "validation", "config": "wikitext-103-raw-v1",
        "generate_prompt_field": "text",
    },
    # tiny-textbooks: train/test
    "nampdn-ai/tiny-textbooks": {
        "text_field": "text", "train_split": "train", "val_split": "test", "config": None,
        "generate_prompt_field": "text",
    },
}

CANDIDATES = list(TINY_PRESETS.keys())


def choose_datasets(valid_datasets, frac_overlap):
    random.shuffle(valid_datasets)
    assert len(valid_datasets) % 2 == 0
    assert abs(frac_overlap * len(valid_datasets) - round(frac_overlap * len(valid_datasets))) < 1e-3, f"{frac_overlap}, {valid_datasets}, {frac_overlap * len(valid_datasets)}, {int(frac_overlap * len(valid_datasets))}"
    valid_ds_len_half = round(len(valid_datasets)/2)
    n_common = int(round(valid_ds_len_half*frac_overlap))
    common_data = valid_datasets[0:n_common]

    datasets0 = valid_datasets[n_common:valid_ds_len_half]
    datasets1 = valid_datasets[valid_ds_len_half:int(valid_ds_len_half*2)-n_common]
    return datasets0 + common_data, datasets1 + common_data

def _resolve_hf_layout(ds_name, ds_config, text_field, train_split, val_split, auto_preset):
    tf, tr, va, cfg = text_field, train_split, val_split, ds_config
    if auto_preset and ds_name in TINY_PRESETS:
        preset = TINY_PRESETS[ds_name]
        tf = tf or preset["text_field"]
        tr = tr or preset["train_split"]
        va = va or preset["val_split"]
        cfg = cfg if (cfg is not None and cfg != "") else preset["config"]
    tf = tf or "text"
    tr = tr or "train"
    va = va or "validation"
    return tf, tr, va, cfg

def _safe_str_dir(inp: str) -> str:
    return inp.replace("/", "_").replace(":", "_")

def _dtype_for_vocab(vocab_size: int):
    return np.uint16 if vocab_size <= 65535 else np.uint32

def _family_from_model_name(model_name: str) -> str:
    name = model_name.lower()
    if "llama" in name or "meta-llama" in name:
        return "llama"
    if "gemma" in name or "google/gemma" in name:
        return "gemma"  # treat Gemma separately from Gemini if you ever use it
    return "unknown"

def _ensure_tokenized_bins(
    dataset: str,
    model_name: str,
    tokenizer: AutoTokenizer,
    text_field: str = "text",
    hf_dataset_config: Optional[str] = None,
    val_split: str = "validation",
    train_split: str = "train",
) -> str:
    """
    Creates ./data/{dataset}/family_{family}/(train|val).bin and meta.pkl if missing.
    One set per tokenizer family (e.g., llama vs gemini). Always reuses local .bin if present.
    """
    base_dir = os.path.join("data", _safe_str_dir(dataset))
    family = _family_from_model_name(model_name)
    out_dir = os.path.join(base_dir, f"family_{_safe_str_dir(family)}")
    os.makedirs(out_dir, exist_ok=True)
    train_bin = os.path.join(out_dir, "train.bin")
    val_bin = os.path.join(out_dir, "val.bin")
    meta_pkl = os.path.join(out_dir, "meta.pkl")

    if os.path.exists(train_bin) and os.path.exists(val_bin) and os.path.exists(meta_pkl):
        return out_dir

    # --- Load raw text (HF datasets) ---
    ds = load_dataset(dataset, hf_dataset_config)
    if val_split not in ds:
        full = ds[train_split]
        n = len(full)
        cut = max(1, int(0.1 * n))
        ds = {train_split: full.select(range(0, n - cut)),
              val_split:   full.select(range(n - cut, n)),}

    # Special handling for tiny-codes
    if dataset == "nampdn-ai/tiny-codes":
        sep_token = tokenizer.eos_token
        def _concat_table(split_table):
            prompts = split_table["prompt"]
            responses = split_table["response"]
            return [f"{p}{sep_token}{r}" for p, r in zip(prompts, responses)]
        train_texts = _concat_table(ds[train_split])
        val_texts   = _concat_table(ds[val_split])
    else:
        train_texts = ds[train_split][text_field]
        val_texts   = ds[val_split][text_field]

    # --- Tokenize & write ---
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("Tokenizer must define eos_token_id.")

    def encode_texts(texts):
        ids = []
        for t in texts:
            ids.extend(tokenizer.encode(t, add_special_tokens=False))
            ids.append(eos_id)
        return np.array(ids, dtype=_dtype_for_vocab(len(tokenizer)))

    tr_ids = encode_texts(train_texts)
    va_ids = encode_texts(val_texts)

    tr_ids.tofile(train_bin)
    va_ids.tofile(val_bin)
    with open(meta_pkl, "wb") as f:
        pickle.dump({
            "vocab_size": len(tokenizer),
            "tokenizer_family": family,
            "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        }, f)

    return out_dir

def _token_dtype_from_meta(dir_path: str):
    meta_path = os.path.join(dir_path, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab = int(meta.get("vocab_size", 50304))
    return np.uint16 if vocab <= 65535 else np.uint32

# =========================
#     DATASET WRAPPERS
# =========================
class RandomMixtureDataset(IterableDataset):
    """
    Infinite stream where each cycle emits exactly one sample from each source.
    For each emitted sample we open a fresh memmap to that source (nanoGPT style).
    """
    def __init__(self, sources, block_size, seed):
        super().__init__()
        self.sources = sources          # list of {"bin_path": ..., "meta_dir": ...}
        self.block_size = block_size
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed + int.from_bytes(os.urandom(2), "little"))
        order = list(range(len(self.sources)))
        while True:
            rng.shuffle(order)  # randomize order each cycle
            src = self.sources[order[0]]
            dtype = _token_dtype_from_meta(src["meta_dir"])
            data = np.memmap(src["bin_path"], dtype=dtype, mode="r")  # fresh memmap
            i = rng.randint(0, len(data) - self.block_size - 1)
            x = torch.from_numpy((data[i : i + self.block_size]).astype(np.int64))
            yield {"input_ids": x, "labels": x.clone()}

# =========================
#       TRAINING
# =========================
def build_model_and_trainer(
    split_name,
    model_name,
    tokenizer,
    lora_cfg,
    training_args,
    bitsnbytes,
    block_size,
    train_dataset,
    output_dir_chunk,
    hf_token,
    hf_cache,
    use_gc,
):
    load_kwargs = dict(trust_remote_code=True, token=hf_token, cache_dir=hf_cache)
    if bitsnbytes is not None:
        load_kwargs["quantization_config"] = bitsnbytes
        load_kwargs["device_map"] = "auto"

    base = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if use_gc:
        base.gradient_checkpointing_enable()
    if bitsnbytes is not None:
        base = prepare_model_for_kbit_training(base)

    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    trainable_params, all_param = model.get_nb_trainable_parameters()

    logging.info(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )
    return model, trainer

def _free_after_trainer(trainer=None, model=None, extra_refs: list = None):
    """
    Aggressively free GPU memory after a Trainer run.
    """
    # 1) ask accelerate (if present) to release CUDA context
    try:
        if trainer is not None and hasattr(trainer, "accelerator") and trainer.accelerator is not None:
            # accelerate>=0.26 has free_memory(); guarded for older versions
            fm = getattr(trainer.accelerator, "free_memory", None)
            if callable(fm):
                fm()
    except Exception:
        pass

    # 2) move model to CPU to drop CUDA tensors
    try:
        if model is not None:
            model.to("cpu")
    except Exception:
        pass

    # 3) break reference cycles
    try:
        if trainer is not None:
            trainer._callbacks = []          # drop callbacks that may capture model/optimizer
            trainer.train_dataset = None
            trainer.eval_dataset = None
            trainer.model = None
            trainer.args = None
    except Exception:
        pass

    # 4) drop user refs
    if extra_refs:
        for r in extra_refs:
            try:
                del r
            except Exception:
                pass

    # 5) final cleanup
    try:
        del trainer
    except Exception:
        pass
    try:
        del model
    except Exception:
        pass
    gc.collect()
    import torch
    torch.cuda.empty_cache()

def main(args):
    # Output dir per your structure
    out_dir = os.path.join(args.output_dir_base, _safe_str_dir(args.model_name), 
                           f"seed_{args.seed}", f"frac_overlap_{args.frac_overlap}")
    os.makedirs(out_dir, exist_ok=True)
    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=2)
    # DDP detection (Trainer handles distributed under the hood)
    # Seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Tokenizer
    hf_token = os.environ.get(args.hf_token_env) if args.hf_token_env else None
    hf_cache = os.environ.get(args.hf_cache_env) if args.hf_cache_env else None
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, token=hf_token, cache_dir=hf_cache, 
        trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Decide the two 3-dataset mixtures with the requested fractional overlap
    split0_list, split1_list = choose_datasets(CANDIDATES.copy(), args.frac_overlap)

    # Save which datasets were used per split (for reproducibility/inspection)
    split_log_dir = os.path.join(out_dir, "split_indices")
    os.makedirs(split_log_dir, exist_ok=True)
    with open(os.path.join(split_log_dir, "split_0.txt"), "w") as f:
        for d in split0_list: f.write(d + "\n")
    with open(os.path.join(split_log_dir, "split_1.txt"), "w") as f:
        for d in split1_list: f.write(d + "\n")

    # LoRA config
    target_modules = args.target_modules.split(",") if args.target_modules else ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # BitsAndBytes (optional)
    bnb = None
    if args.load_in_4bit or args.load_in_8bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_use_double_quant=True if args.load_in_4bit else None,
            bnb_4bit_quant_type="nf4" if args.load_in_4bit else None,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bfloat16_compute else torch.float16,
        )

    # Train two separate adapters
    for split_name, chosen in zip(["split_0", "split_1"], [split0_list, split1_list]):
        # Tokenize (or reuse cached) for each dataset in this split, then build sources
        train_sources = []
        for ds_name in chosen:
            tf, tr, va, cfg = _resolve_hf_layout(
                ds_name=ds_name, ds_config="", text_field=None,
                train_split=None, val_split=None, auto_preset=True
            )
            model_data_dir = _ensure_tokenized_bins(
                dataset=ds_name,
                model_name=args.model_name,
                tokenizer=tokenizer,
                text_field=tf,
                hf_dataset_config=cfg,
                val_split=va,
                train_split=tr,
            )
            train_sources.append({
                "bin_path": os.path.join(model_data_dir, "train.bin"),
                "meta_dir": model_data_dir,
            })

        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        setup_logging(os.path.join(split_dir, "log.log"), 0)
        logging.info(f"[{split_name}] model={args.model_name}")

        # Iterable dataset over chunk ranges
        train_dataset = RandomMixtureDataset(train_sources, block_size=args.block_size, seed=args.seed)

        # Precision switches
        dtype_kwargs = dict(fp16=False, bf16=False)
        if args.precision == "bfloat16": dtype_kwargs = dict(fp16=False, bf16=True)
        elif args.precision == "float16": dtype_kwargs = dict(fp16=True, bf16=False)

        # Trainer config
        targs = TrainingArguments(
            output_dir=split_dir,
            overwrite_output_dir=True,
            logging_dir=split_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            num_train_epochs=1,  # ignored because max_steps is set
            logging_steps=args.logging_steps,
            save_strategy=args.save_strategy,  # "no" by default; we save final manually
            report_to=["tensorboard"],
            remove_unused_columns=False,
            dataloader_drop_last=True,
            dataloader_num_workers=args.dataloader_workers,
            **dtype_kwargs,
        )

        # Build model + trainer
        model, trainer = build_model_and_trainer(
            split_name, args.model_name, tokenizer, lora_cfg, targs,
            bnb, args.block_size, train_dataset, split_dir, hf_token, hf_cache, args.gradient_checkpointing
        )

        # Train
        start = time.time()
        trainer.train()
        logging.info(f"[{split_name}] train() finished in {time.time()-start:.1f}s")
        # load a handful of examples from the validation split
        # --- Generate ONE sample per dataset in this split, using TRAIN data ---
        with open(os.path.join(split_dir, "out_text.txt"), "a") as f_ptr:
            for ds_name in chosen:
                cfg = TINY_PRESETS[ds_name]["config"]
                gen_field = TINY_PRESETS[ds_name]["generate_prompt_field"]
                train_split_name = TINY_PRESETS[ds_name]["train_split"]

                ds_all = load_dataset(ds_name, cfg)
                train_ds = ds_all[train_split_name] if train_split_name in ds_all else ds_all["train"]
                prompt = str(train_ds[0][gen_field])
                for data_idx in range(len(train_ds)):
                    prompt = str(train_ds[data_idx][gen_field])
                    if len(prompt) > 0:
                        break
                if "wikitext" in ds_name:
                    prompt = " The team returned to the FA Cup final in 1924 , in the second final held at the then new Wembley Stadium . They defeated Aston Villa , winning the club 's second FA Cup . Three years later they won the First Division championship a fourth time in 1926 – 27 , with Hughie Gallacher , one of the most prolific goal scorers in the club 's history , captaining the team . Other key players in this period were Neil Harris , Stan Seymour and Frank Hudspeth . In 1930 , Newcastle United came close to relegation , and at the end of the season Gallacher left the club for Chelsea , and at the same time Andy Cunningham became the club 's first team manager . In 1931 – 32 , the club won the FA Cup a third time . However , a couple of years later , at the end of the 1933 – 34 season , the team were relegated to the Second Division after 35 seasons in the top . Cunningham left as manager and Tom Mather took over . \n"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                gen_ids = model.generate(**inputs, max_new_tokens=250, do_sample=True, top_k=50, top_p=0.95)
                decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

                f_ptr.write(f"\n[{split_name}] Dataset: {ds_name}\n")
                f_ptr.write("Prompt: " + prompt[:200].replace("\n", " ") + "\n")
                f_ptr.write("Generated: " + decoded + "\n")
                f_ptr.write("------------ End of response ------------\n")
        logging.info(f"[{split_name}] sample finished")
        # Save FINAL adapter + tokenizer + compact ft args
        model_to_save = model.module if isinstance(model, DDP) else model
        model_to_save.save_pretrained(split_dir)
        tokenizer.save_pretrained(split_dir)
        with open(os.path.join(split_dir, "ft_args.json"), "w") as f:
            json.dump({
                "base_model_name": args.model_name,
                "lora": {
                    "r": args.lora_r, "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout, "target_modules": target_modules
                },
                "train": {
                    "max_steps": args.max_steps,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "grad_accum": args.gradient_accumulation_steps,
                    "block_size": args.block_size,
                    "seed": args.seed,
                },
                "precision": args.precision,
                "bnb_4bit": args.load_in_4bit, "bnb_8bit": args.load_in_8bit
            }, f, indent=2)
        logging.info(f"[{split_name}] saved final adapter + tokenizer to {split_dir}")
        _free_after_trainer(trainer, model, extra_refs=[train_dataset, targs])
    return

# =========================
#     ARGUMENT PARSER
# =========================
def get_args():
    p = argparse.ArgumentParser("lora_two_splits_trainer", add_help=True)

    # I/O + dataset
    p.add_argument("--output_dir_base", default="./log_dir/lora_two_splits")
    p.add_argument("--frac_overlap", type=float, required=True)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)

    # Raw text source (for tokenization once per model)
    p.add_argument("--hf_dataset_config", default="", help="HF config (if any). Leave blank for default.")
    p.add_argument("--auto_preset", action="store_true", default=False,
                   help="Apply known presets for tiny-* datasets to fill text_field/splits.")
    # Allow blanks so presets can fill these. If provided, explicit values win.
    p.add_argument("--text_field", default="", help="Column that holds raw text. Leave blank to auto-detect.")
    p.add_argument("--train_split", default="", help="HF train split name. Leave blank to auto-detect.")
    p.add_argument("--val_split", default="", help="HF val split name. Leave blank to auto-detect.")

    # Model / tokenizer
    p.add_argument("--model_name", required=True)
    p.add_argument("--hf_token_env", default="my_token", help="Env var holding HF token (or blank to skip)")
    p.add_argument("--hf_cache_env", default="HF_TRANSFORMER_CACHE", help="Env var for HF cache dir (or blank to skip)")

    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--target_modules", default="q_proj,v_proj", help="Comma-separated. Default keeps it lean (Q/V).")

    # BitsAndBytes / memory
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    p.add_argument("--bfloat16_compute", action="store_true", default=False, help="Compute dtype for 4bit path")

    # Trainer / optimization
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_strategy", default="no", choices=["no","steps","epoch"])
    p.add_argument("--precision", choices=["bfloat16","float16","float32"], default="bfloat16")
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)
    p.add_argument("--dataloader_workers", type=int, default=6)

    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
