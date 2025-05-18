"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os, time, math, pickle, sys, random, torch, json, torchvision, argparse, logging
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

DATA_DIR_BASE = "./data"

def setup_logging(log_file, rank):
    # log_file = os.path.join(exp_path, "log.log")
    
    # Clear previous handlers if re-running script
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,  # Only rank 0 prints INFO logs
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)

    # Add handlers to root logger
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(console_handler)

def _merge_adjacent_chunks(chunks):
    """Merge adjacent (start, end) chunks into maximal contiguous ranges."""
    if not chunks:
        return []
    chunks = sorted(chunks)
    merged = [chunks[0]]
    for start, end in chunks[1:]:
        last_start, last_end = merged[-1]
        if start == last_end:
            merged[-1] = (last_start, end)  # extend the last chunk
        else:
            merged.append((start, end))
    return merged

def _partition_chunks(data_len, num_chunks, overlap_fraction, block_size, stride=None):
    """
    Partition chunks into section1 and section2 with exact overlap_fraction.
    
    Each section will have exactly section_len chunks, where
        section_len = num_chunks // 2.
        
    Overlap is defined as:
        len(shared_chunks) / section_len == overlap_fraction.
    
    The chunk_len is computed as:
        chunk_len = max(block_size, data_len // num_chunks)
        
    Only total_needed candidate chunks (a subset of all possible chunks) are used,
    so that leftover candidates remain unused.
    """
    # Fix the number of chunks per section
    section_len = num_chunks // 2  # This is fixed regardless of overlap_fraction.
    # Compute how many of those chunks should be shared
    shared_count = int(overlap_fraction * section_len)
    # Each section gets: (section_len - shared_count) unique chunks plus shared_count chunks
    # So total candidate chunks needed is:
    total_needed = shared_count + 2 * (section_len - shared_count)

    # Compute chunk_len based solely on data_len, num_chunks, and block_size.
    chunk_len = max(block_size, data_len // num_chunks)
    stride = stride or chunk_len

    # Generate candidate non-overlapping chunks.
    all_chunks = [(i, i + chunk_len) for i in range(0, data_len - chunk_len + 1, stride)]
    
    if len(all_chunks) < total_needed:
        raise ValueError(f"Not enough chunks: need {total_needed}, but only {len(all_chunks)} available.")
    
    # Shuffle candidate pool and use only total_needed of them.
    random.shuffle(all_chunks)
    shared_chunks = all_chunks[:shared_count]
    section1_only = all_chunks[shared_count: shared_count + (section_len - shared_count)]
    section2_only = all_chunks[shared_count + (section_len - shared_count): shared_count + 2 * (section_len - shared_count)]
    
    # Form sections by adding unique chunks and the same shared chunks.
    section1_chunks = section1_only + shared_chunks
    section2_chunks = section2_only + shared_chunks

    # Calculate empirical overlap (at the chunk level)
    actual_overlap = len(set(section1_chunks) & set(section2_chunks)) / float(section_len)
    print(f"actual overlap {actual_overlap:.3f}, desired overlap {overlap_fraction}")
    print(f"section1_chunks: {len(section1_chunks)}, section2_chunks: {len(section2_chunks)}")
    
    return section1_chunks, section2_chunks, shared_chunks

def _split_with_overlap(data, overlap_fraction, block_size, num_chunks=80):
    """
    Randomly splits data into two equal-length sections with a specified overlap fraction,
    ensuring each training block is self-contained within chunks. Length of each chunk is 
    dynamically determined by _partition_chunks

    Args:
        data (list of int): Tokenized dataset.
        overlap_fraction (float): Desired overlap between the two sections (0 <= f <= 1).
        block_size (int): Block/window size used during training.
        num_chunks (int): Total number of chunks used across both sections.

    Returns:
        section1, section2 (list): Token sequences.
        chunk_map: dict with chunk ranges for each section, block-safe.
    """
    assert 0 <= overlap_fraction <= 1, "Invalid overlap_fraction"
    data_len = len(data)

    # total_target_len = data_len // 2  # section1 and section2 will each have <= this many tokens

    # Determine overlap ratio in chunks
    # num_shared = int(num_chunks * overlap_fraction)
    # num_unique = num_chunks - num_shared

    section1_chunks, section2_chunks, shared_chunks = _partition_chunks(
        data_len, num_chunks, overlap_fraction, block_size
    )

    # Shuffle to avoid front-loading shared chunks
    random.shuffle(section1_chunks)
    random.shuffle(section2_chunks)
    
    # Trim chunk ends for block safety
    def _extract_chunks(data, chunks, block_size):
        """Extract all chunks and trim their right ends for block-safe sampling."""
        parts = []
        for start, end in chunks:
            if end - start >= block_size:
                parts.append(data[start:end - (block_size)])  # not end - (block_size-1) due to y being one step ahead of x
        return sum(parts, [])

    # turning chunks into indices
    section1 = _extract_chunks(data, section1_chunks, block_size)
    section2 = _extract_chunks(data, section2_chunks, block_size)

    chunk_map = {
        "section1": section1_chunks,
        "section2": section2_chunks,
    }

    return section1, section2, chunk_map


def make_indices_ds_overlap(ds_size, log_base_path, block_size, frac_overlap=0.5):
    """Computes a dictionary of split to indices mappings for each split so that every split has the same
    prescribed fraction of data that's shared with other splits. For example, ret_dict[0] provides
    the indices of datapoints in split 0. Every class is represented in all splits (hence ds_overlap). 
    Important assumption: the dataset is ordered sequentially
    based on the class label with ds_size data points.
    Args:
        ds_size (int): number of images in the dataset
        log_base_path (str): path to base log directory
        frac_overlap (float): fraction of data is shared between any two split. 
            Computation: Assume 2 splits, s1 = split 1 indices, s2 = split 2 indices, 
            common = s1 & s2 (aka indices that's found in both s1 and s2). len(common) / len(s1) 
            should equal len(common) / len(s2), and len(common) / len(s_whatever) 
            is the frac_overlap
    Returns:
        dict[str, list] 
    """
    path_to_log_dir = os.path.join(log_base_path, "split_indices")
    os.makedirs(path_to_log_dir, exist_ok=True)
    all_idx = list(range(ds_size))

    s1_indices, s2_indices, chunk_map = _split_with_overlap(all_idx, frac_overlap, block_size)
    rd = {0: s1_indices, 1: s2_indices, "chunk_map": chunk_map}

    common_indices = np.array(list(set(rd[0]) & set(rd[1])))
    for split in [0, 1]:
        np.save(os.path.join(path_to_log_dir, f"split_{split:d}.npy"), 
                np.array(rd[split]), allow_pickle=False)
    with open(os.path.join(path_to_log_dir, "chunk_map.pickle"), "wb") as f:
        pickle.dump(rd["chunk_map"], f)
    
    np.save(os.path.join(path_to_log_dir, f"common.npy"), 
            common_indices, allow_pickle=False)
    return rd

# poor man's data loader
def get_batch(split, data_dir, block_size, batch_size, device, chunks):
    if split == "train":
        x, y = get_batch_train(data_dir, block_size, batch_size, chunks)
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    elif split == "val":
        x, y = get_batch_val(data_dir, block_size, batch_size)
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    
def get_batch_val(data_dir, block_size, batch_size):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    return x, y

def get_batch_train(data_dir, block_size, batch_size, chunks):
    # Load token data
    data = np.memmap(os.path.join(data_dir, f'train.bin'), dtype=np.uint16, mode='r')
    # chunks include list of (start,end)
    x_batch, y_batch = [], []
    for _ in range(batch_size):
        start, end = random.choice(chunks)
        max_start = end - block_size - 1
        if max_start <= start:
            raise ValueError(f"Chunk too small for block_size ({block_size}): {(start, end)}")
        i = random.randint(start, max_start)
        x = torch.from_numpy((data[i:i+block_size]).astype(np.int64))
        y = torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64))

        x_batch.append(x)
        y_batch.append(y)
    x = torch.stack(x_batch)
    y = torch.stack(y_batch)
    return x, y
    
def main(args):
    # -----------------------------------------------------------------------------
    # default config values designed to train a gpt2 (124M) on OpenWebText
    # I/O
    out_dir = os.path.join(args.output_dir_base + "_" + args.dataset, f"seed_{args.seed}", f"frac_overlap_{args.frac_overlap}")
    # eval_interval = max(int(0.1 * args.max_iters), 200)
    eval_interval = 100
    early_stop_thresh = int(eval_interval*8)
    log_interval = 50
    eval_iters = 300
    eval_only = False # if True, script exits right after the first eval
    always_save_checkpoint = args.always_save_checkpoint # if True, always save a checkpoint after each eval

    # init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
    # data
    dataset = args.dataset
    data_dir = os.path.join(DATA_DIR_BASE, dataset)
    gradient_accumulation_steps = args.grad_accum_iters # used to simulate larger batch sizes
    batch_size = args.batch_size # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size = args.block_size
    # # model <= this model is probably too large
    # n_layer = 12
    # n_head = 12
    # n_embd = 768
    # dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    # bias = False # do we use bias inside LayerNorm and Linear layers?
    
    # # Hyperparameters
    # n_embd = 64
    # n_head = 4
    # n_layer = 4
    # dropout = 0.0
    # bias = False
    
    n_layer = args.n_layer
    n_head = args.n_head
    n_embd = args.n_embd
    dropout = args.dropout
    bias = args.bias
    
    # adamw optimizer
    learning_rate = args.max_lr # max learning rate
    max_iters = args.max_iters # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = int(args.max_iters * args.warmup_fraction) # how many steps to warm up for
    lr_decay_iters = args.max_iters # should be ~= max_iters per Chinchilla
    min_lr = learning_rate/10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    device = torch.device('cuda')
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    # exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------
    
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_rank = 0
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
        args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        with open(os.path.join(out_dir, "args.json"), "w") as f:
            json.dump(args_dict, f, indent=2)
    torch.manual_seed(args.seed + seed_offset)
    random.seed(args.seed + seed_offset)
    np.random.seed(args.seed + seed_offset)
    
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    # device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    device_type = 'cuda'
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    # ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # dataset splitting
    data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    ds_split_dict = make_indices_ds_overlap(len(data), log_base_path=out_dir, block_size=block_size, 
                                            frac_overlap=args.frac_overlap)

    for n_split, (k, chunk) in enumerate(ds_split_dict["chunk_map"].items()):  # all training code from now on
        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        output_dir_chunk = os.path.join(out_dir, f"split_{n_split}")
        os.makedirs(output_dir_chunk, exist_ok=True)
        setup_logging(os.path.join(output_dir_chunk, "log.log"), ddp_rank)
        logging.info(f"tokens per iteration will be: {tokens_per_iter:,}")

        iter_num = 0
        best_val_loss = 1e9

        # attempt to derive vocab_size from the dataset
        meta_path = os.path.join(data_dir, 'meta.pkl')
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            meta_vocab_size = meta['vocab_size']
            logging.info(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

        # model init
        model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                        bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
        # if init_from == 'scratch':
        # init a new model from scratch
        logging.info("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            logging.info("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
        # crop down the model block size if desired, using model surgery
        if block_size < model.config.block_size:
            model.crop_block_size(block_size)
            model_args['block_size'] = block_size # so that the checkpoint will have the right value
        model.to(device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        # scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
        scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

        # optimizer
        optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
        # if init_from == 'resume':
            # optimizer.load_state_dict(checkpoint['optimizer'])
        checkpoint = None # free up memory

        # compile the model
        if compile:
            logging.info("compiling the model... (takes a ~minute)")
            unoptimized_model = model
            model = torch.compile(model) # requires PyTorch 2.0

        # wrap model into DDP container
        if ddp:
            model = DDP(model, device_ids=[ddp_local_rank])

        # helps estimate an arbitrarily accurate loss over either split using many batches
        @torch.no_grad()
        def estimate_loss():
            out = {}
            model.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    # X, Y = get_batch(split)
                    X, Y = get_batch(split, data_dir, block_size, batch_size, device, chunk)
                    with ctx:
                        logits, loss = model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            model.train()
            return out

        # learning rate decay scheduler (cosine with warmup)
        def get_lr(it):
            # 1) linear warmup for warmup_iters steps
            if it < warmup_iters:
                return learning_rate * (it + 1) / (warmup_iters + 1)
            # 2) if it > lr_decay_iters, return min learning rate
            if it > lr_decay_iters:
                return min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
            return min_lr + coeff * (learning_rate - min_lr)

        # training loop
        # X, Y = get_batch('train') # fetch the very first batch
        X, Y = get_batch('train', data_dir, block_size, batch_size, device, chunk)
        t0 = time.time()
        local_iter_num = 0 # number of iterations in the lifetime of this process
        raw_model = model.module if ddp else model # unwrap DDP container if needed
        running_mfu = -1.0
        no_improvement_counter = 0  # for approx how many intervals has val loss not decreased 
        while True:

            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % eval_interval == 0 and master_process:
                losses = estimate_loss()
                logging.info(f"(eval) step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                log_stats = {
                        "iter": int(iter_num),
                        "train/loss": float(losses['train']),
                        "val/loss": float(losses['val']),
                        "lr": float(lr),
                        "mfu": float(running_mfu*100), # convert to percentage
                    }
                with open(os.path.join(output_dir_chunk, "log.txt"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + '\n')
                if losses['val'] < best_val_loss or always_save_checkpoint:
                    best_val_loss = losses['val']
                    best_iter = iter_num
                    no_improvement_counter = 0
                    if iter_num > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config,
                        }
                        logging.info(f"saving checkpoint to {output_dir_chunk}, iter {iter_num}")
                        torch.save(checkpoint, os.path.join(output_dir_chunk, 'ckpt.pt'))
                else:
                    logging.info(f"not saving checkpoint, iter {iter_num}, best_val_loss={best_val_loss}, val_loss={losses['val']}")
                if losses['val'] > best_val_loss:
                    no_improvement_counter += eval_interval
                if no_improvement_counter > early_stop_thresh:
                    logging.info(f"Early stopping. Loss has not decreased since iter {best_iter}, best_val_loss={best_val_loss}")
                    break
            if iter_num == 0 and eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                # X, Y = get_batch('train')
                X, Y = get_batch('train', data_dir, block_size, batch_size, device, chunk)
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % log_interval == 0 and master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * gradient_accumulation_steps
                # if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                logging.info(f"(training) iter {iter_num}: loss {lossf:.4f}, {dt:.4f} seconds per iter, mfu {running_mfu*100:.2f}%")
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > max_iters or args.test_run == True:
                if master_process:
                    losses = estimate_loss()
                    logging.info(f"(final eval) step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                    log_stats = {
                        "iter": int(iter_num),
                        "train/loss": float(losses['train']),
                        "val/loss": float(losses['val']),
                        "lr": float(lr),
                        "mfu": float(running_mfu*100), # convert to percentage
                    }
                    with open(os.path.join(output_dir_chunk, "log.txt"), "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + '\n')
                    if losses['val'] < best_val_loss or always_save_checkpoint:
                        best_val_loss = losses['val']
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config,
                        }
                        torch.save(checkpoint, os.path.join(output_dir_chunk, 'ckpt.pt'))
                        logging.info(f"saving final checkpoint to {output_dir_chunk}, iter {iter_num}")
                    else:
                        logging.info(f"(final eval) not saving checkpoint, iter {iter_num}, best_val_loss={best_val_loss}, val_loss={losses['val']}")
                break  # done training
    if ddp:
        destroy_process_group()
    return

def get_args_parser():
    parser = argparse.ArgumentParser("nanogpt_ds_overlap", add_help=False)
    parser.add_argument("--output_dir_base", default="./log_dir/nanogpt_ds_overlap")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=0, help="seed used for initialization")
    parser.add_argument("--frac_overlap", type=float, required=True, help="Fractional overlap")
    
    # training. default max_iters and max_lr taken from https://wandb.ai/bkkaggle/lm-finetuning/reports/Pretraining-a-124-M-Parameter-GPT-2-Language-Model--VmlldzoyMjg4NzA 
    parser.add_argument("--max_iters", type=int, default=5000, help="Max number of iterations")
    parser.add_argument("--max_lr", type=float, default=1e-3, help="Max learning rate")
    parser.add_argument("--warmup_fraction", type=float, default=0.1, help="Fraction of max_iters allocated for lr warmup")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch sise (not effective batch size)")
    parser.add_argument("--grad_accum_iters", type=int, default=8, help="Iters for which to accumulate gradient")
    parser.add_argument("--block_size", type=int, default=256, help="Context size")
    parser.add_argument("--always_save_checkpoint", action="store_true", default=False, help="Save ckpt regardless of val loss")
    
    # model config
    parser.add_argument("--n_layer", type=int, default=6, help="Number of layers in GPT2 model")
    parser.add_argument("--n_head", type=int, default=4, help="Number of heads in GPT2 model")
    parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension in GPT2 model")
    parser.add_argument("--dropout", type=float, default=0.2, help="probability of applying dropout")
    parser.add_argument("--bias", action="store_true", default=False, help="do we use bias inside LayerNorm and Linear layers?")

    parser.add_argument("--test_run", action="store_true", default=False, help="Test run (only one iter)")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)