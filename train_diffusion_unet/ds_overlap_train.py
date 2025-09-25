import ffcv

import os, json, time, math, random, torchvision, pickle
from typing import List, Optional, Dict
from argparse import ArgumentParser
from datetime import datetime
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch as ch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (RandomHorizontalFlip, Cutout, RandomTranslate, Convert,
                             ToDevice, ToTensor, ToTorchImage, RandomBrightness, RandomContrast, RandomSaturation)
from ffcv.transforms.common import Squeeze

from fastargs import get_current_config, Param, Section
from fastargs.decorators import param

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_cosine_schedule_with_warmup

# ---------- GLOBAL DATASET MAP (edit paths here) ----------
# Point to your existing .beton files and optional split-source roots.
DATASETS: Dict[str, Dict[str, str]] = {
    # CIFAR-10 (32x32, 10 classes)
    "cifar10": {
        "train_beton": "./cifar10_ffcv/cifar10_ffcv_data/train.beton",
        "val_beton":   "./cifar10_ffcv/cifar10_ffcv_data/val.beton",
        "split_src":   "cifar10",  # special flag, see make_indices_ds
        "real_dir":    "./ds_overlap_fid/cifar10_saved_png",
        "num_classes": 10,
        "labels":      "./log_dir/cifar10/labels.pkl",
    },
    # TinyImageNet (64x64, 200 classes)
    "tinyimagenet": {
        "train_beton": "./tinyimagenet_ffcv/tinyimagenet_ffcv_data/train.beton",
        "val_beton":   "./tinyimagenet_ffcv/tinyimagenet_ffcv_data/val.beton",
        "split_src":   "/path/to/tiny_imagenet_200/train",  # adjust if needed
        "real_dir":    "./ds_overlap_fid/tinyimagenet_saved_png",
        "num_classes": 200,
        "labels":      "./log_dir/tinyimagenet/labels.pkl",
    },
}

# ---------------- FASTARGS ----------------
Section('training', 'Diffusion Hparams').params(
    steps=Param(int, 'Total training steps (per split)', default=300_000),
    batch_size=Param(int, 'Batch size', default=512),
    lr=Param(float, 'AdamW learning rate', default=2e-4),
    weight_decay=Param(float, 'AdamW weight decay', default=0.0),
    ema_decay=Param(float, 'EMA decay', default=0.999),
    warmup=Param(int, 'Warmup steps', default=10_000),
    label_dropout=Param(float, 'Classifier-free guidance dropout', default=0.1),
    num_workers=Param(int, 'FFCV workers', default=8),
    seed=Param(int, 'Seed', default=0),
    grad_accum_iters=Param(int, "Gradient accumulation iters", default=1)
)

Section('data', 'Dataset & splits').params(
    dataset_name=Param(str, 'Name in DATASETS dict (cifar10|tinyimagenet)', required=True),
    class_frac_overlap=Param(float, 'Frac overlap between splits', required=True),
    variable_ds_size=Param(bool, 'Allow variable ds size per split', default=False),
    img_size=Param(int, '32 (CIFAR10) or 64 (TinyImageNet)', required=True),
)

Section('output', 'Output control').params(
    output_root=Param(str, 'Root dir for logs/checkpoints', required=True),
    n_fid=Param(int, 'Images for FID (gen & real)', default=10_000),
    log_every=Param(int, 'Log every N steps', default=1000),
)


def loader(path: str):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

# ---------------- DATALOADERS (normalize -> [-1,1]) ----------------
@param('training.batch_size')
@param('training.num_workers')
def make_dataloaders(train_indices,
                     always_rand_order: bool=False,
                     torch_float32: bool=False,
                     batch_size: int=None,
                     num_workers: int=None,
                     train_dataset: str=None,
                     val_dataset: str=None):
    paths = {'train': train_dataset, 'test': val_dataset}
    start_time = time.time()
    loaders = {}

    # Decode -> [0..255] float -> Normalize to [-1,1] using (x-127.5)/127.5
    mean_255 = [127.5, 127.5, 127.5]
    std_255  = [127.5, 127.5, 127.5]

    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(ch.device('cuda:0')), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice(ch.device('cuda:0'), non_blocking=True),
            ToTorchImage(),
            Convert(ch.float32 if torch_float32 else ch.float16),
            torchvision.transforms.Normalize(mean_255, std_255),  # -> [-1,1]
        ])

        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL
        if always_rand_order:
            ordering = OrderOption.RANDOM

        loaders[name] = Loader(
            paths[name],
            batch_size=batch_size,
            num_workers=num_workers,
            order=ordering,
            drop_last=(name == 'train'),
            indices=train_indices if name == "train" else None,
            pipelines={'image': image_pipeline, 'label': label_pipeline}
        )
    return loaders, start_time

# ---------------- SPLITS (dataset_name required) ----------------
def make_indices_ds(ds_size,
                    log_base_path,
                    class_frac_overlap=0.5,
                    num_classes=10,
                    variable_ds_size=False,
                    dataset_name: str="cifar10") -> dict[str, list]:
    """
    Writes split_indices/{split_*.npy, common.npy, info.json} and returns {0: idxs, 1: idxs}.
    The split source depends on dataset_name:
      - 'cifar10': uses torchvision.datasets.CIFAR10(train=True)
      - 'tinyimagenet': uses ImageFolder on DATASETS[dataset_name]['split_src']
    """
    os.makedirs(os.path.join(log_base_path, "split_indices"), exist_ok=True)
    info_path = os.path.join(log_base_path, "split_indices", "info.json")
    with open(info_path, "w") as f:
        json.dump({
            "class_frac_overlap": class_frac_overlap,
            "variable_ds_size": variable_ds_size,
            "dataset_name": dataset_name
        }, f, indent=2)

    # Build the dataset to pass into split_dataset
    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10("/tmp", train=True, download=True)
    elif dataset_name == "tinyimagenet":
        root = DATASETS["tinyimagenet"]["split_src"]
        dataset = torchvision.datasets.ImageFolder(root, transform=None, extensions="jpeg", loader=loader)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    # External splitter
    import sys
    sys.path.append("../data_splitting")
    from data_splitting import split_dataset

    try:
        with open(DATASETS[dataset_name]["labels"], "rb") as f:
            labels = pickle.load(f)[f"labels_{dataset_name}"]
    except FileNotFoundError:
        print("Run save_labels.py first")
        raise FileNotFoundError
    ret = split_dataset(dataset, class_frac_overlap, split_method="horizontal",
                        mode="unconstrained" if variable_ds_size else "constrained", labels=labels)
    rd = {0: ret["s1_indices"], 1: ret["s2_indices"]}
    common = np.array(list(set(rd[0]) & set(rd[1])))

    base = os.path.join(log_base_path, "split_indices")
    np.save(os.path.join(base, "split_0.npy"), np.array(rd[0]), allow_pickle=False)
    np.save(os.path.join(base, "split_1.npy"), np.array(rd[1]), allow_pickle=False)
    np.save(os.path.join(base, "common.npy"),   common, allow_pickle=False)
    return rd

# ---------------- MODEL ----------------
def make_small_cond_unet(img_size: int, num_classes: int, base: int = 96) -> UNet2DModel:
    if img_size == 32:
        downs = ('DownBlock2D','AttnDownBlock2D','AttnDownBlock2D')
        ups   = ('AttnUpBlock2D','AttnUpBlock2D','UpBlock2D')
        outs  = (base, base*2, base*2)
    else:
        downs = ('DownBlock2D','DownBlock2D','AttnDownBlock2D','AttnDownBlock2D')
        ups   = ('AttnUpBlock2D','AttnUpBlock2D','UpBlock2D','UpBlock2D')
        outs  = (base, base, base*2, base*2)
    model = UNet2DModel(
        sample_size=img_size,
        in_channels=3, out_channels=3,
        block_out_channels=outs,
        layers_per_block=2,
        down_block_types=downs,
        up_block_types=ups,
        class_embed_type='timestep',
        num_class_embeds=num_classes+1,  # +1 for CFG during training
        attention_head_dim=8,
        dropout=0.1
    ).to(memory_format=ch.channels_last).cuda()
    print("num_params:", sum(p.numel() for p in model.parameters()))
    return model

# ---------------- TRAIN (inputs already in [-1,1]) ----------------
def diffusion_train_loop(unet, ddpm_sched, loaders, cfg, outdir):
    scaler = GradScaler('cuda')
    grad_accum_iters = cfg['training.grad_accum_iters']
    effective_bsz = cfg['training.batch_size'] * grad_accum_iters
    ref_bsz, ref_lr = 512, cfg['training.lr']
    lr = ref_lr * (effective_bsz / ref_bsz)
    print(f"effective_bsz={effective_bsz}, new lr={lr}")
    opt = AdamW(unet.parameters(), lr=lr, weight_decay=cfg['training.weight_decay'])
    ema = EMAModel(unet.parameters(), decay=cfg['training.ema_decay'])
    lr_sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=cfg['training.warmup'], num_training_steps=int(cfg['training.steps']*1.1)
    )

    unet.train()
    steps = cfg['training.steps']
    total = 0
    iters_per_epoch = len(loaders['train'])
    epochs = math.ceil(steps / max(1, iters_per_epoch))
    os.makedirs(outdir, exist_ok=True)
    
    pbar = tqdm(total=steps, desc='train', ncols=100)
    opt.zero_grad(set_to_none=True)
    microsteps = 0
    for _ in range(epochs):
        for images, labels in loaders['train']:
            # images are in [-1,1]; keep them there for training
            bsz = images.size(0)
            t = ch.randint(0, ddpm_sched.config.num_train_timesteps, (bsz,), device=images.device, dtype=ch.long)
            noise = ch.randn_like(images)

            # We operate in [-1,1]; DDPM math is scale-agnostic here for the epsilon loss
            noisy = ddpm_sched.add_noise(images, noise, t)

            # classifier-free guidance (drop some labels)
            drop_mask = (ch.rand(bsz, device=labels.device) < cfg['training.label_dropout'])
            labels_all = labels.clone().long()
            labels_all[drop_mask] = UNCOND_ID  # null class
            
            with autocast('cuda'):
                eps_pred = unet(noisy, t, class_labels=labels_all).sample
                loss_micro = F.mse_loss(eps_pred, noise)
                loss = loss_micro / grad_accum_iters  # normalize for accumulation

            # guard inputs/scheduler output
            if not ch.isfinite(noisy).all().item():
                with open(os.path.join(outdir, "log.txt"), "a") as f:
                    f.write(f"iter={total}, NON-FINITE in 'noisy'\n")
                raise RuntimeError("Non-finite in scheduler output")
            if not ch.isfinite(loss_micro).all().item():
                with open(os.path.join(outdir, "log.txt"), "a") as f:
                    f.write(f"iter={total}, loss_micro={float(loss_micro.detach().float())} is not finite\n")
                raise RuntimeError("Non-finite loss_micro detected")
            scaler.scale(loss).backward()

            # step every grad_accum_iters micro-batches
            if (microsteps + 1) % grad_accum_iters == 0:
                scaler.unscale_(opt)
                ch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                lr_sched.step()
                ema.step(unet.parameters())

                total += 1  # count optimizer (global) steps
                microsteps = 0
                if total % cfg['output.log_every'] == 0:
                    with open(os.path.join(outdir, "log.log"), "a") as fptr:
                        # log the *unscaled* micro loss from the last micro-batch
                        json.dump({"iter": total, "loss": float(loss_micro.item())}, fptr)
                        fptr.write("\n")
                    pbar.update(cfg['output.log_every'])
                    ch.save({"iter":total, "model":unet.state_dict()}, os.path.join(outdir, f"ckpt.pth"))
                    ch.save({"iter":total, "model":ema.state_dict()}, os.path.join(outdir, f"ema.pt"))
                if total >= steps:
                    break
            microsteps += 1
    # Use EMA weights for sampling/eval
    ema.copy_to(unet.parameters())
    ch.save(unet.state_dict(), os.path.join(outdir, "ckpt.pth"))

# ---------------- DDIM SAMPLING (returns [-1,1]) ----------------
@ch.no_grad()
def ddim_sample_labels(unet, ddim_sched, class_labels, img_size, num_classes, guidance_scale=3.0, device='cuda'):
    """
    class_labels: LongTensor of shape [N] with values in [0, num_classes-1]
    Returns images in [-1, 1], shape [N, 3, H, W]
    """
    class_labels = class_labels.to(device=device, dtype=ch.long)
    uncond_labels = ch.full_like(class_labels, UNCOND_ID)

    ddim_sched.set_timesteps(50, device=device)
    x = ch.randn(class_labels.shape[0], 3, img_size, img_size, device=device)

    for t in ddim_sched.timesteps:
        eps_uncond = unet(x, t, class_labels=uncond_labels).sample
        eps_cond   = unet(x, t, class_labels=class_labels).sample
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        x = ddim_sched.step(eps, t, x).prev_sample

    return x.clamp(-1, 1)

# ---------------- SAVE/LOAD FOR FID ----------------
def save_images_as_png(tensor_m1p1, out_dir, start_idx=0):
    os.makedirs(out_dir, exist_ok=True)
    # [-1,1] -> [0,255]
    arr = ((tensor_m1p1 * 0.5 + 0.5).mul(255).round().clamp(0,255)).to(ch.uint8).permute(0,2,3,1).cpu().numpy()
    from PIL import Image
    for i in range(arr.shape[0]):
        Image.fromarray(arr[i]).save(os.path.join(out_dir, f"{start_idx+i:06d}.png"))
    return


if __name__ == "__main__":
    cfg = get_current_config()
    parser = ArgumentParser(description='ffcv')
    cfg.augment_argparse(parser)
    cfg.collect_argparse_args(parser)
    cfg.validate(mode='stderr')

    ch.manual_seed(cfg['training.seed']); np.random.seed(cfg['training.seed']); random.seed(cfg['training.seed'])

    dsname = cfg['data.dataset_name']
    if dsname not in DATASETS:
        raise ValueError(f"dataset_name '{dsname}' not found in DATASETS. Keys: {list(DATASETS.keys())}")

    # Resolve beton paths from the global map
    train_beton = DATASETS[dsname]['train_beton']
    val_beton   = DATASETS[dsname]['val_beton']
    UNCOND_ID   = DATASETS[dsname]['num_classes']

    # Layout
    log_base_path = os.path.join(
        cfg['output.output_root'], cfg['data.dataset_name'],
        f"seed_{cfg['training.seed']:d}",
        f"frac_overlap_{cfg['data.class_frac_overlap']:g}"
    )
    os.makedirs(log_base_path, exist_ok=False)
    with open(os.path.join(log_base_path, "args.json"), "w") as f:
        cfg.summary(f)

    # Splits (writes split_indices/* and returns {0: idxs, 1: idxs})
    indices_dict = make_indices_ds(
        ds_size=None,
        log_base_path=log_base_path,
        class_frac_overlap=cfg['data.class_frac_overlap'],
        num_classes=None,
        variable_ds_size=cfg['data.variable_ds_size'],
        dataset_name=dsname
    )

    # Schedulers
    ddpm_sched = DDPMScheduler(num_train_timesteps=1000)   # training
    ddim_sched = DDIMScheduler(num_train_timesteps=1000)   # sampling

    for split in indices_dict.keys():   # 0,1
        split_dir = os.path.join(log_base_path, f"split_{split}")
        os.makedirs(split_dir, exist_ok=True)

        # YOUR dataloaders; pass beton paths resolved from DATASETS
        loaders, start_time = make_dataloaders(
            indices_dict[split],
            train_dataset=train_beton,
            val_dataset=val_beton,
            torch_float32=True,
            batch_size=cfg['training.batch_size'],
            num_workers=cfg['training.num_workers'],
        )

        # Model
        unet = make_small_cond_unet(cfg['data.img_size'], DATASETS[dsname]['num_classes'])

        # Train
        diffusion_train_loop(unet, ddpm_sched, loaders, cfg, split_dir)

        # Generate with DDIM
        gen_dir = os.path.join(split_dir, "generated")
        real_dir = DATASETS[dsname]['real_dir']
        n_fid = int(cfg['output.n_fid'])

        if n_fid > 0 and split == 0:
            print(f"Outputting {n_fid} images")
            saved = 0
            os.makedirs(gen_dir, exist_ok=True)
            all_labels = ch.randint(0, DATASETS[dsname]['num_classes'], (n_fid,), device='cuda')
            max_bs = cfg['training.batch_size']
            start_gen = time.time()
            for i in range(0, n_fid, max_bs):
                j = min(i + max_bs, n_fid)                # <-- handles the tail batch
                batch_labels = all_labels[i:j]
                imgs = ddim_sample_labels(
                    unet, ddim_sched, class_labels=batch_labels,
                    img_size=cfg['data.img_size'],
                    num_classes=DATASETS[dsname]['num_classes'],
                    guidance_scale=3.0, device='cuda'
                )
                save_images_as_png(imgs, gen_dir, start_idx=i)  # index from i, so filenames end at n_fid-1
                print(f"Gen images: saved={saved}, time-elapsed={time.time()-start_gen:.2f}")
            print(f"Done outputting {n_fid} images")
        with open(os.path.join(split_dir, "log.txt"), "a") as f:
            f.write(f"Total time (s): {time.time() - start_time:.2f}\n")
