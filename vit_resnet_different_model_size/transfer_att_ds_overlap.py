import ffcv

import sys, os, argparse, tqdm, timm, pickle
import torch as ch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image

from train_ds_overlap import make_dataloaders as make_dataloaders_ds_overlap

def load_eval_loader(dataset_name: str):
    batch_size, num_workers = 100, 1
    # We keep torch_float32=True as in your scaffold (stable gradients)
    if dataset_name == "tinyimagenet":
        dl = make_dataloaders_ds_overlap(
            None, always_rand_order=True, torch_float32=True,
            train_dataset="./tinyimagenet_ffcv_data/train.beton",
            val_dataset="./tinyimagenet_ffcv_data/val.beton",
            batch_size=batch_size, num_workers=num_workers
        )[0]['test']
        num_classes = 200
    elif dataset_name == "cifar100":
        dl = make_dataloaders_ds_overlap(
            None, always_rand_order=True, torch_float32=True,
            train_dataset="./cifar100_ffcv_data/train.beton",
            val_dataset="./cifar100_ffcv_data/val.beton",
            batch_size=batch_size, num_workers=num_workers
        )[0]['test']
        num_classes = 100
    elif dataset_name == "cifar10":
        dl = make_dataloaders_ds_overlap(
            None, always_rand_order=True, torch_float32=True,
            train_dataset="./cifar10_ffcv_data/train.beton",
            val_dataset="./cifar10_ffcv_data/val.beton",
            batch_size=batch_size, num_workers=num_workers
        )[0]['test']
        num_classes = 10
    else:
        raise ValueError("dataset must be one of: tinyimagenet, cifar100, cifar10")
    return dl, num_classes

def build_model(model_name: str, num_classes: int, device):
    m = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    return m.to(device).eval()

# -----------------------
# Attack: MI-FGSM (Linf)
# -----------------------

def _proj_linf_ball(x_star, x, eps_norm):
    # project x_star onto the L_inf ball of radius eps_norm around x
    return ch.clamp(x_star, x - eps_norm, x + eps_norm)

def mi_fgsm(model, x, y, eps_px=8/255, steps=10, mu=1.0, x_min=-1.0, x_max=(255/127)-1.0):
    """
    MI-FGSM, see Algorithm 1 in Boosting Adversarial Attacks with Momentum (Dong et al, CVPR 2018)
    assumes inputs are normalised by the FFCV pipeline:
        x_norm = (x_px - 127)/127
    eps_px is in pixel scale (0..1), need to convert to normalized units.
    """
    model = model.eval()
    # Convert epsilon from pixel scale to your normalized scale
    eps_norm = eps_px * 255.0 / 127.0
    # Algorithm 1, line 1: alpha = eps / T   (also in normalized units)
    alpha_norm = eps_norm / float(steps)

    # Line 2: g0 = 0 ; x0* = x
    g = ch.zeros_like(x, device=x.device)
    x_star = x.detach().clone()
    for _ in range(steps):
        x_star.requires_grad_(True)
        # Line 4: compute grad wrt current x*
        logits = model(x_star)
        loss = F.cross_entropy(logits, y, reduction='mean')
        grad = ch.autograd.grad(loss, x_star, retain_graph=False, create_graph=False)[0]

        # Line 5: momentum update with L1-normalized gradient
        # normalize per-sample by ||grad||_1
        grad_l1 = grad.abs().reshape(grad.size(0), -1).sum(dim=1).reshape(-1,1,1,1).clamp(min=1e-8)
        g = mu * g + grad / grad_l1

        # Line 6: x_{t+1}* = x_t* + alpha * sign(g_{t+1})
        x_star = x_star.detach() + alpha_norm * g.sign()

        # Project to L_inf ball around the clean x, then clip to valid range
        x_star = _proj_linf_ball(x_star, x, eps_norm)
        x_star = ch.clamp(x_star, x_min, x_max)
    return x_star.detach().clone()

def denorm_to_uint01(x_norm):
    """
    Convert normalized (x_px-127)/127 back to [0,1]
    """
    x_px = x_norm * 127.0 + 127.0
    return ch.clamp(x_px / 255.0, 0.0, 1.0)  # [0,1]

def get_args():
    p = argparse.ArgumentParser("adv attack", add_help=True)
    p.add_argument("--log_base_dir", type=str, default="./log_dir/ds_overlap")
    p.add_argument("--dataset", type=str, required=True, choices=["cifar10","cifar100","tinyimagenet"])
    p.add_argument("--model", type=str, required=True)  # e.g. resnet18
    p.add_argument("--eps", type=float, default=1/255, help="epsilon in pixel scale (0..1)")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--mu", type=float, default=1.0, help="momentum for MI-FGSM")
    p.add_argument("--max_adv_images", type=int, default=500, help="How many images to generate per (seed, split)")
    return p.parse_args()

def main():
    args = get_args()
    device = ch.device("cuda:0")

    eval_dl, num_classes = load_eval_loader(args.dataset)

    base_dir = os.path.join(args.log_base_dir, f"{args.dataset}_{args.model}")
    os.makedirs(base_dir, exist_ok=True)

    path_to_log = os.path.join(base_dir, "transfer_attack.csv")
    with open(path_to_log, "a") as f:
        f.write(f"seed,frac_overlap,normal_asr,transfer_asr\n")
    for seed in tqdm.tqdm(range(4), desc="Seeds"):
        seed_dir = os.path.join(base_dir, f"seed_{seed:d}")
        for p in os.listdir(seed_dir):
            split_dir0 = os.path.join(seed_dir, p, "split_0")
            split_dir1 = os.path.join(seed_dir, p, "split_1")
            
            path_to_splits = os.path.join(seed_dir, p, "split_indices")
            s0 = set(list(np.load(os.path.join(path_to_splits, "split_0.npy"))))
            s1 = set(list(np.load(os.path.join(path_to_splits, "split_1.npy"))))
            frac_overlap = len(s1 & s0) / max(1, len(s0))

            # Load source split_0 and victim split_1 models
            m_src = build_model(args.model, num_classes, device).eval()
            m_src.load_state_dict(ch.load(os.path.join(split_dir0, "checkpoint.pth"), map_location=device))
            m_victim = build_model(args.model, num_classes, device).eval()
            m_victim.load_state_dict(ch.load(os.path.join(split_dir1, "checkpoint.pth"), map_location=device))

            # Generate up to max_adv_images from eval loader
            adv_list, clean_list, lab_list = [], [], []
            total = 0
            for ims, labs in eval_dl:
                # ims already normalized by your FFCV pipeline
                x_adv = mi_fgsm(m_src, ims, labs, eps_px=args.eps, steps=args.steps, mu=args.mu)
                adv_list.append(x_adv.detach().cpu())
                clean_list.append(ims.detach().cpu())
                lab_list.append(labs.detach().cpu())
                total += ims.size(0)
                if total >= args.max_adv_images:
                    break

            adv_tensor = ch.cat(adv_list)[:args.max_adv_images]     # [N,3,H,W], normalized, ready-to-use
            clean_tensor = ch.cat(clean_list)[:args.max_adv_images]   # same normalization
            labels_tensor = ch.cat(lab_list)[:args.max_adv_images]

            with ch.no_grad():
                transfer_asr = float((m_victim(adv_tensor.to(device)).argmax(1).cpu() != labels_tensor).float().mean())
                normal_asr = float((m_src(adv_tensor.to(device)).argmax(1).cpu() != labels_tensor).float().mean())
            # Save pickle, normalized tensors, ready to feed to victim
            out_pickle = os.path.join(seed_dir, p, f"adv_split0_to_split1_eps{args.eps}_steps{args.steps}_n{adv_tensor.size(0)}.pkl")
            payload = {
                "dataset": args.dataset, "model": args.model, "seed_dir": seed_dir,
                "subdir": p, "frac_overlap": float(frac_overlap), "eps": float(args.eps),
                "steps": int(args.steps), "mu": float(args.mu),
                "tensors_norm_adv": adv_tensor,     # normalized (x_px-127)/127
                "tensors_norm_clean": clean_tensor, # normalized
                "labels": labels_tensor,            # int64
                # transfer ASR over the saved batch
                "transfer_asr_estimate": transfer_asr,
            }
            with open(out_pickle, "wb") as f:
                pickle.dump(payload, f)

            with open(path_to_log, "a") as f:
                f.write(f"{seed},{frac_overlap},{normal_asr:g},{transfer_asr:g}\n")
            
            # Save two example pairs
            ex_count = 2
            ex_clean = denorm_to_uint01(clean_tensor[:ex_count])  # [0,1] for PNG
            ex_adv = denorm_to_uint01(adv_tensor[:ex_count])

            for i in range(ex_count):
                save_image(ex_clean[i], os.path.join(split_dir0, f"example_{i}_clean.png"))
                save_image(ex_adv[i], os.path.join(split_dir0, f"example_{i}_adv.png"))

            print(f"[Saved] {out_pickle}")
            print(f"[Examples] Two images saved into {split_dir0}")

    print("Done.")

if __name__ == "__main__":
    main()
