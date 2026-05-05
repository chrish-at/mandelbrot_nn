"""
Parameter-scale comparison across zoom views.

Trains Fourier MLPs at different parameter scales on the high-data (N=5M)
stratified dataset and renders comparison images on seahorse and minibrot
zoom regions.

Usage:
    python scaling_zoom_comparison.py --device cuda:0
    python scaling_zoom_comparison.py --device cuda:1 --sigmas 2.0 6.0 10.0 30.0 --suffix _highfreq
    python scaling_zoom_comparison.py --render-only --suffix _highfreq
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import smooth_escape_grid
from models import MLPFourierRes
from scaling_stratified import get_or_build_master

OUTPUT = Path("output")
CKPT_DIR = Path("checkpoints/scaling_zoom")

MODEL_CONFIGS = [
    {"hidden_dim": 32,  "num_blocks": 2},
    {"hidden_dim": 48,  "num_blocks": 3},
    {"hidden_dim": 64,  "num_blocks": 4},
    {"hidden_dim": 96,  "num_blocks": 6},
    {"hidden_dim": 128, "num_blocks": 8},
    {"hidden_dim": 192, "num_blocks": 10},
]

VIEWS = {
    "zoom1_seahorse": {
        "xlim": (-0.82, -0.7),
        "ylim": (0.08, 0.22),
        "res": (1920, 1080),
    },
    "zoom2_minibrot": {
        "xlim": (-1.82, -1.72),
        "ylim": (-0.05, 0.05),
        "res": (1920, 1080),
    },
}

N_DATA = 5_000_000
MAX_STEPS = 25_000
BATCH_SIZE = 4096
LR = 3e-4
DEFAULT_SIGMAS = (2.0, 6.0, 10.0)


def format_params(n):
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def make_model(hidden_dim, num_blocks, sigmas):
    return MLPFourierRes(
        num_feats=hidden_dim, sigmas=sigmas,
        hidden_dim=hidden_dim, num_blocks=num_blocks, act="silu", seed=0,
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def ckpt_tag(hidden_dim, num_blocks, suffix):
    return f"fourier_h{hidden_dim}_b{num_blocks}{suffix}"


def train_model(hidden_dim, num_blocks, sigmas, X_train, y_train, device,
                suffix=""):
    model = make_model(hidden_dim, num_blocks, sigmas).to(device)
    n_params = count_params(model)
    tag = ckpt_tag(hidden_dim, num_blocks, suffix)

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).unsqueeze(-1).to(device)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE,
                    shuffle=True, drop_last=True)

    steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)
    epochs = max(1, MAX_STEPS // steps_per_epoch)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    t0 = time.time()
    print(f"  Training {tag} (P={n_params:,}, sigmas={sigmas}) "
          f"for {epochs} epochs ...")

    for epoch in range(1, epochs + 1):
        model.train()
        running, count = 0.0, 0
        for xb, yb in dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
            count += xb.size(0)
        sched.step()
        if epoch % 5 == 0 or epoch == epochs:
            print(f"    [{tag}] epoch {epoch:3d}/{epochs}  "
                  f"loss={running / count:.6f}  [{time.time() - t0:.0f}s]")

    wall = time.time() - t0
    print(f"  {tag} done in {wall:.0f}s")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT_DIR / f"{tag}.pt")
    model.eval()
    return model, n_params, tag


def load_model(hidden_dim, num_blocks, sigmas, device, suffix=""):
    model = make_model(hidden_dim, num_blocks, sigmas).to(device)
    tag = ckpt_tag(hidden_dim, num_blocks, suffix)
    path = CKPT_DIR / f"{tag}.pt"
    model.load_state_dict(torch.load(path, map_location=device,
                                     weights_only=True))
    model.eval()
    n_params = count_params(model)
    print(f"  Loaded {tag} (P={n_params:,})")
    return model, n_params, tag


@torch.no_grad()
def predict_grid(model, xs, ys, device, batch_rows=64):
    W, H = len(xs), len(ys)
    out = np.empty((H, W), dtype=np.float32)
    xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
    for i in range(0, H, batch_rows):
        j = min(i + batch_rows, H)
        cy = torch.tensor(ys[i:j], dtype=torch.float32, device=device)
        gx = xs_t.unsqueeze(0).expand(j - i, -1)
        gy = cy.unsqueeze(1).expand(-1, W)
        coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        pred = model(coords).squeeze(-1)
        out[i:j] = pred.cpu().numpy().reshape(j - i, W)
    return out


def render_comparisons(models_info, device, sigmas, suffix=""):
    OUTPUT.mkdir(parents=True, exist_ok=True)

    sigma_str = ", ".join(f"{s:.0f}" for s in sigmas)
    gt_cache = {}

    for view_name, view in VIEWS.items():
        xlim, ylim, res = view["xlim"], view["ylim"], view["res"]
        xs = np.linspace(xlim[0], xlim[1], res[0], dtype=np.float64)
        ys = np.linspace(ylim[0], ylim[1], res[1], dtype=np.float64)
        extent = [xs[0], xs[-1], ys[-1], ys[0]]

        if view_name not in gt_cache:
            print(f"  Computing ground truth for {view_name} ...")
            gt_cache[view_name] = smooth_escape_grid(xs, ys)
        gt = gt_cache[view_name]

        preds = {}
        for model, n_params, tag in models_info:
            print(f"  Predicting {tag} on {view_name} ...")
            preds[tag] = np.clip(predict_grid(model, xs, ys, device), 0, 1)

        ncols = 1 + len(models_info)
        fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5))
        panels = [("Ground Truth", gt)]
        for model, n_params, tag in models_info:
            panels.append((f"P = {format_params(n_params)}", preds[tag]))
        for ax, (title, img) in zip(axes, panels):
            ax.imshow(img, extent=extent, cmap="inferno", vmin=0, vmax=1,
                      aspect="auto")
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Re(c)")
            ax.set_ylabel("Im(c)")

        view_label = view_name.replace("_", " ").title()
        fig.suptitle(
            f"Parameter Scaling  (N=5M, \u03c3=[{sigma_str}])  \u2014  {view_label}",
            fontsize=15, y=1.02)
        fig.tight_layout()
        path = OUTPUT / f"{view_name}_scaling_comparison{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

        errors = {}
        for model, n_params, tag in models_info:
            errors[tag] = np.abs(preds[tag] - gt)
        shared_vmax = max(0.05, float(max(
            np.percentile(err, 99.5) for err in errors.values())))

        n = len(models_info)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, (model, n_params, tag) in zip(axes, models_info):
            err = errors[tag]
            im = ax.imshow(err, extent=extent, cmap="hot", vmin=0,
                           vmax=shared_vmax, aspect="auto")
            mean_err = float(err.mean())
            max_err = float(err.max())
            ax.set_title(
                f"|Error|  P = {format_params(n_params)}\n(mean={mean_err:.4f}, max={max_err:.4f})",
                fontsize=11)
            ax.set_xlabel("Re(c)")
            ax.set_ylabel("Im(c)")
            fig.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle(
            f"Absolute Error  (N=5M, \u03c3=[{sigma_str}])  \u2014  {view_label}",
            fontsize=15, y=1.02)
        fig.tight_layout()
        path = OUTPUT / f"{view_name}_scaling_error{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=list(DEFAULT_SIGMAS),
                        help="Fourier feature sigma scales")
    parser.add_argument("--suffix", default="",
                        help="Suffix for output filenames and checkpoints")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()

    sigmas = tuple(args.sigmas)
    suffix = args.suffix
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sigmas: {sigmas}")
    print(f"Suffix: {suffix!r}\n")

    models_info = []

    if args.render_only:
        for cfg in MODEL_CONFIGS:
            model, n_params, tag = load_model(
                cfg["hidden_dim"], cfg["num_blocks"], sigmas, device,
                suffix=suffix)
            models_info.append((model, n_params, tag))
    else:
        X_train, y_train, X_test, y_test = get_or_build_master()
        print(f"Training data: {X_train.shape[0]:,} samples\n")

        for cfg in MODEL_CONFIGS:
            model, n_params, tag = train_model(
                cfg["hidden_dim"], cfg["num_blocks"], sigmas,
                X_train, y_train, device, suffix=suffix)
            models_info.append((model, n_params, tag))

    print(f"\nRendering zoom comparisons (suffix={suffix!r}) ...")
    render_comparisons(models_info, device, sigmas, suffix=suffix)
    print("\nDone.")


if __name__ == "__main__":
    main()
