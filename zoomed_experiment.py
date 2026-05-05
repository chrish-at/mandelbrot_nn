"""
Zoomed minibrot experiment: precision vs capacity.

Trains MLPFourierRes with high-frequency Fourier modes on data sampled
exclusively from the mini-Mandelbrot region, then compares against the
globally-trained model.

Usage:
    python zoomed_experiment.py                         # train hf + render all
    python zoomed_experiment.py --render-only           # skip training
    python zoomed_experiment.py --device cuda:1
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

from data import build_boundary_biased_dataset, smooth_escape_grid
from models import MLPFourierRes

OUTPUT = Path("output")
CKPT = Path("checkpoints")

MINIBROT_XLIM = (-1.82, -1.72)
MINIBROT_YLIM = (-0.05, 0.05)
RENDER_RES = (1920, 1080)

FOURIER_CFG = dict(
    num_feats=512, sigmas=(2.0, 6.0, 10.0),
    hidden_dim=512, num_blocks=20, act="silu", seed=0,
)

FOURIER_HF_CFG = dict(
    num_feats=512, sigmas=(2.0, 6.0, 10.0, 30.0, 100.0),
    hidden_dim=512, num_blocks=20, act="silu", seed=0,
)


def build_zoomed_dataset(n_total=1_000_000, xlim=MINIBROT_XLIM,
                         ylim=MINIBROT_YLIM, seed=0):
    print(f"Building zoomed dataset: {n_total:,} samples in "
          f"xlim={xlim}, ylim={ylim}")
    X, y, ylim_out = build_boundary_biased_dataset(
        n_total=n_total, xlim=xlim, ylim=ylim, seed=seed,
    )
    print(f"  Dataset: {X.shape[0]:,} points, "
          f"y range [{y.min():.4f}, {y.max():.4f}]")
    return X, y


def train_zoomed(X, y, device, cfg, ckpt_name,
                 epochs=100, batch_size=4096, lr=3e-4):
    model = MLPFourierRes(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining MLPFourierRes ({ckpt_name}) on minibrot region "
          f"({n_params:,} params) for {epochs} epochs")
    print(f"  sigmas={cfg['sigmas']}")

    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).unsqueeze(-1).to(device)
    ds = TensorDataset(Xt, yt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    history = []
    t0 = time.time()
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
        epoch_loss = running / count
        history.append(epoch_loss)
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr_now = sched.get_last_lr()[0]
            print(f"  epoch {epoch:3d}/{epochs}  loss={epoch_loss:.6f}"
                  f"  lr={lr_now:.2e}  [{elapsed:.0f}s]")

    CKPT.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT / f"{ckpt_name}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    hist_path = CKPT / f"{ckpt_name}_loss.json"
    with open(hist_path, "w") as f:
        json.dump({"train": history}, f)
    print(f"  Loss history saved: {hist_path}")
    return model, history


def load_model(cfg, ckpt_name, device):
    model = MLPFourierRes(**cfg).to(device)
    path = CKPT / f"{ckpt_name}.pt"
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded {ckpt_name} from {path}")
    return model


@torch.no_grad()
def predict_grid(model, xs, ys, device, batch_rows=64):
    W, H = len(xs), len(ys)
    out = np.empty((H, W), dtype=np.float32)
    xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
    for i in range(0, H, batch_rows):
        j = min(i + batch_rows, H)
        chunk_ys = ys[i:j]
        cy = torch.tensor(chunk_ys, dtype=torch.float32, device=device)
        gx = xs_t.unsqueeze(0).expand(len(chunk_ys), -1)
        gy = cy.unsqueeze(1).expand(-1, W)
        coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        pred = model(coords).squeeze(-1)
        out[i:j] = pred.cpu().numpy().reshape(len(chunk_ys), W)
    return out


def render_all(models, device):
    """Render all comparison and error plots.

    models: dict with keys 'global', 'minibrot', 'minibrot_hf'
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)

    xs = np.linspace(MINIBROT_XLIM[0], MINIBROT_XLIM[1], RENDER_RES[0],
                     dtype=np.float64)
    ys = np.linspace(MINIBROT_YLIM[0], MINIBROT_YLIM[1], RENDER_RES[1],
                     dtype=np.float64)
    extent = [xs[0], xs[-1], ys[-1], ys[0]]

    print("Computing ground truth for minibrot region ...")
    gt = smooth_escape_grid(xs, ys)

    preds, mses = {}, {}
    for key, model in models.items():
        print(f"Predicting with {key} model ...")
        preds[key] = np.clip(predict_grid(model, xs, ys, device), 0, 1)
        mses[key] = float(np.mean((preds[key] - gt) ** 2))

    print(f"\n{'='*60}")
    print(f"MSE on minibrot eval grid ({RENDER_RES[0]}x{RENDER_RES[1]}):")
    for key in models:
        print(f"  {key:20s}: {mses[key]:.6f}")
    if 'global' in mses and 'minibrot_hf' in mses:
        ratio = mses['global'] / mses['minibrot_hf']
        print(f"  Ratio (global / minibrot_hf): {ratio:.1f}x")
    print(f"{'='*60}\n")

    # --- 4-panel comparison (all models) ---
    keys_4 = [k for k in ['global', 'minibrot', 'minibrot_hf'] if k in models]
    ncols = 1 + len(keys_4)
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 7))
    labels = {
        'global': 'Global Fourier',
        'minibrot': 'Minibrot Fourier',
        'minibrot_hf': 'Minibrot Fourier (high-freq)',
    }
    panels = [("Ground Truth", gt)]
    for k in keys_4:
        panels.append((f"{labels[k]} (MSE={mses[k]:.6f})", preds[k]))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, extent=extent, cmap="inferno", vmin=0, vmax=1,
                  aspect="auto")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Re(c)")
        ax.set_ylabel("Im(c)")
    fig.suptitle("Precision vs Capacity: Minibrot Region", fontsize=16, y=1.02)
    fig.tight_layout()
    path_4 = OUTPUT / "zoomed_minibrot_comparison_hf.png"
    fig.savefig(path_4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"4-panel comparison saved: {path_4}")

    # --- 3-panel error map ---
    errors = {k: np.abs(preds[k] - gt) for k in keys_4}
    shared_vmax = max(0.05, float(max(
        np.percentile(err, 99.5) for err in errors.values())))
    n_err = len(keys_4)
    fig, axes = plt.subplots(1, n_err, figsize=(9 * n_err, 7))
    if n_err == 1:
        axes = [axes]
    for ax, k in zip(axes, keys_4):
        err = errors[k]
        im = ax.imshow(err, extent=extent, cmap="hot", vmin=0,
                       vmax=shared_vmax, aspect="auto")
        mean_err = float(err.mean())
        max_err = float(err.max())
        ax.set_title(f"|Error| {labels[k]}\n(mean={mean_err:.4f}, "
                     f"max={max_err:.4f})", fontsize=13)
        ax.set_xlabel("Re(c)")
        ax.set_ylabel("Im(c)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Absolute Error -- Minibrot Region", fontsize=16, y=1.02)
    fig.tight_layout()
    path_err = OUTPUT / "zoomed_minibrot_error_hf.png"
    fig.savefig(path_err, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Error map saved: {path_err}")

    # --- Clean 3-panel comparison (GT | Global | Minibrot HF as "Minibrot Fourier") ---
    if 'minibrot_hf' in models:
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        mse_g = mses['global']
        mse_z = mses['minibrot_hf']
        panels_clean = [
            ("Ground Truth", gt),
            (f"Global Fourier (MSE={mse_g:.6f})", preds['global']),
            (f"Minibrot Fourier (MSE={mse_z:.6f})", preds['minibrot_hf']),
        ]
        for ax, (title, img) in zip(axes, panels_clean):
            ax.imshow(img, extent=extent, cmap="inferno", vmin=0, vmax=1,
                      aspect="auto")
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Re(c)")
            ax.set_ylabel("Im(c)")
        fig.suptitle("Precision vs Capacity: Minibrot Region",
                     fontsize=16, y=1.02)
        fig.tight_layout()
        path_clean = OUTPUT / "zoomed_minibrot_comparison.png"
        fig.savefig(path_clean, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Clean 3-panel comparison saved: {path_clean}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-only", action="store_true",
                        help="Skip training, load existing checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=1_000_000)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.render_only:
        hf_model = load_model(FOURIER_HF_CFG, "fourier_minibrot_hf", device)
    else:
        X, y = build_zoomed_dataset(n_total=args.n_samples)
        hf_model, _ = train_zoomed(
            X, y, device, cfg=FOURIER_HF_CFG,
            ckpt_name="fourier_minibrot_hf", epochs=args.epochs)
        hf_model.eval()

    models = {
        'global': load_model(FOURIER_CFG, "fourier", device),
        'minibrot': load_model(FOURIER_CFG, "fourier_minibrot", device),
        'minibrot_hf': hf_model,
    }

    render_all(models, device)
    print("Done.")


if __name__ == "__main__":
    main()
