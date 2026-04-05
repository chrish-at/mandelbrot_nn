"""
Render chrome-free panel images for the SVG scaling figure.

Generates:
  - full_mandelbrot.png  (global view, no axes/titles)
  - seahorse_gt.png, minibrot_gt.png  (ground truth zooms)
  - seahorse_p{N}.png, minibrot_p{N}.png  (model predictions per config)

Usage:
    python render_clean_panels.py --device cuda:0
    python render_clean_panels.py --render-only   # skip training, load checkpoints
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import smooth_escape_grid, compute_ylim, DEFAULT_XLIM, DEFAULT_MAX_ITER
from models import MLPFourierRes

PANEL_DIR = Path("output/clean_panels")
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
    "seahorse": {
        "xlim": (-0.83, -0.69),
        "ylim": (0.08, 0.22),
    },
    "minibrot": {
        "xlim": (-1.82, -1.72),
        "ylim": (-0.05, 0.05),
    },
}

PANEL_RES = (720, 720)
FULL_RES = (1920, 1080)
DEFAULT_SIGMAS = (2.0, 6.0, 10.0)


def format_params(n):
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def save_clean(array, path, cmap="inferno"):
    """Save a 2D array as a borderless PNG with the given colormap."""
    plt.imsave(str(path), array, cmap=cmap, vmin=0, vmax=1)
    print(f"  Saved {path}")


def make_model(hidden_dim, num_blocks, sigmas):
    return MLPFourierRes(
        num_feats=hidden_dim, sigmas=sigmas,
        hidden_dim=hidden_dim, num_blocks=num_blocks, act="silu", seed=0,
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def ckpt_tag(hidden_dim, num_blocks, suffix=""):
    return f"fourier_h{hidden_dim}_b{num_blocks}{suffix}"


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


def render_full_mandelbrot():
    """Render the full smooth Mandelbrot set without any chrome."""
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    xlim = DEFAULT_XLIM
    ylim = compute_ylim(xlim, FULL_RES)
    xs = np.linspace(xlim[0], xlim[1], FULL_RES[0], dtype=np.float64)
    ys = np.linspace(ylim[0], ylim[1], FULL_RES[1], dtype=np.float64)
    print("Computing full Mandelbrot ground truth ...")
    grid = smooth_escape_grid(xs, ys, DEFAULT_MAX_ITER)
    save_clean(grid, PANEL_DIR / "full_mandelbrot.png")
    return grid


def render_ground_truths():
    """Render ground truth for each zoom view."""
    for name, view in VIEWS.items():
        xlim, ylim = view["xlim"], view["ylim"]
        xs = np.linspace(xlim[0], xlim[1], PANEL_RES[0], dtype=np.float64)
        ys = np.linspace(ylim[0], ylim[1], PANEL_RES[1], dtype=np.float64)
        print(f"Computing ground truth for {name} ...")
        grid = smooth_escape_grid(xs, ys, DEFAULT_MAX_ITER)
        save_clean(grid, PANEL_DIR / f"{name}_gt.png")


def render_model_panels(device, sigmas):
    """Load each checkpoint and render predictions for each zoom view."""
    for cfg in MODEL_CONFIGS:
        h, b = cfg["hidden_dim"], cfg["num_blocks"]
        model = make_model(h, b, sigmas).to(device)
        tag = ckpt_tag(h, b)
        path = CKPT_DIR / f"{tag}.pt"
        model.load_state_dict(torch.load(path, map_location=device,
                                         weights_only=True))
        model.eval()
        n_params = count_params(model)
        label = format_params(n_params)
        print(f"Loaded {tag} (P={n_params:,})")

        for name, view in VIEWS.items():
            xlim, ylim = view["xlim"], view["ylim"]
            xs = np.linspace(xlim[0], xlim[1], PANEL_RES[0], dtype=np.float64)
            ys = np.linspace(ylim[0], ylim[1], PANEL_RES[1], dtype=np.float64)
            pred = np.clip(predict_grid(model, xs, ys, device), 0, 1)
            fname = f"{name}_p{n_params}.png"
            save_clean(pred, PANEL_DIR / fname)

        del model
        torch.cuda.empty_cache()


def write_manifest(sigmas):
    """Write a JSON manifest listing panels and their metadata."""
    import json
    manifest = {
        "full": "full_mandelbrot.png",
        "views": {},
        "sigmas": list(sigmas),
    }
    for name, view in VIEWS.items():
        panels = [{"label": "Ground Truth", "file": f"{name}_gt.png", "params": None}]
        for cfg in MODEL_CONFIGS:
            h, b = cfg["hidden_dim"], cfg["num_blocks"]
            model = make_model(h, b, sigmas)
            n = count_params(model)
            panels.append({
                "label": f"P = {format_params(n)}",
                "file": f"{name}_p{n}.png",
                "params": n,
            })
            del model
        manifest["views"][name] = {
            "xlim": list(view["xlim"]),
            "ylim": list(view["ylim"]),
            "panels": panels,
        }
    path = PANEL_DIR / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=list(DEFAULT_SIGMAS))
    args = parser.parse_args()

    sigmas = tuple(args.sigmas)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sigmas: {sigmas}\n")

    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    render_full_mandelbrot()
    render_ground_truths()
    render_model_panels(device, sigmas)
    write_manifest(sigmas)

    print("\nDone. All panels in", PANEL_DIR)


if __name__ == "__main__":
    main()
