"""
Render predictions, ground-truth, and error maps for trained Mandelbrot models.

Usage:
    python render.py
    python render.py --target discrete
    python render.py --models baseline fourier gated_bilinear gated_bilinear_tied
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import smooth_escape_grid, discrete_escape_grid, compute_ylim
from models import MLPRes, MLPFourierRes, MLPGatedRes, MLPFourierGatedRes

OUTPUT = Path("output")
CKPT = Path("checkpoints")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

VIEWS = {
    "global": {"xlim": (-2.4, 1.0), "res": (1920, 1080)},
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

_GATED_DEFAULTS = dict(hidden_dim=512, num_blocks=20)
_BILINEAR_DEEP_DEFAULTS = dict(
    hidden_dim=128, num_blocks=100, gate_type="bilinear",
    weight_tie=True, use_layernorm=False, in_act="none",
)


def _parse_model_key(key):
    """Parse a model key like 'gated_bilinear_tied' into constructor args."""
    if key == "bilinear_deep":
        return "bilinear_deep", {}, key
    parts = key.split("_")
    base = parts[0]
    if base == "fourier" and len(parts) > 1 and parts[1] == "gated":
        base = "fourier_gated"
        parts = parts[2:]
    elif base == "gated":
        parts = parts[1:]
    else:
        return base, {}, key

    gate_type = parts[0] if parts else "bilinear"
    weight_tie = "tied" in parts
    return base, dict(gate_type=gate_type, weight_tie=weight_tie), key


def load_model(key, target="smooth"):
    base, kwargs, _ = _parse_model_key(key)
    if base == "baseline":
        m = MLPRes(hidden_dim=512, num_blocks=20, act="silu")
    elif base == "fourier":
        m = MLPFourierRes(
            num_feats=512, sigmas=(2.0, 6.0, 10.0),
            hidden_dim=512, num_blocks=20, act="silu", seed=0,
        )
    elif base == "gated":
        m = MLPGatedRes(**kwargs, **_GATED_DEFAULTS)
    elif base == "fourier_gated":
        m = MLPFourierGatedRes(
            num_feats=512, sigmas=(2.0, 6.0, 10.0), seed=0,
            **kwargs, **_GATED_DEFAULTS,
        )
    elif base == "bilinear_deep":
        m = MLPGatedRes(**_BILINEAR_DEEP_DEFAULTS)
    else:
        raise ValueError(f"Unknown model key: {key}")

    stem = key
    if target == "discrete":
        stem += "_discrete"
    path = CKPT / f"{stem}.pt"
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    m.to(DEVICE).eval()
    return m


@torch.no_grad()
def predict_grid(model, xs, ys, batch_rows=64, apply_sigmoid=False):
    W, H = len(xs), len(ys)
    out = np.empty((H, W), dtype=np.float32)
    xs_t = torch.tensor(xs, dtype=torch.float32, device=DEVICE)

    for i in range(0, H, batch_rows):
        j = min(i + batch_rows, H)
        chunk_ys = ys[i:j]
        cy = torch.tensor(chunk_ys, dtype=torch.float32, device=DEVICE)
        gx = xs_t.unsqueeze(0).expand(len(chunk_ys), -1)
        gy = cy.unsqueeze(1).expand(-1, W)
        coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        logits = model(coords).squeeze(-1)
        if apply_sigmoid:
            logits = torch.sigmoid(logits)
        out[i:j] = logits.cpu().numpy().reshape(len(chunk_ys), W)
    return out


def make_grid(view):
    xlim = view["xlim"]
    res = view["res"]
    if "ylim" in view:
        ylim = view["ylim"]
    else:
        ylim = compute_ylim(xlim, res)
    xs = np.linspace(xlim[0], xlim[1], res[0]).astype(np.float64)
    ys = np.linspace(ylim[0], ylim[1], res[1]).astype(np.float64)
    return xs, ys


def render_comparison(view_name, view, models, gt_cache, target="smooth"):
    xs, ys = make_grid(view)
    names = list(models.keys())
    n = len(names)

    if view_name not in gt_cache:
        print(f"  Computing ground truth for {view_name} (target={target}) ...")
        if target == "discrete":
            gt_cache[view_name] = discrete_escape_grid(xs, ys)
        else:
            gt_cache[view_name] = smooth_escape_grid(xs, ys)
    gt = gt_cache[view_name]

    cmap = "gray_r" if target == "discrete" else "inferno"
    suffix = f"_{target}" if target == "discrete" else ""

    use_sigmoid = target == "discrete"
    preds = {}
    for name, model in models.items():
        print(f"  Predicting {name} on {view_name} ...")
        preds[name] = np.clip(
            predict_grid(model, xs, ys, apply_sigmoid=use_sigmoid), 0, 1)

    extent = [xs[0], xs[-1], ys[-1], ys[0]]

    ncols = 1 + n
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 7))
    if ncols == 1:
        axes = [axes]
    panels = [("Ground Truth", gt)] + [(name, preds[name]) for name in names]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, extent=extent, cmap=cmap,
                  vmin=0, vmax=1, aspect="auto")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Re(c)")
        ax.set_ylabel("Im(c)")
    target_label = target.capitalize()
    fig.suptitle(f"Mandelbrot Set ({target_label}) -- {view_name}", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{view_name}_comparison{suffix}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, n, figsize=(9 * n, 7))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        err = np.abs(preds[name] - gt)
        vmax = max(0.05, float(np.percentile(err, 99.5)))
        im = ax.imshow(err, extent=extent, cmap="hot",
                       vmin=0, vmax=vmax, aspect="auto")
        mean_err = float(err.mean())
        max_err = float(err.max())
        ax.set_title(
            f"|Error|  {name}  (mean={mean_err:.4f}, max={max_err:.4f})",
            fontsize=13,
        )
        ax.set_xlabel("Re(c)")
        ax.set_ylabel("Im(c)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Absolute Error ({target_label}) -- {view_name}", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{view_name}_error{suffix}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    return gt_cache


_COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#1abc9c"]


def render_loss_curves(model_keys, target="smooth"):
    suffix = f"_{target}" if target == "discrete" else ""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, color in zip(model_keys, _COLORS):
        stem = name
        if target == "discrete":
            stem += "_discrete"
        path = CKPT / f"{stem}_loss.json"
        if not path.exists():
            continue
        with open(path) as f:
            hist = json.load(f)
        ax.plot(range(1, len(hist) + 1), hist, label=name,
                color=color, linewidth=2)
    loss_label = "BCE Loss" if target == "discrete" else "MSE Loss"
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel(loss_label, fontsize=13)
    target_label = target.capitalize()
    ax.set_title(f"Training Loss ({target_label})", fontsize=15)
    ax.legend(fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"loss_curves{suffix}.png", dpi=150)
    plt.close(fig)


def _discover_model_keys(target="smooth"):
    """Find all checkpoint stems that have both .pt and _loss.json files."""
    suffix = "_discrete" if target == "discrete" else ""
    keys = []
    for pt in sorted(CKPT.glob("*.pt")):
        stem = pt.stem
        if target == "discrete":
            if not stem.endswith("_discrete"):
                continue
            stem = stem[: -len("_discrete")]
        else:
            if stem.endswith("_discrete"):
                continue
        keys.append(stem)
    return keys


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="smooth",
                        choices=["smooth", "discrete"])
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model keys to render (auto-discovers if omitted)")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    model_keys = args.models or _discover_model_keys(target=args.target)
    print(f"Loading models (target={args.target}): {model_keys}")
    models = {}
    for key in model_keys:
        try:
            models[key] = load_model(key, target=args.target)
        except FileNotFoundError:
            print(f"  [skip] No checkpoint for {key}")

    gt_cache = {}
    for view_name, view in VIEWS.items():
        print(f"\nRendering {view_name} ...")
        gt_cache = render_comparison(view_name, view, models, gt_cache,
                                     target=args.target)

    print("\nRendering loss curves ...")
    render_loss_curves(list(models.keys()), target=args.target)

    print(f"\nAll plots saved to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
