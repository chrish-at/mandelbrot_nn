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
from train import _NAMED_EXPERIMENTS, make_model

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


def load_model(key, target="smooth", ckpt_suffix=""):
    m = make_model(key, DEVICE)
    stem = key
    if target == "discrete":
        stem += "_discrete"
    path = CKPT / f"{stem}{ckpt_suffix}.pt"
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


def render_comparison(view_name, view, models, gt_cache, target="smooth",
                      display_names=None):
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
    _dn = display_names or {}
    panels = [("Ground Truth", gt)] + [(_dn.get(name, name), preds[name]) for name in names]
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

    errors = {name: np.abs(preds[name] - gt) for name in names}
    shared_vmax = max(0.05, float(max(
        np.percentile(err, 99.5) for err in errors.values())))

    fig, axes = plt.subplots(1, n, figsize=(9 * n, 7))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        err = errors[name]
        im = ax.imshow(err, extent=extent, cmap="hot",
                       vmin=0, vmax=shared_vmax, aspect="auto")
        mean_err = float(err.mean())
        max_err = float(err.max())
        disp = _dn.get(name, name)
        ax.set_title(
            f"|Error|  {disp}  (mean={mean_err:.4f}, max={max_err:.4f})",
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


def render_loss_curves(model_keys, models=None, target="smooth",
                       display_names=None, ckpt_suffix="",
                       output_suffix=""):
    suffix = f"_{target}" if target == "discrete" else ""
    suffix += output_suffix
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, color in zip(model_keys, _COLORS):
        stem = name
        if target == "discrete":
            stem += "_discrete"
        path = CKPT / f"{stem}{ckpt_suffix}_loss.json"
        if not path.exists():
            continue
        with open(path) as f:
            raw = json.load(f)
        train_hist = raw["train"] if isinstance(raw, dict) else raw
        test_hist = raw.get("test") if isinstance(raw, dict) else None
        label = display_names.get(name, name) if display_names else name
        epochs = range(1, len(train_hist) + 1)
        ax.plot(epochs, train_hist, color=color, linewidth=2,
                label=f"{label} train")
        if test_hist:
            ax.plot(range(1, len(test_hist) + 1), test_hist, color=color,
                    linewidth=1.5, linestyle="--", label=f"{label} test")

    loss_label = "BCE Loss" if target == "discrete" else "MSE Loss"
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel(loss_label, fontsize=13)
    target_label = target.capitalize()
    ax.set_title(f"Training Loss ({target_label})", fontsize=15)
    ax.legend(fontsize=10)
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


DISPLAY_NAMES = {
    "swiglu_untied": "swiglu",
    "fourier_swiglu_untied": "fourier_swiglu",
    "fourier_hf": "fourier",
    "fourier_swiglu_untied_hf": "fourier_swiglu",
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="smooth",
                        choices=["smooth", "discrete"])
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model keys for comparison images (auto-discovers if omitted)")
    parser.add_argument("--loss_models", nargs="+", default=None,
                        help="Model keys for loss curves (defaults to --models)")
    parser.add_argument("--ckpt_suffix", default="",
                        help="Suffix appended to checkpoint stems (e.g. '_strat')")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    comp_keys = args.models or _discover_model_keys(target=args.target)
    loss_keys = args.loss_models or comp_keys

    all_keys = list(dict.fromkeys(comp_keys + loss_keys))
    print(f"Loading models (target={args.target}, suffix={args.ckpt_suffix!r}): {all_keys}")
    models = {}
    for key in all_keys:
        try:
            models[key] = load_model(key, target=args.target,
                                     ckpt_suffix=args.ckpt_suffix)
        except FileNotFoundError:
            print(f"  [skip] No checkpoint for {key}")

    comp_models = {k: models[k] for k in comp_keys if k in models}
    gt_cache = {}
    for view_name, view in VIEWS.items():
        print(f"\nRendering {view_name} ...")
        gt_cache = render_comparison(view_name, view, comp_models, gt_cache,
                                     target=args.target,
                                     display_names=DISPLAY_NAMES)

    print("\nRendering loss curves ...")
    render_loss_curves(loss_keys, models, target=args.target,
                       display_names=DISPLAY_NAMES,
                       ckpt_suffix=args.ckpt_suffix)

    print(f"\nAll plots saved to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
