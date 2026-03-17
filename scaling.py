"""
Scaling law experiments for the Mandelbrot learning problem.

Sweeps data size (N), model size (P), and compute (C) for both
baseline MLP and Fourier MLP, following Kaplan et al. (2020).

Usage:
    python scaling.py                    # full sweep, both GPUs
    python scaling.py --axis data        # data scaling only
    python scaling.py --axis model       # model scaling only
    python scaling.py --plot-only        # just regenerate plots from saved results
"""

import argparse
import json
import time
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import (get_or_build_master, subsample_dataset,
                  smooth_escape_grid, compute_ylim)
from models import MLPRes, MLPFourierRes

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("output")

EVAL_RES = (960, 540)
EVAL_XLIM = (-2.4, 1.0)

DATA_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000]

MODEL_CONFIGS = [
    {"hidden_dim": 32,   "num_blocks": 2},
    {"hidden_dim": 64,   "num_blocks": 4},
    {"hidden_dim": 128,  "num_blocks": 8},
    {"hidden_dim": 256,  "num_blocks": 12},
    {"hidden_dim": 384,  "num_blocks": 16},
    {"hidden_dim": 512,  "num_blocks": 20},
    {"hidden_dim": 768,  "num_blocks": 20},
    {"hidden_dim": 1024, "num_blocks": 20},
]

EPOCHS = 100
BATCH_SIZE = 4096
LR = 3e-4


def get_eval_grid():
    """Return the fixed evaluation grid and its ground-truth values (cached)."""
    cache = Path("data/eval_grid.npz")
    ylim = compute_ylim(EVAL_XLIM, EVAL_RES)
    xs = np.linspace(EVAL_XLIM[0], EVAL_XLIM[1], EVAL_RES[0]).astype(np.float64)
    ys = np.linspace(ylim[0], ylim[1], EVAL_RES[1]).astype(np.float64)
    if cache.exists():
        gt = np.load(cache)["gt"]
    else:
        print("Computing eval grid ground truth (one-time) ...")
        gt = smooth_escape_grid(xs, ys)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, gt=gt, xs=xs, ys=ys)
    return xs, ys, gt


def make_model(model_type, hidden_dim, num_blocks, device):
    if model_type == "baseline":
        m = MLPRes(hidden_dim=hidden_dim, num_blocks=num_blocks, act="silu")
    else:
        m = MLPFourierRes(
            num_feats=hidden_dim, sigmas=(2.0, 6.0, 10.0),
            hidden_dim=hidden_dim, num_blocks=num_blocks, act="silu", seed=0,
        )
    return m.to(device)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def eval_on_grid(model, xs, ys, device, batch_rows=64):
    """Predict on the eval grid, return (H, W) float32 array."""
    W, H = len(xs), len(ys)
    out = np.empty((H, W), dtype=np.float32)
    xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
    for i in range(0, H, batch_rows):
        j = min(i + batch_rows, H)
        cy = torch.tensor(ys[i:j], dtype=torch.float32, device=device)
        gx = xs_t.unsqueeze(0).expand(j - i, -1)
        gy = cy.unsqueeze(1).expand(-1, W)
        coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        pred = model(coords).squeeze(-1).cpu().numpy()
        out[i:j] = pred.reshape(j - i, W)
    return out


def train_and_eval(config):
    """
    Train a single model and evaluate on the fixed grid.
    config dict keys: model_type, hidden_dim, num_blocks, n_data,
                      device, X, y, xs, ys, gt, epochs, batch_size, lr
    Returns dict with results.
    """
    model_type = config["model_type"]
    hidden_dim = config["hidden_dim"]
    num_blocks = config["num_blocks"]
    n_data = config["n_data"]
    device_str = config["device"]
    epochs = config.get("epochs", EPOCHS)
    batch_size = config.get("batch_size", BATCH_SIZE)
    lr = config.get("lr", LR)

    device = torch.device(device_str)
    model = make_model(model_type, hidden_dim, num_blocks, device)
    n_params = count_params(model)

    Xsub, ysub = subsample_dataset(config["X"], config["y"], n_data)
    Xt = torch.from_numpy(Xsub).to(device)
    yt = torch.from_numpy(ysub).unsqueeze(-1).to(device)
    ds = TensorDataset(Xt, yt)
    effective_bs = min(batch_size, n_data)
    dl = DataLoader(ds, batch_size=effective_bs, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    tag = f"{model_type}_h{hidden_dim}_b{num_blocks}_n{n_data}"
    t0 = time.time()
    train_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for xb, yb in dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
            count += xb.size(0)
        sched.step()
        train_losses.append(running / max(count, 1))

    wall_time = time.time() - t0

    model.eval()
    pred_grid = eval_on_grid(model, config["xs"], config["ys"], device)
    pred_grid = np.clip(pred_grid, 0, 1)
    test_mse = float(np.mean((pred_grid - config["gt"]) ** 2))

    flops = 6.0 * n_data * n_params * epochs

    result = {
        "tag": tag,
        "model_type": model_type,
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "n_params": n_params,
        "n_data": n_data,
        "epochs": epochs,
        "train_loss_final": train_losses[-1],
        "test_mse": test_mse,
        "flops": flops,
        "wall_time_s": wall_time,
    }
    print(f"  {tag:50s}  P={n_params:>10,}  test_mse={test_mse:.6f}  "
          f"train_loss={train_losses[-1]:.6f}  [{wall_time:.0f}s]")
    return result


def run_on_device(configs, device_str, master_X, master_y, xs, ys, gt):
    """Run a list of configs sequentially on one GPU."""
    results = []
    for c in configs:
        c["device"] = device_str
        c["X"] = master_X
        c["y"] = master_y
        c["xs"] = xs
        c["ys"] = ys
        c["gt"] = gt
        r = train_and_eval(c)
        results.append(r)
    return results


def run_worker_subprocess(configs_json_path, device_str, results_json_path):
    """Launch a subprocess that trains configs on one GPU and writes results."""
    cmd = [
        sys.executable, "-u", __file__,
        "--worker",
        "--configs-path", str(configs_json_path),
        "--device", device_str,
        "--results-path", str(results_json_path),
    ]
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    return proc


def worker_main(configs_path, device_str, results_path):
    """Entry point for a worker subprocess."""
    with open(configs_path) as f:
        configs = json.load(f)
    d = np.load("data/master_2M.npz")
    master_X, master_y = d["X"], d["y"]
    e = np.load("data/eval_grid.npz")
    xs = e["xs"].astype(np.float64)
    ys = e["ys"].astype(np.float64)
    gt = e["gt"]
    results = run_on_device(configs, device_str, master_X, master_y, xs, ys, gt)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Worker {device_str} done: {len(results)} results -> {results_path}")


def build_data_scaling_configs():
    """Data scaling: fix large model, vary N."""
    configs = []
    for n in DATA_SIZES:
        for mt in ["baseline", "fourier"]:
            configs.append({
                "model_type": mt,
                "hidden_dim": 512,
                "num_blocks": 20,
                "n_data": n,
            })
    return configs


def build_model_scaling_configs():
    """Model scaling: fix data at 1M, vary model size."""
    configs = []
    for mc in MODEL_CONFIGS:
        for mt in ["baseline", "fourier"]:
            configs.append({
                "model_type": mt,
                "n_data": 1_000_000,
                **mc,
            })
    return configs


def split_for_gpus(configs):
    """Interleave configs across two GPUs for balanced load."""
    gpu0, gpu1 = [], []
    for i, c in enumerate(configs):
        (gpu0 if i % 2 == 0 else gpu1).append(c)
    return gpu0, gpu1


def power_law(x, a, alpha):
    return a * np.power(x, -alpha)


def fit_power_law(x, y):
    """Fit y = a * x^(-alpha) in log-log space, return (a, alpha)."""
    try:
        log_x = np.log(np.array(x, dtype=np.float64))
        log_y = np.log(np.array(y, dtype=np.float64))
        mask = np.isfinite(log_x) & np.isfinite(log_y)
        log_x, log_y = log_x[mask], log_y[mask]
        if len(log_x) < 2:
            return None, None
        coeffs = np.polyfit(log_x, log_y, 1)
        alpha = -coeffs[0]
        a = np.exp(coeffs[1])
        return a, alpha
    except Exception:
        return None, None


def plot_data_scaling(results):
    fig, ax = plt.subplots(figsize=(10, 7))
    for mt, color, marker in [("baseline", "#e74c3c", "o"), ("fourier", "#2ecc71", "s")]:
        pts = sorted([r for r in results if r["model_type"] == mt], key=lambda r: r["n_data"])
        ns = [r["n_data"] for r in pts]
        losses = [r["test_mse"] for r in pts]
        ax.scatter(ns, losses, c=color, marker=marker, s=60, zorder=5, label=f"{mt}")
        a, alpha = fit_power_law(ns, losses)
        if a is not None:
            xs_fit = np.logspace(np.log10(min(ns)), np.log10(max(ns)), 100)
            ax.plot(xs_fit, power_law(xs_fit, a, alpha), color=color, ls="--",
                    lw=1.5, alpha=0.7, label=f"{mt} fit: $\\alpha$={alpha:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size N", fontsize=13)
    ax.set_ylabel("Test MSE", fontsize=13)
    ax.set_title("Data Scaling Law", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scaling_data.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / 'scaling_data.png'}")


def plot_model_scaling(results):
    fig, ax = plt.subplots(figsize=(10, 7))
    for mt, color, marker in [("baseline", "#e74c3c", "o"), ("fourier", "#2ecc71", "s")]:
        pts = sorted([r for r in results if r["model_type"] == mt], key=lambda r: r["n_params"])
        ps = [r["n_params"] for r in pts]
        losses = [r["test_mse"] for r in pts]
        ax.scatter(ps, losses, c=color, marker=marker, s=60, zorder=5, label=f"{mt}")
        a, alpha = fit_power_law(ps, losses)
        if a is not None:
            xs_fit = np.logspace(np.log10(min(ps)), np.log10(max(ps)), 100)
            ax.plot(xs_fit, power_law(xs_fit, a, alpha), color=color, ls="--",
                    lw=1.5, alpha=0.7, label=f"{mt} fit: $\\alpha$={alpha:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters P", fontsize=13)
    ax.set_ylabel("Test MSE", fontsize=13)
    ax.set_title("Model Scaling Law", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scaling_model.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / 'scaling_model.png'}")


def plot_compute_scaling(all_results):
    fig, ax = plt.subplots(figsize=(10, 7))
    for mt, color, marker in [("baseline", "#e74c3c", "o"), ("fourier", "#2ecc71", "s")]:
        pts = [r for r in all_results if r["model_type"] == mt]
        cs = [r["flops"] for r in pts]
        losses = [r["test_mse"] for r in pts]
        ax.scatter(cs, losses, c=color, marker=marker, s=40, alpha=0.6, label=f"{mt}")
        a, alpha = fit_power_law(cs, losses)
        if a is not None:
            xs_fit = np.logspace(np.log10(min(cs)), np.log10(max(cs)), 100)
            ax.plot(xs_fit, power_law(xs_fit, a, alpha), color=color, ls="--",
                    lw=1.5, alpha=0.7, label=f"{mt} fit: $\\alpha$={alpha:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute C (approx FLOPs)", fontsize=13)
    ax.set_ylabel("Test MSE", fontsize=13)
    ax.set_title("Compute Scaling Law", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scaling_compute.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / 'scaling_compute.png'}")


def plot_summary(data_results, model_results, all_results):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Data scaling
    ax = axes[0]
    for mt, color, marker in [("baseline", "#e74c3c", "o"), ("fourier", "#2ecc71", "s")]:
        pts = sorted([r for r in data_results if r["model_type"] == mt], key=lambda r: r["n_data"])
        ns = [r["n_data"] for r in pts]
        losses = [r["test_mse"] for r in pts]
        ax.scatter(ns, losses, c=color, marker=marker, s=50, zorder=5, label=mt)
        a, alpha = fit_power_law(ns, losses)
        if a is not None:
            xs_fit = np.logspace(np.log10(min(ns)), np.log10(max(ns)), 100)
            ax.plot(xs_fit, power_law(xs_fit, a, alpha), color=color, ls="--", lw=1.5, alpha=0.7)
            ax.text(0.05, 0.95 if mt == "baseline" else 0.88,
                    f"{mt} $\\alpha_N$={alpha:.2f}", transform=ax.transAxes,
                    fontsize=10, color=color, va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Dataset size N"); ax.set_ylabel("Test MSE")
    ax.set_title("Data Scaling"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")

    # Model scaling
    ax = axes[1]
    for mt, color, marker in [("baseline", "#e74c3c", "o"), ("fourier", "#2ecc71", "s")]:
        pts = sorted([r for r in model_results if r["model_type"] == mt], key=lambda r: r["n_params"])
        ps = [r["n_params"] for r in pts]
        losses = [r["test_mse"] for r in pts]
        ax.scatter(ps, losses, c=color, marker=marker, s=50, zorder=5, label=mt)
        a, alpha = fit_power_law(ps, losses)
        if a is not None:
            xs_fit = np.logspace(np.log10(min(ps)), np.log10(max(ps)), 100)
            ax.plot(xs_fit, power_law(xs_fit, a, alpha), color=color, ls="--", lw=1.5, alpha=0.7)
            ax.text(0.05, 0.95 if mt == "baseline" else 0.88,
                    f"{mt} $\\alpha_P$={alpha:.2f}", transform=ax.transAxes,
                    fontsize=10, color=color, va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Parameters P"); ax.set_ylabel("Test MSE")
    ax.set_title("Model Scaling"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")

    # Compute scaling
    ax = axes[2]
    for mt, color, marker in [("baseline", "#e74c3c", "o"), ("fourier", "#2ecc71", "s")]:
        pts = [r for r in all_results if r["model_type"] == mt]
        cs = [r["flops"] for r in pts]
        losses = [r["test_mse"] for r in pts]
        ax.scatter(cs, losses, c=color, marker=marker, s=40, alpha=0.6, label=mt)
        a, alpha = fit_power_law(cs, losses)
        if a is not None:
            xs_fit = np.logspace(np.log10(min(cs)), np.log10(max(cs)), 100)
            ax.plot(xs_fit, power_law(xs_fit, a, alpha), color=color, ls="--", lw=1.5, alpha=0.7)
            ax.text(0.05, 0.95 if mt == "baseline" else 0.88,
                    f"{mt} $\\alpha_C$={alpha:.2f}", transform=ax.transAxes,
                    fontsize=10, color=color, va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Compute C (FLOPs)"); ax.set_ylabel("Test MSE")
    ax.set_title("Compute Scaling"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Neural Scaling Laws for the Mandelbrot Set", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scaling_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / 'scaling_summary.png'}")


def generate_all_plots(results_path):
    with open(results_path) as f:
        all_results = json.load(f)
    data_results = [r for r in all_results if r.get("axis") == "data"]
    model_results = [r for r in all_results if r.get("axis") == "model"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating plots ...")
    if data_results:
        plot_data_scaling(data_results)
    if model_results:
        plot_model_scaling(model_results)
    if all_results:
        plot_compute_scaling(all_results)
    if data_results and model_results:
        plot_summary(data_results, model_results, all_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", default="all", choices=["all", "data", "model"])
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--sanity", action="store_true",
                        help="Run a tiny sanity-check sweep (2 configs, 5 epochs)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--configs-path", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--results-path", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker_main(args.configs_path, args.device, args.results_path)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "scaling_results.json"

    if args.plot_only:
        generate_all_plots(results_path)
        return

    # Prepare master dataset and eval grid
    print("Preparing master dataset ...")
    master_X, master_y, ylim = get_or_build_master()
    xs, ys, gt = get_eval_grid()
    master_path = "data/master_2M.npz"
    eval_path = "data/eval_grid.npz"

    all_results = []
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)

    if args.sanity:
        print("\n=== SANITY CHECK MODE ===")
        configs = [
            {"model_type": "fourier", "hidden_dim": 64, "num_blocks": 4,
             "n_data": 10_000, "epochs": 5},
            {"model_type": "baseline", "hidden_dim": 64, "num_blocks": 4,
             "n_data": 10_000, "epochs": 5},
        ]
        for c in configs:
            c["device"] = "cuda:0"
            c["X"] = master_X; c["y"] = master_y
            c["xs"] = xs; c["ys"] = ys; c["gt"] = gt
            r = train_and_eval(c)
            r["axis"] = "sanity"
            print(f"    -> test_mse={r['test_mse']:.6f}  wall={r['wall_time_s']:.1f}s")
        print("Sanity check passed.")
        return

    def run_axis(axis_name, configs):
        print(f"\n{'='*60}")
        print(f"  {axis_name} scaling: {len(configs)} runs")
        print(f"{'='*60}")

        gpu0_cfgs, gpu1_cfgs = split_for_gpus(configs)

        tmp = RESULTS_DIR / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        cfg0_path = tmp / f"{axis_name}_gpu0_configs.json"
        cfg1_path = tmp / f"{axis_name}_gpu1_configs.json"
        res0_path = tmp / f"{axis_name}_gpu0_results.json"
        res1_path = tmp / f"{axis_name}_gpu1_results.json"

        with open(cfg0_path, "w") as f:
            json.dump(gpu0_cfgs, f)
        with open(cfg1_path, "w") as f:
            json.dump(gpu1_cfgs, f)

        p0 = run_worker_subprocess(cfg0_path, "cuda:0", res0_path)
        p1 = run_worker_subprocess(cfg1_path, "cuda:1", res1_path)
        p0.wait()
        p1.wait()

        if p0.returncode != 0 or p1.returncode != 0:
            print(f"ERROR: worker returned non-zero exit code "
                  f"(gpu0={p0.returncode}, gpu1={p1.returncode})")
            sys.exit(1)

        with open(res0_path) as f:
            res0 = json.load(f)
        with open(res1_path) as f:
            res1 = json.load(f)

        results = res0 + res1
        for r in results:
            r["axis"] = axis_name
        return results

    if args.axis in ("all", "data"):
        data_configs = build_data_scaling_configs()
        data_results = run_axis("data", data_configs)
        all_results.extend(data_results)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved intermediate results ({len(all_results)} total)")

    if args.axis in ("all", "model"):
        model_configs = build_model_scaling_configs()
        model_results = run_axis("model", model_configs)
        all_results.extend(model_results)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved final results ({len(all_results)} total)")

    generate_all_plots(results_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
