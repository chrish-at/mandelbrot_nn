"""
Scaling law experiments for the Mandelbrot learning problem.

Sweeps data size (N), model size (P), and compute (C) for the
Fourier MLP (and optionally baseline), following Kaplan et al. (2020).

Supports both smooth (continuous escape-time) and discrete (binary
membership) targets.

Usage:
    python scaling.py                                    # default: smooth, fourier, 2 GPUs
    python scaling.py --axis model --num-gpus 8          # model scaling only, 8 GPUs
    python scaling.py --target discrete --num-gpus 8     # discrete sweep, 8 GPUs
    python scaling.py --run-all --num-gpus 8             # all pending experiments at once
    python scaling.py --sweep --hf --num-gpus 8                         # full HF sweep
    python scaling.py --sweep --hf --test-label _3070test --num-gpus 8  # with output label
    python scaling.py --plot-only --hf --test-label _3070test           # regenerate plots
    python scaling.py --run-targeted --num-gpus 8        # (legacy) param scaling at N=5M
    python scaling.py --run-isodata --num-gpus 8         # (legacy) isodata at N=100K,500K,1M
"""

import argparse
import json
import time
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import (get_or_build_master, subsample_dataset,
                  smooth_escape_grid, discrete_escape_grid, compute_ylim)
from models import MLPRes, MLPFourierRes

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("output")

EVAL_RES = (960, 540)
EVAL_XLIM = (-2.4, 1.0)

DEFAULT_SIGMAS = (2.0, 6.0, 10.0)
HF_SIGMAS = (2.0, 6.0, 10.0, 30.0, 100.0)

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

MAX_STEPS = 25_000
MASTER_N_TOTAL = 10_000_000

TARGETED_MODEL_CONFIGS = [
    {"hidden_dim": 32,   "num_blocks": 2},
    {"hidden_dim": 48,   "num_blocks": 3},
    {"hidden_dim": 64,   "num_blocks": 4},
    {"hidden_dim": 96,   "num_blocks": 6},
    {"hidden_dim": 128,  "num_blocks": 8},
    {"hidden_dim": 192,  "num_blocks": 10},
    {"hidden_dim": 256,  "num_blocks": 12},
    {"hidden_dim": 320,  "num_blocks": 14},
    {"hidden_dim": 384,  "num_blocks": 16},
    {"hidden_dim": 512,  "num_blocks": 20},
]
TARGETED_N_DATA = 5_000_000

ISODATA_MODEL_CONFIGS = [
    {"hidden_dim": 32,   "num_blocks": 2},
    {"hidden_dim": 48,   "num_blocks": 3},
    {"hidden_dim": 64,   "num_blocks": 4},
    {"hidden_dim": 96,   "num_blocks": 6},
    {"hidden_dim": 128,  "num_blocks": 8},
    {"hidden_dim": 192,  "num_blocks": 10},
    {"hidden_dim": 320,  "num_blocks": 14},
    {"hidden_dim": 512,  "num_blocks": 20},
]
ISODATA_HIDDEN_DIMS = {mc["hidden_dim"] for mc in ISODATA_MODEL_CONFIGS}
ISODATA_N_VALUES = [100_000, 500_000, 1_000_000]

SWEEP_MODEL_CONFIGS = [
    {"hidden_dim": 32,   "num_blocks": 2},
    {"hidden_dim": 48,   "num_blocks": 3},
    {"hidden_dim": 64,   "num_blocks": 4},
    {"hidden_dim": 96,   "num_blocks": 6},
    {"hidden_dim": 128,  "num_blocks": 8},
    {"hidden_dim": 192,  "num_blocks": 10},
    {"hidden_dim": 256,  "num_blocks": 12},
    {"hidden_dim": 320,  "num_blocks": 14},
    {"hidden_dim": 384,  "num_blocks": 16},
    {"hidden_dim": 512,  "num_blocks": 20},
]
SWEEP_N_VALUES = [100_000, 500_000, 1_000_000, 5_000_000]
SWEEP_HIDDEN_DIMS = {mc["hidden_dim"] for mc in SWEEP_MODEL_CONFIGS}

SCALING_CKPT_DIR = Path("checkpoints/scaling")


# ---------------------------------------------------------------------------
# Paths and caching
# ---------------------------------------------------------------------------

def _eval_grid_cache_path(target):
    if target == "discrete":
        return Path("data/eval_grid_discrete.npz")
    return Path("data/eval_grid.npz")


def _results_path(target, hf=False, test_label=""):
    hf_tag = "_hf" if hf else ""
    if target == "discrete":
        return RESULTS_DIR / f"scaling_results_discrete{hf_tag}{test_label}.json"
    return RESULTS_DIR / f"scaling_results{hf_tag}{test_label}.json"


def get_eval_grid(target="smooth"):
    """Return the fixed evaluation grid and its ground-truth values (cached)."""
    cache = _eval_grid_cache_path(target)
    ylim = compute_ylim(EVAL_XLIM, EVAL_RES)
    xs = np.linspace(EVAL_XLIM[0], EVAL_XLIM[1], EVAL_RES[0]).astype(np.float64)
    ys = np.linspace(ylim[0], ylim[1], EVAL_RES[1]).astype(np.float64)
    if cache.exists():
        d = np.load(cache)
        return d["xs"].astype(np.float64), d["ys"].astype(np.float64), d["gt"]
    print(f"Computing eval grid ground truth ({target}, one-time) ...")
    gt = discrete_escape_grid(xs, ys) if target == "discrete" else smooth_escape_grid(xs, ys)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, gt=gt, xs=xs, ys=ys)
    return xs, ys, gt


# ---------------------------------------------------------------------------
# Model construction and evaluation
# ---------------------------------------------------------------------------

def make_model(model_type, hidden_dim, num_blocks, device,
               sigmas=DEFAULT_SIGMAS):
    if model_type == "baseline":
        m = MLPRes(hidden_dim=hidden_dim, num_blocks=num_blocks, act="silu")
    else:
        m = MLPFourierRes(
            num_feats=hidden_dim, sigmas=sigmas,
            hidden_dim=hidden_dim, num_blocks=num_blocks, act="silu", seed=0,
        )
    return m.to(device)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def eval_on_grid(model, xs, ys, device, batch_rows=64, apply_sigmoid=False):
    W, H = len(xs), len(ys)
    out = np.empty((H, W), dtype=np.float32)
    xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
    for i in range(0, H, batch_rows):
        j = min(i + batch_rows, H)
        cy = torch.tensor(ys[i:j], dtype=torch.float32, device=device)
        gx = xs_t.unsqueeze(0).expand(j - i, -1)
        gy = cy.unsqueeze(1).expand(-1, W)
        coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        logits = model(coords).squeeze(-1)
        if apply_sigmoid:
            logits = torch.sigmoid(logits)
        out[i:j] = logits.cpu().numpy().reshape(j - i, W)
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _compute_epochs_for_steps(max_steps, n_data, batch_size):
    """Compute the number of epochs needed to reach max_steps gradient updates."""
    effective_bs = min(batch_size, n_data)
    steps_per_epoch = max(1, n_data // effective_bs)
    return max(1, max_steps // steps_per_epoch)


def _eval_on_held_out(model, X_test, y_test, device, loss_fn, batch_size=8192):
    """Compute loss on held-out test set via batched forward pass."""
    model.eval()
    Xt = torch.from_numpy(X_test).to(device)
    yt = torch.from_numpy(y_test).unsqueeze(-1).to(device)
    total_loss = 0.0
    total_mse = 0.0
    count = 0
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            xb = Xt[i:i + batch_size]
            yb = yt[i:i + batch_size]
            pred = model(xb)
            total_loss += loss_fn(pred, yb).item() * xb.size(0)
            pred_clamped = pred.clamp(0, 1)
            total_mse += ((pred_clamped - yb) ** 2).sum().item()
            count += xb.size(0)
    return total_loss / max(count, 1), total_mse / max(count, 1)


def train_and_eval(config):
    """Train a single model and evaluate on held-out test data."""
    model_type = config["model_type"]
    hidden_dim = config["hidden_dim"]
    num_blocks = config["num_blocks"]
    n_data = config["n_data"]
    device_str = config["device"]
    target = config.get("target", "smooth")
    batch_size = config.get("batch_size", BATCH_SIZE)
    lr = config.get("lr", LR)

    max_steps = config.get("max_steps", None)
    if max_steps is not None:
        epochs = _compute_epochs_for_steps(max_steps, n_data, batch_size)
    else:
        epochs = config.get("epochs", EPOCHS)

    sigmas = tuple(config.get("sigmas", DEFAULT_SIGMAS))
    device = torch.device(device_str)
    model = make_model(model_type, hidden_dim, num_blocks, device, sigmas=sigmas)
    n_params = count_params(model)

    Xsub, ysub = subsample_dataset(config["X"], config["y"], n_data)
    Xt = torch.from_numpy(Xsub).to(device)
    yt = torch.from_numpy(ysub).unsqueeze(-1).to(device)
    ds = TensorDataset(Xt, yt)
    effective_bs = min(batch_size, n_data)
    dl = DataLoader(ds, batch_size=effective_bs, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss() if target == "discrete" else nn.MSELoss()

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

    # Evaluate on held-out test set from same distribution
    X_test = config.get("X_test")
    y_test = config.get("y_test")
    test_mse = None
    test_bce = None

    if X_test is not None and y_test is not None:
        test_loss, test_mse = _eval_on_held_out(
            model, X_test, y_test, device, loss_fn)
        if target == "discrete":
            test_bce = test_loss
    elif "xs" in config and "ys" in config and "gt" in config:
        model.eval()
        pred_logits = eval_on_grid(model, config["xs"], config["ys"], device,
                                   apply_sigmoid=False)
        if target == "discrete":
            pred_probs = 1.0 / (1.0 + np.exp(-np.clip(pred_logits, -20, 20)))
            test_bce = float(F.binary_cross_entropy_with_logits(
                torch.tensor(pred_logits.flatten()),
                torch.tensor(config["gt"].flatten().astype(np.float32))).item())
        else:
            pred_probs = np.clip(pred_logits, 0, 1)
        test_mse = float(np.mean((pred_probs - config["gt"]) ** 2))

    # Save checkpoint
    SCALING_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = SCALING_CKPT_DIR / f"{tag}.pt"
    torch.save(model.state_dict(), ckpt_path)

    flops = 6.0 * n_data * n_params * epochs

    result = {
        "tag": tag,
        "model_type": model_type,
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "n_params": n_params,
        "n_data": n_data,
        "epochs": epochs,
        "target": target,
        "axis": config.get("axis", "unknown"),
        "train_loss_final": train_losses[-1],
        "test_mse": test_mse,
        "flops": flops,
        "wall_time_s": wall_time,
    }
    if test_bce is not None:
        result["test_bce"] = test_bce

    loss_str = f"test_bce={test_bce:.6f}" if test_bce is not None else f"test_mse={test_mse:.6f}"
    print(f"  {tag:50s}  P={n_params:>10,}  {loss_str}  "
          f"train_loss={train_losses[-1]:.6f}  [{wall_time:.0f}s]")
    return result


# ---------------------------------------------------------------------------
# Worker infrastructure
# ---------------------------------------------------------------------------

def _load_datasets_for_targets(targets, n_total=2_000_000, test_frac=0.1):
    """Load master datasets, split train/test, and optionally load eval grids."""
    from data import _master_smooth_path, _cache_path
    datasets = {}
    eval_grids = {}
    for target in targets:
        smooth_path = _master_smooth_path(n_total)
        master_path = _cache_path(smooth_path, target)
        d = np.load(master_path)
        X, y = d["X"], d["y"]
        split = int(len(X) * (1 - test_frac))
        datasets[target] = (X[:split], y[:split], X[split:], y[split:])
        cache = _eval_grid_cache_path(target)
        if cache.exists():
            e = np.load(str(cache))
            eval_grids[target] = (
                e["xs"].astype(np.float64),
                e["ys"].astype(np.float64),
                e["gt"],
            )
    return datasets, eval_grids


def run_on_device(configs, device_str, datasets, eval_grids=None):
    """Run a list of configs sequentially on one GPU."""
    results = []
    for c in configs:
        target = c.get("target", "smooth")
        c["device"] = device_str
        X_train, y_train, X_test, y_test = datasets[target]
        c["X"], c["y"] = X_train, y_train
        c["X_test"], c["y_test"] = X_test, y_test
        if eval_grids and target in eval_grids:
            c["xs"], c["ys"], c["gt"] = eval_grids[target]
        results.append(train_and_eval(c))
    return results


def run_worker_subprocess(configs_json_path, device_str, results_json_path):
    cmd = [
        sys.executable, "-u", __file__,
        "--worker",
        "--configs-path", str(configs_json_path),
        "--device", device_str,
        "--results-path", str(results_json_path),
    ]
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def worker_main(configs_path, device_str, results_path):
    with open(configs_path) as f:
        configs = json.load(f)
    targets = set(c.get("target", "smooth") for c in configs)
    n_totals = {c.get("master_n_total", 2_000_000) for c in configs}
    n_total = max(n_totals)
    datasets, eval_grids = _load_datasets_for_targets(targets, n_total=n_total)
    results = run_on_device(configs, device_str, datasets, eval_grids=eval_grids)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Worker {device_str} done: {len(results)} results -> {results_path}")


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def build_data_scaling_configs(model_types=("fourier",), target="smooth",
                               sigmas=DEFAULT_SIGMAS):
    configs = []
    for n in DATA_SIZES:
        for mt in model_types:
            configs.append({
                "model_type": mt, "hidden_dim": 512, "num_blocks": 20,
                "n_data": n, "target": target, "axis": "data",
                "sigmas": list(sigmas),
            })
    return configs


def build_model_scaling_configs(model_types=("fourier",), target="smooth",
                                sigmas=DEFAULT_SIGMAS):
    configs = []
    for mc in MODEL_CONFIGS:
        for mt in model_types:
            configs.append({
                "model_type": mt, "n_data": 1_000_000,
                "target": target, "axis": "model",
                "sigmas": list(sigmas), **mc,
            })
    return configs


def build_targeted_configs(target="smooth", sigmas=DEFAULT_SIGMAS):
    """Parameter scaling at N=5M with fixed gradient-step budget."""
    configs = []
    for mc in TARGETED_MODEL_CONFIGS:
        configs.append({
            "model_type": "fourier",
            "n_data": TARGETED_N_DATA,
            "target": target,
            "axis": "targeted",
            "max_steps": MAX_STEPS,
            "master_n_total": MASTER_N_TOTAL,
            "sigmas": list(sigmas),
            **mc,
        })
    return configs


def build_isodata_configs(target="smooth", sigmas=DEFAULT_SIGMAS):
    """Parameter scaling at multiple N values with the 8 evenly-spaced model configs."""
    configs = []
    for n_data in ISODATA_N_VALUES:
        for mc in ISODATA_MODEL_CONFIGS:
            configs.append({
                "model_type": "fourier",
                "n_data": n_data,
                "target": target,
                "axis": "isodata",
                "max_steps": MAX_STEPS,
                "master_n_total": MASTER_N_TOTAL,
                "sigmas": list(sigmas),
                **mc,
            })
    return configs


def build_sweep_configs(target="smooth", sigmas=DEFAULT_SIGMAS):
    """Unified parameter scaling: 10 model sizes x 4 N values."""
    configs = []
    for n_data in SWEEP_N_VALUES:
        for mc in SWEEP_MODEL_CONFIGS:
            configs.append({
                "model_type": "fourier",
                "n_data": n_data,
                "target": target,
                "axis": "sweep",
                "max_steps": MAX_STEPS,
                "master_n_total": MASTER_N_TOTAL,
                "sigmas": list(sigmas),
                **mc,
            })
    return configs


# ---------------------------------------------------------------------------
# GPU scheduling
# ---------------------------------------------------------------------------

def _estimate_runtime(config):
    """Rough runtime estimate (seconds) for load balancing."""
    n_data = config["n_data"]
    hidden = config["hidden_dim"]
    blocks = config["num_blocks"]
    param_factor = hidden * hidden * blocks
    return n_data * param_factor / 1e11


def balance_across_gpus(configs, num_gpus):
    """LPT heuristic: assign heaviest jobs first to lightest GPU."""
    ranked = sorted(enumerate(configs), key=lambda t: _estimate_runtime(t[1]), reverse=True)
    buckets = [[] for _ in range(num_gpus)]
    loads = [0.0] * num_gpus
    for idx, cfg in ranked:
        lightest = min(range(num_gpus), key=lambda g: loads[g])
        buckets[lightest].append(cfg)
        loads[lightest] += _estimate_runtime(cfg)
    return buckets


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def launch_workers(configs, num_gpus, run_tag="combined"):
    """Distribute configs across GPUs and run worker subprocesses."""
    gpu_buckets = balance_across_gpus(configs, num_gpus)

    tmp = RESULTS_DIR / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    procs = []
    res_paths = []
    for gpu_id, bucket in enumerate(gpu_buckets):
        if not bucket:
            continue
        cfg_path = tmp / f"{run_tag}_gpu{gpu_id}_configs.json"
        res_path = tmp / f"{run_tag}_gpu{gpu_id}_results.json"
        with open(cfg_path, "w") as f:
            json.dump(bucket, f)
        procs.append(run_worker_subprocess(cfg_path, f"cuda:{gpu_id}", res_path))
        res_paths.append(res_path)

    for p in procs:
        p.wait()

    failed = [p for p in procs if p.returncode != 0]
    if failed:
        print(f"ERROR: {len(failed)} worker(s) returned non-zero exit code")
        sys.exit(1)

    results = []
    for rp in res_paths:
        with open(rp) as f:
            results.extend(json.load(f))
    return results


def run_axis(axis_name, configs, num_gpus):
    print(f"\n{'='*60}")
    print(f"  {axis_name} scaling: {len(configs)} runs on {num_gpus} GPUs")
    print(f"{'='*60}")
    return launch_workers(configs, num_gpus, run_tag=axis_name)


# ---------------------------------------------------------------------------
# Plotting (addition_nn style)
# ---------------------------------------------------------------------------

def _plot_scaling(x, y, xlabel, ylabel, title, filename):
    """Single-series log-log plot with power-law fit, matching addition_nn style."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, "o-", linewidth=2, markersize=6, color="#2176AE")

    valid = (y > 0) & (x > 0)
    if valid.sum() >= 2:
        log_x = np.log10(x[valid])
        log_y = np.log10(y[valid])
        coeffs = np.polyfit(log_x, log_y, 1)
        fit_x = np.linspace(log_x.min(), log_x.max(), 100)
        fit_y = np.polyval(coeffs, fit_x)
        ax.plot(10**fit_x, 10**fit_y, "--", color="#B66D0D", linewidth=1.5,
                label=f"fit: slope={coeffs[0]:.2f}")
        ax.legend(fontsize=11)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"  Saved {filename}")


def _loss_key(target):
    return "test_mse"


def _loss_label(target):
    return "Test MSE"


def _get_loss(r, target):
    return r["test_mse"]


def plot_data_scaling(results, target="smooth", hf=False, test_label=""):
    """Data scaling: loss vs N for the big model (h=512).

    Combines original data-scaling runs and the targeted/sweep N=5M run.
    Uses test_mse for consistency (available for all runs).
    """
    hf_tag = "_hf" if hf else ""
    suffix = ("_discrete" if target == "discrete" else "") + hf_tag + test_label
    pts = sorted(
        [r for r in results if r["model_type"] == "fourier"
         and r["hidden_dim"] == 512 and r["num_blocks"] == 20],
        key=lambda r: r["n_data"],
    )
    seen = set()
    deduped = []
    for r in pts:
        if r["n_data"] not in seen:
            seen.add(r["n_data"])
            deduped.append(r)
    pts = deduped

    if not pts:
        print("  [skip] No fourier data-scaling results.")
        return
    _plot_scaling(
        [r["n_data"] for r in pts],
        [r["test_mse"] for r in pts],
        xlabel="Dataset Size (N)", ylabel="Test MSE",
        title=f"Data Scaling: Mandelbrot ({target.capitalize()})",
        filename=OUTPUT_DIR / f"scaling_data{suffix}.png",
    )


def plot_model_scaling(results, target="smooth", hf=False, test_label=""):
    """Parameter scaling: family of L(P) curves at different N values."""
    hf_tag = "_hf" if hf else ""
    suffix = ("_discrete" if target == "discrete" else "") + hf_tag + test_label
    all_hidden_dims = SWEEP_HIDDEN_DIMS | ISODATA_HIDDEN_DIMS
    fourier = [r for r in results if r["model_type"] == "fourier"
               and r.get("hidden_dim") in all_hidden_dims]
    if not fourier:
        print("  [skip] No fourier model-scaling results.")
        return

    from collections import defaultdict
    by_n = defaultdict(list)
    for r in fourier:
        by_n[r["n_data"]].append(r)

    sorted_ns = sorted(by_n.keys())
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ltblue_purple",
                                              ["#89CFF0", "#6A0DAD"])
    colors = [cmap(i / max(1, len(sorted_ns) - 1)) for i in range(len(sorted_ns))]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ylabel = _loss_label(target)

    for idx, n_data in enumerate(sorted_ns):
        group = sorted(by_n[n_data], key=lambda r: r["n_params"])
        x = [r["n_params"] for r in group]
        y = [_get_loss(r, target) for r in group]
        if n_data >= 1_000_000:
            label = f"N={n_data / 1_000_000:.0f}M"
        else:
            label = f"N={n_data / 1_000:.0f}K"
        ax.plot(x, y, "o-", linewidth=1.8, markersize=5, color=colors[idx],
                label=label, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters (P)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f"Parameter Scaling: Mandelbrot ({target.capitalize()})", fontsize=14)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    filename = OUTPUT_DIR / f"scaling_params{suffix}.png"
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"  Saved {filename}")


def _pareto_frontier(x, y):
    """Extract Pareto frontier: for increasing x, the minimum y seen so far."""
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]
    front_x, front_y = [], []
    best = np.inf
    for xi, yi in zip(x_s, y_s):
        if yi < best:
            best = yi
            front_x.append(xi)
            front_y.append(yi)
    return np.array(front_x), np.array(front_y)


def plot_compute_scaling(all_results, target="smooth", hf=False, test_label=""):
    """Compute scaling: scatter all runs, highlight Pareto frontier."""
    hf_tag = "_hf" if hf else ""
    suffix = ("_discrete" if target == "discrete" else "") + hf_tag + test_label
    pts = [r for r in all_results if r["model_type"] == "fourier"]
    if not pts:
        print("  [skip] No fourier results for compute scaling.")
        return

    ylabel = _loss_label(target)
    fig, ax = plt.subplots(figsize=(7, 5))

    x_all = np.array([r["flops"] for r in pts], dtype=np.float64)
    y_all = np.array([_get_loss(r, target) for r in pts], dtype=np.float64)

    axes = [r.get("axis", "unknown") for r in pts]
    colors = {"data": "#7FBFFF", "model": "#FFB07F", "targeted": "#7FDF7F",
              "isodata": "#7FDF7F", "sweep": "#7FDF7F", "unknown": "#CCCCCC"}
    for ax_type in sorted(set(axes)):
        mask = np.array([a == ax_type for a in axes])
        ax.scatter(x_all[mask], y_all[mask], s=30, alpha=0.6,
                   color=colors.get(ax_type, "#CCCCCC"), label=ax_type, zorder=2)

    front_x, front_y = _pareto_frontier(x_all, y_all)
    ax.plot(front_x, front_y, "o-", color="#2176AE", linewidth=2, markersize=5,
            zorder=3, label="frontier")

    valid = (front_y > 0) & (front_x > 0)
    if valid.sum() >= 2:
        log_x = np.log10(front_x[valid])
        log_y = np.log10(front_y[valid])
        coeffs = np.polyfit(log_x, log_y, 1)
        fit_x = np.linspace(log_x.min(), log_x.max(), 100)
        fit_y = np.polyval(coeffs, fit_x)
        ax.plot(10**fit_x, 10**fit_y, "--", color="#B66D0D", linewidth=1.5,
                label=f"frontier fit: slope={coeffs[0]:.2f}", zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute (FLOPs, approx.)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f"Compute Scaling: Mandelbrot ({target.capitalize()})", fontsize=14)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    filename = OUTPUT_DIR / f"scaling_compute{suffix}.png"
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"  Saved {filename}")


def generate_all_plots(target="smooth", hf=False, test_label=""):
    rpath = _results_path(target, hf=hf, test_label=test_label)
    if not rpath.exists():
        print(f"No results file at {rpath}")
        return
    with open(rpath) as f:
        all_results = json.load(f)

    all_fourier = [r for r in all_results if r["model_type"] == "fourier"]
    data_and_targeted = [r for r in all_fourier
                         if r.get("axis") in ("data", "targeted", "sweep")]
    model_results = [r for r in all_fourier
                     if r.get("axis") in ("model", "targeted", "isodata", "sweep")]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating plots (target={target}, hf={hf}, test_label={test_label!r}) ...")
    if data_and_targeted:
        plot_data_scaling(data_and_targeted, target=target, hf=hf,
                          test_label=test_label)
    if model_results:
        plot_model_scaling(model_results, target=target, hf=hf,
                           test_label=test_label)
    if all_fourier:
        plot_compute_scaling(all_fourier, target=target, hf=hf,
                             test_label=test_label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _ensure_datasets(*targets, n_total=2_000_000):
    """Pre-build and cache master datasets and eval grids."""
    for t in targets:
        print(f"Preparing {t} master dataset ({n_total/1e6:.0f}M) ...")
        get_or_build_master(n_total=n_total, target=t)
        get_eval_grid(target=t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", default="all", choices=["all", "data", "model"])
    parser.add_argument("--target", default="smooth", choices=["smooth", "discrete"])
    parser.add_argument("--model-type", default="fourier",
                        choices=["fourier", "baseline", "both"])
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all pending experiments (smooth model + discrete all)")
    parser.add_argument("--run-targeted", action="store_true",
                        help="Run targeted param scaling at N=5M for both targets")
    parser.add_argument("--run-isodata", action="store_true",
                        help="Run isodata param scaling at N=100K,500K,1M for both targets")
    parser.add_argument("--hf", action="store_true",
                        help="Use high-frequency Fourier sigmas (2,6,10,30,100)")
    parser.add_argument("--sweep", action="store_true",
                        help="Unified param sweep: 10 models x 4 N values x 2 targets")
    parser.add_argument("--test-label", type=str, default="",
                        help="Suffix for output filenames (e.g. _3070test)")
    parser.add_argument("--sanity", action="store_true")
    # Hidden worker args
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

    if args.plot_only:
        generate_all_plots(target=args.target, hf=args.hf,
                           test_label=args.test_label)
        return

    sigmas = HF_SIGMAS if args.hf else DEFAULT_SIGMAS

    model_types = {
        "fourier": ("fourier",),
        "baseline": ("baseline",),
        "both": ("baseline", "fourier"),
    }[args.model_type]

    # ------------------------------------------------------------------
    # --sweep: unified param sweep (10 models x 4 N x 2 targets)
    # ------------------------------------------------------------------
    if args.sweep:
        _ensure_datasets("smooth", "discrete", n_total=MASTER_N_TOTAL)

        all_configs = []
        all_configs.extend(build_sweep_configs(target="smooth", sigmas=sigmas))
        all_configs.extend(build_sweep_configs(target="discrete", sigmas=sigmas))

        hf_label = " (HF)" if args.hf else ""
        print(f"\n{'='*60}")
        print(f"  Sweep{hf_label}: {len(all_configs)} configs on {args.num_gpus} GPUs")
        print(f"  N values: {SWEEP_N_VALUES}, max_steps={MAX_STEPS:,}")
        print(f"{'='*60}")

        results = launch_workers(all_configs, args.num_gpus, run_tag="sweep")

        for target in ("smooth", "discrete"):
            rpath = _results_path(target, hf=args.hf,
                                  test_label=args.test_label)
            existing = []
            if rpath.exists():
                with open(rpath) as f:
                    existing = json.load(f)
            new = [r for r in results if r.get("target") == target]
            existing_tags = {r["tag"] for r in existing}
            deduped = [r for r in new if r["tag"] not in existing_tags]
            merged = existing + deduped
            with open(rpath, "w") as f:
                json.dump(merged, f, indent=2)
            print(f"Saved {len(deduped)} new results to {rpath} "
                  f"({len(merged)} total)")

        generate_all_plots(target="smooth", hf=args.hf,
                           test_label=args.test_label)
        generate_all_plots(target="discrete", hf=args.hf,
                           test_label=args.test_label)
        print("\nDone.")
        return

    # ------------------------------------------------------------------
    # --run-targeted: param scaling at N=5M with fixed gradient steps
    # ------------------------------------------------------------------
    if args.run_targeted:
        _ensure_datasets("smooth", "discrete", n_total=MASTER_N_TOTAL)

        all_configs = []
        all_configs.extend(build_targeted_configs(target="smooth", sigmas=sigmas))
        all_configs.extend(build_targeted_configs(target="discrete", sigmas=sigmas))

        hf_label = " (HF)" if args.hf else ""
        print(f"\n{'='*60}")
        print(f"  Targeted run{hf_label}: {len(all_configs)} configs on {args.num_gpus} GPUs")
        print(f"  N={TARGETED_N_DATA:,}, max_steps={MAX_STEPS:,}")
        print(f"{'='*60}")

        results = launch_workers(all_configs, args.num_gpus, run_tag="targeted")

        for target in ("smooth", "discrete"):
            rpath = _results_path(target, hf=args.hf)
            existing = []
            if rpath.exists():
                with open(rpath) as f:
                    existing = json.load(f)
            new = [r for r in results if r.get("target") == target]
            existing_tags = {r["tag"] for r in existing}
            deduped = [r for r in new if r["tag"] not in existing_tags]
            merged = existing + deduped
            with open(rpath, "w") as f:
                json.dump(merged, f, indent=2)
            print(f"Saved {len(deduped)} new results to {rpath} "
                  f"({len(merged)} total)")

        generate_all_plots(target="smooth", hf=args.hf)
        generate_all_plots(target="discrete", hf=args.hf)
        print("\nDone.")
        return

    # ------------------------------------------------------------------
    # --run-isodata: param scaling at N=100K,500K,1M (8 model configs)
    # ------------------------------------------------------------------
    if args.run_isodata:
        _ensure_datasets("smooth", "discrete", n_total=MASTER_N_TOTAL)

        all_configs = []
        all_configs.extend(build_isodata_configs(target="smooth", sigmas=sigmas))
        all_configs.extend(build_isodata_configs(target="discrete", sigmas=sigmas))

        hf_label = " (HF)" if args.hf else ""
        print(f"\n{'='*60}")
        print(f"  Isodata run{hf_label}: {len(all_configs)} configs on {args.num_gpus} GPUs")
        print(f"  N values: {ISODATA_N_VALUES}, max_steps={MAX_STEPS:,}")
        print(f"{'='*60}")

        results = launch_workers(all_configs, args.num_gpus, run_tag="isodata")

        for target in ("smooth", "discrete"):
            rpath = _results_path(target, hf=args.hf)
            existing = []
            if rpath.exists():
                with open(rpath) as f:
                    existing = json.load(f)
            new = [r for r in results if r.get("target") == target]
            existing_tags = {r["tag"] for r in existing}
            deduped = [r for r in new if r["tag"] not in existing_tags]
            merged = existing + deduped
            with open(rpath, "w") as f:
                json.dump(merged, f, indent=2)
            print(f"Saved {len(deduped)} new results to {rpath} "
                  f"({len(merged)} total)")

        generate_all_plots(target="smooth", hf=args.hf)
        generate_all_plots(target="discrete", hf=args.hf)
        print("\nDone.")
        return

    # ------------------------------------------------------------------
    # --run-all: combined mode — smooth model + discrete data & model
    # ------------------------------------------------------------------
    if args.run_all:
        _ensure_datasets("smooth", "discrete")

        all_configs = []
        all_configs.extend(build_model_scaling_configs(
            model_types=model_types, target="smooth"))
        all_configs.extend(build_data_scaling_configs(
            model_types=model_types, target="discrete"))
        all_configs.extend(build_model_scaling_configs(
            model_types=model_types, target="discrete"))

        print(f"\n{'='*60}")
        print(f"  Combined run: {len(all_configs)} configs on {args.num_gpus} GPUs")
        print(f"{'='*60}")

        results = launch_workers(all_configs, args.num_gpus, run_tag="combined")

        for target in ("smooth", "discrete"):
            rpath = _results_path(target)
            existing = []
            if rpath.exists():
                with open(rpath) as f:
                    existing = json.load(f)
            new = [r for r in results if r.get("target") == target]
            merged = existing + new
            with open(rpath, "w") as f:
                json.dump(merged, f, indent=2)
            print(f"Saved {len(new)} new results to {rpath} "
                  f"({len(merged)} total)")

        generate_all_plots(target="smooth")
        generate_all_plots(target="discrete")
        print("\nDone.")
        return

    # ------------------------------------------------------------------
    # Standard single-target mode
    # ------------------------------------------------------------------
    _ensure_datasets(args.target)
    results_path = _results_path(args.target)

    all_results = []
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)

    if args.sanity:
        print("\n=== SANITY CHECK MODE ===")
        cfg = {
            "model_type": "fourier", "hidden_dim": 64, "num_blocks": 4,
            "n_data": 10_000, "epochs": 5, "target": args.target, "axis": "sanity",
        }
        datasets, eval_grids = _load_datasets_for_targets({args.target})
        cfg["device"] = "cuda:0"
        cfg["X"], cfg["y"] = datasets[args.target]
        cfg["xs"], cfg["ys"], cfg["gt"] = eval_grids[args.target]
        r = train_and_eval(cfg)
        print(f"    -> test_mse={r['test_mse']:.6f}  wall={r['wall_time_s']:.1f}s")
        print("Sanity check passed.")
        return

    if args.axis in ("all", "data"):
        data_configs = build_data_scaling_configs(
            model_types=model_types, target=args.target)
        data_results = run_axis("data", data_configs, args.num_gpus)
        all_results.extend(data_results)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved intermediate results ({len(all_results)} total)")

    if args.axis in ("all", "model"):
        model_configs = build_model_scaling_configs(
            model_types=model_types, target=args.target)
        model_results = run_axis("model", model_configs, args.num_gpus)
        all_results.extend(model_results)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved final results ({len(all_results)} total)")

    generate_all_plots(target=args.target)
    print("\nDone.")


if __name__ == "__main__":
    main()
