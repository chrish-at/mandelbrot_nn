"""
Parameter scaling experiment with stratified train/test sampling.

Trains Fourier MLPs at 8 model sizes x 4 data sizes (N=100K..5M),
evaluates on a held-out stratified test split, and plots L(P) curves.

Usage:
    python scaling_stratified.py --num-gpus 8
    python scaling_stratified.py --plot-only
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from data import build_stratified_dataset, subsample_dataset
from models import MLPFourierRes

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("output")
CACHE_PATH = Path("data/master_stratified_5M.npz")

MODEL_CONFIGS = [
    {"hidden_dim": 32,  "num_blocks": 2},
    {"hidden_dim": 48,  "num_blocks": 3},
    {"hidden_dim": 64,  "num_blocks": 4},
    {"hidden_dim": 96,  "num_blocks": 6},
    {"hidden_dim": 128, "num_blocks": 8},
    {"hidden_dim": 192, "num_blocks": 10},
    {"hidden_dim": 320, "num_blocks": 14},
    {"hidden_dim": 512, "num_blocks": 20},
]

N_DATA_VALUES = [100_000, 500_000, 1_000_000, 5_000_000]

MASTER_N_TOTAL = 5_000_000
MAX_STEPS = 25_000
BATCH_SIZE = 4096
LR = 3e-4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def get_or_build_master():
    if CACHE_PATH.exists():
        print(f"Loading cached master stratified dataset from {CACHE_PATH}")
        d = np.load(CACHE_PATH)
        return d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    print(f"Building master stratified dataset (N={MASTER_N_TOTAL:,}) ...")
    X_train, y_train, X_test, y_test, _ = build_stratified_dataset(
        n_total=MASTER_N_TOTAL, seed=0,
    )
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test)
    print(f"  Saved to {CACHE_PATH}  "
          f"(train={len(X_train):,}, test={len(X_test):,})")
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def _compute_epochs(max_steps, n_data, batch_size):
    effective_bs = min(batch_size, n_data)
    steps_per_epoch = max(1, n_data // effective_bs)
    return max(1, max_steps // steps_per_epoch)


def make_model(hidden_dim, num_blocks, device):
    m = MLPFourierRes(
        num_feats=hidden_dim, sigmas=(2.0, 6.0, 10.0),
        hidden_dim=hidden_dim, num_blocks=num_blocks, act="silu", seed=0,
    )
    return m.to(device)


def train_and_eval(config):
    hidden_dim = config["hidden_dim"]
    num_blocks = config["num_blocks"]
    n_data = config["n_data"]
    device = torch.device(config["device"])

    model = make_model(hidden_dim, num_blocks, device)
    n_params = sum(p.numel() for p in model.parameters())

    X_train, y_train = subsample_dataset(
        config["X_train"], config["y_train"], n_data)
    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).unsqueeze(-1).to(device)
    effective_bs = min(BATCH_SIZE, n_data)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=effective_bs,
                    shuffle=True, drop_last=True)

    Xt_test = torch.from_numpy(config["X_test"]).to(device)
    yt_test = torch.from_numpy(config["y_test"]).unsqueeze(-1).to(device)

    epochs = _compute_epochs(MAX_STEPS, n_data, BATCH_SIZE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    tag = f"fourier_h{hidden_dim}_b{num_blocks}_n{n_data}"
    t0 = time.time()
    train_losses = []

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
        train_losses.append(running / max(count, 1))

    wall_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        test_loss = loss_fn(model(Xt_test), yt_test).item()

    result = {
        "tag": tag,
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "n_params": n_params,
        "n_data": n_data,
        "epochs": epochs,
        "train_loss": train_losses[-1],
        "test_loss": test_loss,
        "wall_time_s": wall_time,
    }
    print(f"  {tag:45s}  P={n_params:>10,}  "
          f"train={train_losses[-1]:.6f}  test={test_loss:.6f}  "
          f"[{wall_time:.0f}s]")
    return result


# ---------------------------------------------------------------------------
# Worker subprocess
# ---------------------------------------------------------------------------

def worker_main(configs_path, device_str, results_path):
    with open(configs_path) as f:
        configs = json.load(f)

    X_train, y_train, X_test, y_test = get_or_build_master()

    results = []
    for c in configs:
        c["device"] = device_str
        c["X_train"] = X_train
        c["y_train"] = y_train
        c["X_test"] = X_test
        c["y_test"] = y_test
        results.append(train_and_eval(c))

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Worker {device_str} done: {len(results)} results -> {results_path}")


# ---------------------------------------------------------------------------
# GPU scheduling & orchestration
# ---------------------------------------------------------------------------

def _estimate_runtime(config):
    n_data = config["n_data"]
    hidden = config["hidden_dim"]
    blocks = config["num_blocks"]
    return n_data * hidden * hidden * blocks / 1e11


def balance_across_gpus(configs, num_gpus):
    ranked = sorted(enumerate(configs),
                    key=lambda t: _estimate_runtime(t[1]), reverse=True)
    buckets = [[] for _ in range(num_gpus)]
    loads = [0.0] * num_gpus
    for _, cfg in ranked:
        lightest = min(range(num_gpus), key=lambda g: loads[g])
        buckets[lightest].append(cfg)
        loads[lightest] += _estimate_runtime(cfg)
    return buckets


def launch_workers(configs, num_gpus):
    gpu_buckets = balance_across_gpus(configs, num_gpus)

    tmp = RESULTS_DIR / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    procs, res_paths = [], []
    for gpu_id, bucket in enumerate(gpu_buckets):
        if not bucket:
            continue
        cfg_path = tmp / f"strat_gpu{gpu_id}_configs.json"
        res_path = tmp / f"strat_gpu{gpu_id}_results.json"
        with open(cfg_path, "w") as f:
            json.dump(bucket, f)
        cmd = [sys.executable, "-u", __file__,
               "--worker",
               "--configs-path", str(cfg_path),
               "--device", f"cuda:{gpu_id}",
               "--results-path", str(res_path)]
        procs.append(subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr))
        res_paths.append(res_path)

    for p in procs:
        p.wait()

    failed = [p for p in procs if p.returncode != 0]
    if failed:
        print(f"ERROR: {len(failed)} worker(s) failed")
        sys.exit(1)

    results = []
    for rp in res_paths:
        with open(rp) as f:
            results.extend(json.load(f))
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scaling(results):
    from collections import defaultdict
    by_n = defaultdict(list)
    for r in results:
        by_n[r["n_data"]].append(r)

    sorted_ns = sorted(by_n.keys())
    cmap = LinearSegmentedColormap.from_list(
        "ltblue_purple", ["#89CFF0", "#6A0DAD"])
    colors = [cmap(i / max(1, len(sorted_ns) - 1))
              for i in range(len(sorted_ns))]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for idx, n_data in enumerate(sorted_ns):
        group = sorted(by_n[n_data], key=lambda r: r["n_params"])
        x = [r["n_params"] for r in group]
        y = [r["test_loss"] for r in group]
        if n_data >= 1_000_000:
            label = f"N={n_data / 1_000_000:.0f}M"
        else:
            label = f"N={n_data / 1_000:.0f}K"
        ax.plot(x, y, "o-", linewidth=1.8, markersize=5,
                color=colors[idx], label=label, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters (P)", fontsize=13)
    ax.set_ylabel("Test MSE (stratified)", fontsize=13)
    ax.set_title("Parameter Scaling: Mandelbrot (Smooth)", fontsize=14)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    filename = OUTPUT_DIR / "scaling_params_stratified.png"
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"  Saved {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--plot-only", action="store_true")
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

    results_path = RESULTS_DIR / "scaling_stratified.json"

    if args.plot_only:
        with open(results_path) as f:
            results = json.load(f)
        plot_scaling(results)
        return

    print("Step 1: Ensure master stratified dataset exists ...")
    get_or_build_master()

    configs = []
    for n_data in N_DATA_VALUES:
        for mc in MODEL_CONFIGS:
            configs.append({"n_data": n_data, **mc})

    print(f"\nStep 2: Launching {len(configs)} runs on {args.num_gpus} GPUs ...")
    results = launch_workers(configs, args.num_gpus)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\nStep 3: Generating plot ...")
    plot_scaling(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
