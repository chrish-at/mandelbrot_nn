"""
Training script for Mandelbrot neural networks.

Usage:
    python train.py --model baseline
    python train.py --model fourier
    python train.py --model gated --gate_type bilinear --weight_tie
    python train.py --model fourier_swiglu_tied
    python train.py --model hybrid_bilinear
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import get_or_build_dataset, get_or_build_stratified
from models import (
    MLPRes, MLPFourierRes, MLPGatedRes, MLPFourierGatedRes, HybridGatedRes,
)

# ---------------------------------------------------------------------------
# Named experiment configs: each maps a model name to constructor kwargs
# and the class to instantiate. This avoids a giant if/elif chain.
# ---------------------------------------------------------------------------

_NAMED_EXPERIMENTS = {
    "baseline": (MLPRes, dict(hidden_dim=512, num_blocks=20, act="silu")),
    "fourier": (MLPFourierRes, dict(
        num_feats=512, sigmas=(2.0, 6.0, 10.0),
        hidden_dim=512, num_blocks=20, act="silu", seed=0,
    )),
    "bilinear_deep": (MLPGatedRes, dict(
        hidden_dim=128, num_blocks=100, gate_type="bilinear",
        weight_tie=True, norm_type="none", in_act="none",
    )),
    # --- Previously trained gated models (backward compat for render) ---
    "gated_bilinear": (MLPGatedRes, dict(
        hidden_dim=512, num_blocks=20, gate_type="bilinear", weight_tie=False,
    )),
    "gated_bilinear_tied": (MLPGatedRes, dict(
        hidden_dim=512, num_blocks=20, gate_type="bilinear", weight_tie=True,
    )),
    "swiglu_untied": (MLPGatedRes, dict(
        hidden_dim=512, num_blocks=20, gate_type="swiglu", weight_tie=False,
    )),
    # --- Gate type comparison (tied, hidden=512, 20 blocks, LayerNorm) ---
    "swiglu_tied": (MLPGatedRes, dict(
        hidden_dim=512, num_blocks=20, gate_type="swiglu", weight_tie=True,
    )),
    "glu_tied": (MLPGatedRes, dict(
        hidden_dim=512, num_blocks=20, gate_type="glu", weight_tie=True,
    )),
    "geglu_tied": (MLPGatedRes, dict(
        hidden_dim=512, num_blocks=20, gate_type="geglu", weight_tie=True,
    )),
    # --- Normalization ablations ---
    "bilinear_tied_noln": (MLPGatedRes, dict(
        hidden_dim=512, num_blocks=20, gate_type="bilinear",
        weight_tie=True, norm_type="none",
    )),
    "bilinear_deep_ln": (MLPGatedRes, dict(
        hidden_dim=128, num_blocks=100, gate_type="bilinear",
        weight_tie=True, norm_type="layernorm", in_act="silu",
    )),
    "bilinear_tied_small": (MLPGatedRes, dict(
        hidden_dim=128, num_blocks=20, gate_type="bilinear",
        weight_tie=True, norm_type="layernorm",
    )),
    "swiglu_tied_rms": (MLPGatedRes, dict(
        hidden_dim=512, num_blocks=20, gate_type="swiglu",
        weight_tie=True, norm_type="rmsnorm",
    )),
    # --- Fourier + gated ---
    "fourier_swiglu_tied": (MLPFourierGatedRes, dict(
        num_feats=512, sigmas=(2.0, 6.0, 10.0), seed=0,
        hidden_dim=512, num_blocks=20, gate_type="swiglu", weight_tie=True,
    )),
    "fourier_swiglu_untied": (MLPFourierGatedRes, dict(
        num_feats=512, sigmas=(2.0, 6.0, 10.0), seed=0,
        hidden_dim=512, num_blocks=20, gate_type="swiglu", weight_tie=False,
    )),
    # --- High-frequency Fourier variants ---
    "fourier_hf": (MLPFourierRes, dict(
        num_feats=512, sigmas=(2.0, 6.0, 10.0, 30.0, 100.0),
        hidden_dim=512, num_blocks=20, act="silu", seed=0,
    )),
    "fourier_swiglu_untied_hf": (MLPFourierGatedRes, dict(
        num_feats=512, sigmas=(2.0, 6.0, 10.0, 30.0, 100.0), seed=0,
        hidden_dim=512, num_blocks=20, gate_type="swiglu", weight_tie=False,
    )),
    # --- Hybrid: tied iteration blocks + untied SiLU readout head ---
    "hybrid_bilinear": (HybridGatedRes, dict(
        hidden_dim=512, num_iter_blocks=20, num_head_blocks=4,
        gate_type="bilinear",
    )),
    "hybrid_swiglu": (HybridGatedRes, dict(
        hidden_dim=512, num_iter_blocks=20, num_head_blocks=4,
        gate_type="swiglu",
    )),
}

_GATED_DEFAULTS = dict(hidden_dim=512, num_blocks=20)


def make_model(name, device, gate_type="bilinear", inner_dim=None,
               weight_tie=False, norm_type="layernorm"):
    if name in _NAMED_EXPERIMENTS:
        cls, kwargs = _NAMED_EXPERIMENTS[name]
        m = cls(**kwargs)
    elif name == "gated":
        m = MLPGatedRes(
            gate_type=gate_type, inner_dim=inner_dim,
            weight_tie=weight_tie, norm_type=norm_type, **_GATED_DEFAULTS,
        )
    elif name == "fourier_gated":
        m = MLPFourierGatedRes(
            num_feats=512, sigmas=(2.0, 6.0, 10.0), seed=0,
            gate_type=gate_type, inner_dim=inner_dim,
            weight_tie=weight_tie, norm_type=norm_type, **_GATED_DEFAULTS,
        )
    else:
        raise ValueError(name)
    return m.to(device)


def _ckpt_stem(model_name, target, gate_type="bilinear", weight_tie=False,
               **_kwargs):
    if model_name in _NAMED_EXPERIMENTS:
        stem = model_name
    elif model_name in ("gated", "fourier_gated"):
        stem = f"{model_name}_{gate_type}"
        if weight_tie:
            stem += "_tied"
    else:
        stem = model_name
    if target == "discrete":
        stem += "_discrete"
    return stem


def train_one(model_name, X, y, device, epochs=100, batch_size=4096,
              lr=3e-4, ckpt_dir=Path("checkpoints"), target="smooth",
              gate_type="bilinear", inner_dim=None, weight_tie=False,
              norm_type="layernorm", X_test=None, y_test=None,
              ckpt_suffix=""):
    model = make_model(model_name, device, gate_type=gate_type,
                       inner_dim=inner_dim, weight_tie=weight_tie,
                       norm_type=norm_type)
    n_params = sum(p.numel() for p in model.parameters())
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Training  {model_name}  target={target}  ({n_params:,} params)  for {epochs} epochs")
    print(sep)

    Xt_train = torch.from_numpy(X).to(device)
    yt_train = torch.from_numpy(y).unsqueeze(-1).to(device)
    has_test = X_test is not None and y_test is not None
    if has_test:
        Xt_test = torch.from_numpy(X_test).to(device)
        yt_test = torch.from_numpy(y_test).unsqueeze(-1).to(device)
    print(f"  train={len(X):,}  test={len(X_test) if has_test else 0:,}")

    ds = TensorDataset(Xt_train, yt_train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss() if target == "discrete" else nn.MSELoss()

    train_history, test_history = [], []
    t0 = time.time()

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
        epoch_loss = running / count
        train_history.append(epoch_loss)

        if has_test:
            model.eval()
            with torch.no_grad():
                test_pred = model(Xt_test)
                test_loss = loss_fn(test_pred, yt_test).item()
            test_history.append(test_loss)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr_now = sched.get_last_lr()[0]
            test_str = f"  test={test_loss:.6f}" if has_test else ""
            print(f"  epoch {epoch:3d}/{epochs}  train={epoch_loss:.6f}"
                  f"{test_str}  lr={lr_now:.2e}  [{elapsed:.0f}s]")

    stem = _ckpt_stem(model_name, target, gate_type=gate_type,
                      weight_tie=weight_tie, norm_type=norm_type)
    stem += ckpt_suffix
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{stem}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    hist = {"train": train_history}
    if test_history:
        hist["test"] = test_history
    hist_path = ckpt_dir / f"{stem}_loss.json"
    with open(hist_path, "w") as f:
        json.dump(hist, f)
    print(f"  Loss history saved: {hist_path}")

    return model, train_history


_MODEL_GROUPS = {
    "both": ["baseline", "fourier"],
    "both_hf": ["baseline", "fourier_hf"],
    "hf": ["fourier_hf", "fourier_swiglu_untied_hf"],
    "all": list(_NAMED_EXPERIMENTS.keys()),
}
_VALID_MODELS = (
    list(_NAMED_EXPERIMENTS.keys()) + ["gated", "fourier_gated"]
    + list(_MODEL_GROUPS.keys())
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="both", choices=_VALID_MODELS)
    parser.add_argument("--target", default="smooth",
                        choices=["smooth", "discrete"])
    parser.add_argument("--gate_type", default="bilinear",
                        choices=["bilinear", "swiglu", "reglu", "geglu", "glu"])
    parser.add_argument("--inner_dim", type=int, default=None,
                        help="Inner dim for gated blocks (default: 2/3 * hidden_dim)")
    parser.add_argument("--weight_tie", action="store_true",
                        help="Tie weights across all gated blocks (unrolled RNN)")
    parser.add_argument("--norm_type", default="layernorm",
                        choices=["layernorm", "rmsnorm", "none"])
    parser.add_argument("--dataset", default="boundary",
                        choices=["boundary", "stratified"],
                        help="Sampling strategy for the dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_samples", type=int, default=1_000_000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  target: {args.target}  dataset: {args.dataset}")

    X_test, y_test, ckpt_suffix = None, None, ""
    if args.dataset == "stratified":
        X, y, X_test, y_test, ylim = get_or_build_stratified(
            n_total=args.n_samples,
        )
        ckpt_suffix = "_strat"
    else:
        X, y, ylim = get_or_build_dataset(
            target=args.target, n_total=args.n_samples,
        )
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(X))
        split = int(0.9 * len(X))
        X, X_test = X[idx[:split]], X[idx[split:]]
        y, y_test = y[idx[:split]], y[idx[split:]]

    models_to_train = _MODEL_GROUPS.get(args.model, [args.model])
    for name in models_to_train:
        train_one(name, X, y, device, epochs=args.epochs,
                  batch_size=args.batch_size, lr=args.lr,
                  target=args.target, gate_type=args.gate_type,
                  inner_dim=args.inner_dim, weight_tie=args.weight_tie,
                  norm_type=args.norm_type,
                  X_test=X_test, y_test=y_test,
                  ckpt_suffix=ckpt_suffix)


if __name__ == "__main__":
    main()
