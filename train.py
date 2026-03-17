"""
Training script for Mandelbrot neural networks.

Usage:
    python train.py --model baseline
    python train.py --model fourier
    python train.py --model both
    python train.py --model gated --gate_type bilinear
    python train.py --model gated --gate_type bilinear --weight_tie
    python train.py --model both --target discrete
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import get_or_build_dataset
from models import MLPRes, MLPFourierRes, MLPGatedRes, MLPFourierGatedRes

_GATED_DEFAULTS = dict(hidden_dim=512, num_blocks=20)


_BILINEAR_DEEP_DEFAULTS = dict(
    hidden_dim=128, num_blocks=100, gate_type="bilinear",
    weight_tie=True, use_layernorm=False, in_act="none",
)


def make_model(name, device, gate_type="bilinear", inner_dim=None,
               weight_tie=False):
    if name == "baseline":
        m = MLPRes(hidden_dim=512, num_blocks=20, act="silu")
    elif name == "fourier":
        m = MLPFourierRes(
            num_feats=512, sigmas=(2.0, 6.0, 10.0),
            hidden_dim=512, num_blocks=20, act="silu", seed=0,
        )
    elif name == "gated":
        m = MLPGatedRes(
            gate_type=gate_type, inner_dim=inner_dim,
            weight_tie=weight_tie, **_GATED_DEFAULTS,
        )
    elif name == "fourier_gated":
        m = MLPFourierGatedRes(
            num_feats=512, sigmas=(2.0, 6.0, 10.0), seed=0,
            gate_type=gate_type, inner_dim=inner_dim,
            weight_tie=weight_tie, **_GATED_DEFAULTS,
        )
    elif name == "bilinear_deep":
        m = MLPGatedRes(**_BILINEAR_DEEP_DEFAULTS)
    else:
        raise ValueError(name)
    return m.to(device)


def _ckpt_stem(model_name, target, gate_type="bilinear", weight_tie=False):
    stem = model_name
    if model_name in ("gated", "fourier_gated"):
        stem = f"{model_name}_{gate_type}"
        if weight_tie:
            stem += "_tied"
    if target == "discrete":
        stem += "_discrete"
    return stem


# bilinear_deep has its own stem — no gate_type/weight_tie suffix needed


def train_one(model_name, X, y, device, epochs=100, batch_size=4096,
              lr=3e-4, ckpt_dir=Path("checkpoints"), target="smooth",
              gate_type="bilinear", inner_dim=None, weight_tie=False):
    model = make_model(model_name, device, gate_type=gate_type,
                       inner_dim=inner_dim, weight_tie=weight_tie)
    n_params = sum(p.numel() for p in model.parameters())
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Training  {model_name}  target={target}  ({n_params:,} params)  for {epochs} epochs")
    print(sep)

    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).unsqueeze(-1).to(device)
    ds = TensorDataset(Xt, yt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss() if target == "discrete" else nn.MSELoss()

    history = []
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
        history.append(epoch_loss)
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr_now = sched.get_last_lr()[0]
            print(f"  epoch {epoch:3d}/{epochs}  loss={epoch_loss:.6f}  "
                  f"lr={lr_now:.2e}  [{elapsed:.0f}s]")

    stem = _ckpt_stem(model_name, target, gate_type=gate_type,
                      weight_tie=weight_tie)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{stem}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    hist_path = ckpt_dir / f"{stem}_loss.json"
    with open(hist_path, "w") as f:
        json.dump(history, f)
    print(f"  Loss history saved: {hist_path}")

    return model, history


_MODEL_GROUPS = {
    "both": ["baseline", "fourier"],
    "all": ["baseline", "fourier", "gated", "fourier_gated", "bilinear_deep"],
}
_ALL_MODELS = ["baseline", "fourier", "gated", "fourier_gated", "bilinear_deep"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="both",
                        choices=_ALL_MODELS + list(_MODEL_GROUPS))
    parser.add_argument("--target", default="smooth",
                        choices=["smooth", "discrete"])
    parser.add_argument("--gate_type", default="bilinear",
                        choices=["bilinear", "swiglu", "reglu", "geglu"])
    parser.add_argument("--inner_dim", type=int, default=None,
                        help="Inner dim for gated blocks (default: 2/3 * hidden_dim)")
    parser.add_argument("--weight_tie", action="store_true",
                        help="Tie weights across all gated blocks (unrolled RNN)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_samples", type=int, default=1_000_000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  target: {args.target}")

    X, y, ylim = get_or_build_dataset(
        target=args.target, n_total=args.n_samples,
    )

    models_to_train = _MODEL_GROUPS.get(args.model, [args.model])
    for name in models_to_train:
        train_one(name, X, y, device, epochs=args.epochs,
                  batch_size=args.batch_size, lr=args.lr,
                  target=args.target, gate_type=args.gate_type,
                  inner_dim=args.inner_dim, weight_tie=args.weight_tie)


if __name__ == "__main__":
    main()
