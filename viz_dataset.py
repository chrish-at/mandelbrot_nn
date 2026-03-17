"""
Visualize a cached Mandelbrot dataset as a 2D scatter plot.

Usage:
    python viz_dataset.py                          # default: data/dataset.npz
    python viz_dataset.py --path data/dataset.npz
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT = Path("output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/dataset.npz")
    parser.add_argument("--max_points", type=int, default=200_000,
                        help="Subsample for readability")
    args = parser.parse_args()

    d = np.load(args.path)
    X, y = d["X"], d["y"]
    tag = Path(args.path).stem
    print(f"Loaded {X.shape[0]} points from {args.path}")

    if X.shape[0] > args.max_points:
        idx = np.random.default_rng(0).choice(X.shape[0], args.max_points, replace=False)
        X, y = X[idx], y[idx]
        print(f"Subsampled to {args.max_points} for plotting")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: scatter colored by escape value ---
    fig, ax = plt.subplots(figsize=(14, 8))
    order = np.argsort(y)
    sc = ax.scatter(X[order, 0], X[order, 1], c=y[order], cmap="inferno",
                    s=0.15, alpha=0.6, rasterized=True)
    ax.set_xlabel("Re(c)", fontsize=13)
    ax.set_ylabel("Im(c)", fontsize=13)
    ax.set_title(f"Dataset samples colored by smooth escape value  ({tag})",
                 fontsize=15)
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label="Smooth escape value", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{tag}_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT / f'{tag}_scatter.png'}")

    # --- Plot 2: 2D density heatmap showing sample concentration ---
    fig, ax = plt.subplots(figsize=(14, 8))
    h = ax.hist2d(X[:, 0], X[:, 1], bins=(400, 225), cmap="magma",
                  norm=matplotlib.colors.LogNorm())
    ax.set_xlabel("Re(c)", fontsize=13)
    ax.set_ylabel("Im(c)", fontsize=13)
    ax.set_title(f"Sample density (log scale)  ({tag})", fontsize=15)
    ax.set_aspect("equal")
    fig.colorbar(h[3], ax=ax, label="Count per bin (log)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{tag}_density.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT / f'{tag}_density.png'}")

    # --- Plot 3: histogram of escape values ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(y, bins=200, color="#3498db", edgecolor="none", alpha=0.8)
    ax.set_xlabel("Smooth escape value", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(f"Distribution of target values  ({tag})", fontsize=15)
    ax.axvline(0.35, color="#e74c3c", ls="--", lw=1.5, label="Band low (0.35)")
    ax.axvline(0.95, color="#e74c3c", ls="--", lw=1.5, label="Band high (0.95)")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{tag}_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT / f'{tag}_histogram.png'}")


if __name__ == "__main__":
    main()
