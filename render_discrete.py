"""
Render the Mandelbrot set in three styles:
  - binary membership (in/out)
  - smooth escape-time (the canonical continuous coloring)

Usage:
    python render_discrete.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from data import DEFAULT_MAX_ITER, DEFAULT_XLIM, compute_ylim, smooth_escape_grid

OUTPUT = Path("output")
RES = (1920, 1080)
MAX_ITER = DEFAULT_MAX_ITER
XLIM = DEFAULT_XLIM


def membership_grid(xs, ys, max_iter=MAX_ITER):
    """Binary membership: 1.0 if c is in the Mandelbrot set, 0.0 otherwise."""
    cx, cy = np.meshgrid(xs, ys)
    c = cx + 1j * cy
    z = np.zeros_like(c, dtype=np.complex128)
    escaped = np.zeros(c.shape, dtype=bool)
    active = np.ones(c.shape, dtype=bool)

    for n in range(max_iter):
        z[active] = z[active] ** 2 + c[active]
        r2 = z.real ** 2 + z.imag ** 2
        just_escaped = active & (r2 > 4.0)
        escaped[just_escaped] = True
        active[just_escaped] = False
        if not active.any():
            break

    return (~escaped).astype(np.float32)


def render_binary(xs, ys, extent):
    print("Computing binary membership ...")
    member = membership_grid(xs, ys, MAX_ITER)
    cmap = ListedColormap(["#1a1a2e", "#f5c518"])

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(member, extent=extent, cmap=cmap,
              vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Re(c)", fontsize=13)
    ax.set_ylabel("Im(c)", fontsize=13)
    ax.set_title("Mandelbrot Set — Binary Membership", fontsize=15)
    fig.tight_layout()
    out_path = OUTPUT / "discrete_mandelbrot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")


def render_smooth(xs, ys, extent):
    print("Computing smooth escape-time ...")
    grid = smooth_escape_grid(xs, ys, MAX_ITER)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(grid, extent=extent, cmap="inferno",
              vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("Re(c)", fontsize=13)
    ax.set_ylabel("Im(c)", fontsize=13)
    ax.set_title("Mandelbrot Set — Smooth Escape-Time", fontsize=15)
    fig.tight_layout()
    out_path = OUTPUT / "smooth_mandelbrot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.resolve()}")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ylim = compute_ylim(XLIM, RES)
    xs = np.linspace(XLIM[0], XLIM[1], RES[0], dtype=np.float64)
    ys = np.linspace(ylim[0], ylim[1], RES[1], dtype=np.float64)
    extent = [xs[0], xs[-1], ys[-1], ys[0]]

    render_binary(xs, ys, extent)
    render_smooth(xs, ys, extent)


if __name__ == "__main__":
    main()
