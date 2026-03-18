"""
Escape-time computation and boundary-biased dataset generation
for learning the Mandelbrot set.

Supports two target modes:
  - "smooth": continuous escape-time in [0, 1]
  - "discrete": binary membership (1.0 inside, 0.0 outside)
"""

import math
import numpy as np
from pathlib import Path

DEFAULT_MAX_ITER = 1000
DEFAULT_XLIM = (-2.4, 1.0)
DEFAULT_RES = (3840, 2160)


def smooth_escape(x: float, y: float, max_iter: int = DEFAULT_MAX_ITER) -> float:
    """Scalar smooth escape-time value in [0, 1]."""
    c = complex(x, y)
    z = 0j
    for n in range(max_iter):
        z = z * z + c
        r2 = z.real * z.real + z.imag * z.imag
        if r2 > 4.0:
            r = math.sqrt(r2)
            mu = n + 1 - math.log(math.log(r)) / math.log(2.0)
            v = math.log1p(mu) / math.log1p(max_iter)
            return float(np.clip(v, 0.0, 1.0))
    return 1.0


def smooth_escape_grid(xs, ys, max_iter: int = DEFAULT_MAX_ITER) -> np.ndarray:
    """
    Compute smooth escape-time on a meshgrid.
    xs: 1-D array of x coords, ys: 1-D array of y coords.
    Returns (len(ys), len(xs)) float32 array.
    """
    cx, cy = np.meshgrid(xs, ys)
    c = cx + 1j * cy
    z = np.zeros_like(c, dtype=np.complex128)
    escape = np.full(c.shape, max_iter, dtype=np.float64)
    r2_at_escape = np.full(c.shape, 0.0, dtype=np.float64)
    active = np.ones(c.shape, dtype=bool)

    for n in range(max_iter):
        z[active] = z[active] ** 2 + c[active]
        r2 = z.real ** 2 + z.imag ** 2
        escaped = active & (r2 > 4.0)
        escape[escaped] = n
        r2_at_escape[escaped] = r2[escaped]
        active[escaped] = False
        if not active.any():
            break

    inside = escape == max_iter
    r = np.sqrt(r2_at_escape)
    r = np.clip(r, 1e-30, None)
    log_r = np.log(r)
    log_r = np.clip(log_r, 1e-30, None)
    mu = escape + 1.0 - np.log(log_r) / math.log(2.0)
    v = np.log1p(mu) / math.log1p(max_iter)
    v = np.clip(v, 0.0, 1.0)
    v[inside] = 1.0
    return v.astype(np.float32)


def compute_ylim(xlim, resolution, ycenter=0.0):
    aspect = resolution[1] / resolution[0]
    xspan = xlim[1] - xlim[0]
    yspan = xspan * aspect
    return (ycenter - yspan / 2, ycenter + yspan / 2)


def sample_uniform(n, xlim, ylim, rng):
    x = rng.uniform(xlim[0], xlim[1], size=(n, 1)).astype(np.float32)
    y = rng.uniform(ylim[0], ylim[1], size=(n, 1)).astype(np.float32)
    return np.hstack([x, y])


def smooth_escape_batch(X: np.ndarray, max_iter: int = DEFAULT_MAX_ITER) -> np.ndarray:
    """Vectorised smooth escape for an (N, 2) array of points."""
    N = X.shape[0]
    cx = X[:, 0].astype(np.float64)
    cy = X[:, 1].astype(np.float64)
    c = cx + 1j * cy
    z = np.zeros(N, dtype=np.complex128)
    escape = np.full(N, max_iter, dtype=np.float64)
    r2_at_escape = np.zeros(N, dtype=np.float64)
    active = np.ones(N, dtype=bool)

    for n in range(max_iter):
        z[active] = z[active] ** 2 + c[active]
        r2 = z.real ** 2 + z.imag ** 2
        escaped = active & (r2 > 4.0)
        escape[escaped] = n
        r2_at_escape[escaped] = r2[escaped]
        active[escaped] = False
        if not active.any():
            break

    inside = escape == max_iter
    r = np.sqrt(np.clip(r2_at_escape, 1e-30, None))
    log_r = np.clip(np.log(r), 1e-30, None)
    mu = escape + 1.0 - np.log(log_r) / math.log(2.0)
    v = np.log1p(mu) / math.log1p(max_iter)
    v = np.clip(v, 0.0, 1.0)
    v[inside] = 1.0
    return v.astype(np.float32)


def discrete_escape_batch(X: np.ndarray, max_iter: int = DEFAULT_MAX_ITER) -> np.ndarray:
    """Vectorised binary membership for an (N, 2) array of points.
    Returns 1.0 (inside) or 0.0 (outside)."""
    N = X.shape[0]
    cx = X[:, 0].astype(np.float64)
    cy = X[:, 1].astype(np.float64)
    c = cx + 1j * cy
    z = np.zeros(N, dtype=np.complex128)
    escaped = np.zeros(N, dtype=bool)
    active = np.ones(N, dtype=bool)

    for n in range(max_iter):
        z[active] = z[active] ** 2 + c[active]
        r2 = z.real ** 2 + z.imag ** 2
        just_escaped = active & (r2 > 4.0)
        escaped[just_escaped] = True
        active[just_escaped] = False
        if not active.any():
            break

    return (~escaped).astype(np.float32)


def discrete_escape_grid(xs, ys, max_iter: int = DEFAULT_MAX_ITER) -> np.ndarray:
    """Binary membership on a meshgrid. Returns (len(ys), len(xs)) float32 array."""
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


def build_boundary_biased_dataset(
    n_total: int = 1_000_000,
    frac_boundary: float = 0.7,
    xlim=DEFAULT_XLIM,
    resolution=DEFAULT_RES,
    ycenter: float = 0.0,
    max_iter: int = DEFAULT_MAX_ITER,
    band=(0.35, 0.95),
    seed: int = 0,
    pool_chunk_size: int = 10_000_000,
):
    rng = np.random.default_rng(seed)
    ylim = compute_ylim(xlim, resolution, ycenter=ycenter)

    n_boundary = int(n_total * frac_boundary)
    n_uniform = n_total - n_boundary

    Xu = sample_uniform(n_uniform, xlim, ylim, rng)

    pool_factor = 20
    n_pool_total = n_boundary * pool_factor

    Xb_parts, yb_parts = [], []
    collected = 0
    for start in range(0, n_pool_total, pool_chunk_size):
        chunk_n = min(pool_chunk_size, n_pool_total - start)
        pool = sample_uniform(chunk_n, xlim, ylim, rng)
        yp = smooth_escape_batch(pool, max_iter=max_iter)
        mask = (yp > band[0]) & (yp < band[1])
        Xb_parts.append(pool[mask])
        yb_parts.append(yp[mask])
        collected += mask.sum()
        if collected >= n_boundary:
            break
        print(f"    boundary pool chunk: {collected:,}/{n_boundary:,} collected")

    Xb = np.concatenate(Xb_parts, axis=0)
    yb = np.concatenate(yb_parts, axis=0)

    if len(Xb) < n_boundary:
        keep = min(len(Xb), n_boundary)
        print(f"[warn] Boundary band too strict; got {len(Xb)} points, using {keep}.")
        Xb = Xb[:keep]
        yb = yb[:keep]
        n_boundary = keep
        n_uniform = n_total - n_boundary
        Xu = sample_uniform(n_uniform, xlim, ylim, rng)
    else:
        Xb = Xb[:n_boundary]
        yb = yb[:n_boundary]

    yu = smooth_escape_batch(Xu, max_iter=max_iter)

    X = np.concatenate([Xu, Xb], axis=0).astype(np.float32)
    y = np.concatenate([yu, yb], axis=0).astype(np.float32)

    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm], ylim


def _cache_path(base: str, target: str) -> str:
    if target == "discrete":
        p = Path(base)
        return str(p.with_stem(p.stem + "_discrete"))
    return base


def _build_discrete_from_smooth(smooth_cache: str, discrete_cache: str):
    """Load X from the smooth dataset and compute exact binary labels."""
    d = np.load(smooth_cache)
    X, ylim = d["X"], tuple(d["ylim"])
    print(f"Computing discrete labels for {X.shape[0]} points ...")
    y = discrete_escape_batch(X)
    Path(discrete_cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez(discrete_cache, X=X, y=y, ylim=np.array(ylim))
    print(f"Discrete dataset saved to {discrete_cache}")
    return X, y, ylim


def get_or_build_dataset(cache_path: str = None, target: str = "smooth", **kwargs):
    smooth_path = "data/dataset.npz"
    if cache_path is None:
        cache_path = _cache_path(smooth_path, target)
    cache = Path(cache_path)
    if cache.exists():
        print(f"Loading cached dataset from {cache}")
        d = np.load(cache)
        return d["X"], d["y"], tuple(d["ylim"])

    if target == "discrete":
        smooth = Path(smooth_path)
        if not smooth.exists():
            print("Smooth dataset not found; building it first ...")
            get_or_build_dataset(cache_path=smooth_path, target="smooth", **kwargs)
        return _build_discrete_from_smooth(smooth_path, cache_path)

    print("Building dataset (this takes a few minutes) ...")
    X, y, ylim = build_boundary_biased_dataset(**kwargs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=y, ylim=np.array(ylim))
    print(f"Dataset saved to {cache}  ({X.shape[0]} samples)")
    return X, y, ylim


def _master_smooth_path(n_total: int) -> str:
    return f"data/master_{n_total // 1_000_000}M.npz"


def get_or_build_master(cache_path: str = None,
                        n_total: int = 2_000_000, target: str = "smooth", **kwargs):
    """Build or load a master dataset used for scaling experiments."""
    smooth_path = _master_smooth_path(n_total)
    if cache_path is None:
        cache_path = _cache_path(smooth_path, target)
    cache = Path(cache_path)
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"], tuple(d["ylim"])

    if target == "discrete":
        smooth = Path(smooth_path)
        if not smooth.exists():
            print("Smooth master dataset not found; building it first ...")
            get_or_build_master(cache_path=smooth_path, n_total=n_total,
                                target="smooth", **kwargs)
        return _build_discrete_from_smooth(smooth_path, cache_path)

    print(f"Building {n_total/1e6:.0f}M master dataset ...")
    X, y, ylim = build_boundary_biased_dataset(n_total=n_total, **kwargs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=y, ylim=np.array(ylim))
    print(f"Master dataset saved to {cache}  ({X.shape[0]} samples)")
    return X, y, ylim


def subsample_dataset(X, y, n, seed=42):
    """Deterministically subsample n points from a larger dataset."""
    if n >= X.shape[0]:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], n, replace=False)
    return X[idx], y[idx]
