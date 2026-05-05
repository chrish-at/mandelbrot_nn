"""
Neural network architectures for learning the Mandelbrot set.

Models:
  - MLPRes:           Baseline deep residual MLP on raw (x, y) coordinates.
  - MLPFourierRes:    Same backbone but preceded by multi-scale Gaussian Fourier features.
  - MLPGatedRes:      Residual MLP with configurable gated blocks (bilinear, swiglu, etc.)
  - MLPFourierGatedRes: Fourier features + gated residual backbone.
  - HybridGatedRes:   Tied gated iteration blocks + untied SiLU readout head.
"""

import torch
import torch.nn as nn


def _make_norm(norm_type: str, dim: int):
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "rmsnorm":
        return nn.RMSNorm(dim)
    elif norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm_type: {norm_type}")


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, act: str = "silu", dropout: float = 0.0):
        super().__init__()
        activation = nn.ReLU if act.lower() == "relu" else nn.SiLU
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = activation()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.act(self.fc1(self.ln1(x)))
        h = self.drop(h)
        h = self.fc2(self.ln2(h))
        return x + h


class MultiScaleGaussianFourierFeatures(nn.Module):
    def __init__(self, in_dim: int = 2, num_feats: int = 512,
                 sigmas=(2.0, 6.0, 10.0), seed: int = 0):
        super().__init__()
        k = len(sigmas)
        per = [num_feats // k] * k
        per[0] += num_feats - sum(per)

        Bs = []
        g = torch.Generator()
        g.manual_seed(seed)
        for s, m in zip(sigmas, per):
            B = torch.randn(in_dim, m, generator=g) * s
            Bs.append(B)
        self.register_buffer("B", torch.cat(Bs, dim=1))

    def forward(self, x):
        proj = (2 * torch.pi) * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


_GATE_ACTIVATIONS = {
    "bilinear": nn.Identity,
    "swiglu": nn.SiLU,
    "reglu": nn.ReLU,
    "geglu": nn.GELU,
    "glu": nn.Sigmoid,
}


class GatedResidualBlock(nn.Module):
    """Gated residual block: x + W_down(gate_act(W_gate·h) * W_value·h).

    With gate_type="bilinear" this is a pure quadratic (degree-2) map,
    which can exactly represent one Mandelbrot iteration z² + c.
    """

    def __init__(self, dim: int, inner_dim: int | None = None,
                 gate_type: str = "bilinear", dropout: float = 0.0,
                 norm_type: str = "layernorm"):
        super().__init__()
        if inner_dim is None:
            inner_dim = (dim * 2 // 3 + 7) & ~7
        act_cls = _GATE_ACTIVATIONS[gate_type]
        self.ln = _make_norm(norm_type, dim)
        self.fc_gate = nn.Linear(dim, inner_dim)
        self.fc_value = nn.Linear(dim, inner_dim)
        self.gate_act = act_cls()
        self.fc_down = nn.Linear(inner_dim, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.zeros_(self.fc_down.weight)
        nn.init.zeros_(self.fc_down.bias)

    def forward(self, x):
        h = self.ln(x)
        h = self.gate_act(self.fc_gate(h)) * self.fc_value(h)
        h = self.drop(h)
        h = self.fc_down(h)
        return x + h


class MLPGatedRes(nn.Module):
    """Residual MLP with gated blocks on raw coordinates.

    Args:
        weight_tie: If True, a single GatedResidualBlock is reused for all
            iterations (unrolled RNN), matching the repeated-function structure
            of the Mandelbrot iteration.
        norm_type: "layernorm", "rmsnorm", or "none".
        in_act: Input activation — "silu" or "none".
    """

    def __init__(self, hidden_dim=512, num_blocks=20, gate_type="bilinear",
                 inner_dim=None, weight_tie=False, dropout=0.0, out_dim=1,
                 norm_type="layernorm", in_act="silu"):
        super().__init__()
        self.in_proj = nn.Linear(2, hidden_dim)
        self.in_act = nn.SiLU() if in_act == "silu" else nn.Identity()
        self.weight_tie = weight_tie
        self.num_iters = num_blocks
        if weight_tie:
            self.block = GatedResidualBlock(
                hidden_dim, inner_dim=inner_dim,
                gate_type=gate_type, dropout=dropout,
                norm_type=norm_type,
            )
        else:
            self.blocks = nn.Sequential(
                *[GatedResidualBlock(hidden_dim, inner_dim=inner_dim,
                                     gate_type=gate_type, dropout=dropout,
                                     norm_type=norm_type)
                  for _ in range(num_blocks)]
            )
        self.out_ln = _make_norm(norm_type, hidden_dim)
        self.out_act = nn.SiLU() if in_act == "silu" else nn.Identity()
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.in_act(self.in_proj(x))
        if self.weight_tie:
            for _ in range(self.num_iters):
                x = self.block(x)
        else:
            x = self.blocks(x)
        x = self.out_act(self.out_ln(x))
        return self.out_proj(x)


class MLPFourierGatedRes(nn.Module):
    """Fourier features + gated residual backbone with optional weight tying."""

    def __init__(self, num_feats=512, sigmas=(2.0, 6.0, 10.0),
                 hidden_dim=512, num_blocks=20, gate_type="bilinear",
                 inner_dim=None, weight_tie=False, dropout=0.0,
                 out_dim=1, seed=0, norm_type="layernorm"):
        super().__init__()
        self.ff = MultiScaleGaussianFourierFeatures(
            2, num_feats=num_feats, sigmas=sigmas, seed=seed,
        )
        self.in_proj = nn.Linear(2 * num_feats, hidden_dim)
        self.in_act = nn.SiLU()
        self.weight_tie = weight_tie
        self.num_iters = num_blocks
        if weight_tie:
            self.block = GatedResidualBlock(
                hidden_dim, inner_dim=inner_dim,
                gate_type=gate_type, dropout=dropout,
                norm_type=norm_type,
            )
        else:
            self.blocks = nn.Sequential(
                *[GatedResidualBlock(hidden_dim, inner_dim=inner_dim,
                                     gate_type=gate_type, dropout=dropout,
                                     norm_type=norm_type)
                  for _ in range(num_blocks)]
            )
        self.out_ln = _make_norm(norm_type, hidden_dim)
        self.out_act = nn.SiLU()
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.ff(x)
        x = self.in_act(self.in_proj(x))
        if self.weight_tie:
            for _ in range(self.num_iters):
                x = self.block(x)
        else:
            x = self.blocks(x)
        x = self.out_act(self.out_ln(x))
        return self.out_proj(x)


class HybridGatedRes(nn.Module):
    """Tied gated iteration blocks followed by untied SiLU readout head.

    Separates the repeated quadratic iteration (tied gated blocks) from
    the escape-time readout (untied SiLU MLP blocks).
    """

    def __init__(self, hidden_dim=512, num_iter_blocks=20, num_head_blocks=4,
                 gate_type="bilinear", inner_dim=None, dropout=0.0,
                 out_dim=1, norm_type="layernorm"):
        super().__init__()
        self.in_proj = nn.Linear(2, hidden_dim)
        self.in_act = nn.SiLU()
        self.num_iters = num_iter_blocks
        self.iter_block = GatedResidualBlock(
            hidden_dim, inner_dim=inner_dim,
            gate_type=gate_type, dropout=dropout,
            norm_type=norm_type,
        )
        self.head_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, act="silu", dropout=dropout)
              for _ in range(num_head_blocks)]
        )
        self.out_ln = _make_norm(norm_type, hidden_dim)
        self.out_act = nn.SiLU()
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.in_act(self.in_proj(x))
        for _ in range(self.num_iters):
            x = self.iter_block(x)
        x = self.head_blocks(x)
        x = self.out_act(self.out_ln(x))
        return self.out_proj(x)


class MLPRes(nn.Module):
    """Baseline residual MLP on raw coordinates."""

    def __init__(self, hidden_dim=512, num_blocks=20, act="silu",
                 dropout=0.0, out_dim=1):
        super().__init__()
        self.in_proj = nn.Linear(2, hidden_dim)
        self.in_act = nn.SiLU() if act == "silu" else nn.ReLU()
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, act=act, dropout=dropout)
              for _ in range(num_blocks)]
        )
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.out_act = nn.SiLU() if act == "silu" else nn.ReLU()
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.in_act(self.in_proj(x))
        x = self.blocks(x)
        x = self.out_act(self.out_ln(x))
        return self.out_proj(x)


class MLPFourierRes(nn.Module):
    """Residual MLP with multi-scale Gaussian Fourier feature input encoding."""

    def __init__(self, num_feats=512, sigmas=(2.0, 6.0, 10.0),
                 hidden_dim=512, num_blocks=20, act="silu",
                 dropout=0.0, out_dim=1, seed=0):
        super().__init__()
        self.ff = MultiScaleGaussianFourierFeatures(
            2, num_feats=num_feats, sigmas=sigmas, seed=seed,
        )
        self.in_proj = nn.Linear(2 * num_feats, hidden_dim)
        self.in_act = nn.SiLU() if act == "silu" else nn.ReLU()
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, act=act, dropout=dropout)
              for _ in range(num_blocks)]
        )
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.out_act = nn.SiLU() if act == "silu" else nn.ReLU()
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.ff(x)
        x = self.in_act(self.in_proj(x))
        x = self.blocks(x)
        x = self.out_act(self.out_ln(x))
        return self.out_proj(x)
