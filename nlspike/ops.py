import math
from typing import Optional

import torch
import torch.nn as nn


def quantize_floor(x: torch.Tensor, q_bits: int = 8) -> torch.Tensor:
    """Uniform floor quantization used by the float reference operators."""
    step = x.new_tensor(float(1 << q_bits))
    return torch.floor(x * step) / step


class PiecewiseLinearExp(nn.Module):
    """Piecewise-linear float reference for exp(x) on [-clip_bound, clip_bound].

    Values below -clip_bound are treated as zero. Values above clip_bound are
    clamped to exp(clip_bound). This mirrors the clipping used by the NLSpike
    operator approximation while staying independent of any SNN runtime.
    """

    def __init__(self, clip_bound: float = 5.0, n_segments: int = 64):
        super().__init__()
        if clip_bound <= 0:
            raise ValueError("clip_bound must be positive")
        if n_segments <= 0:
            raise ValueError("n_segments must be positive")

        self.clip_bound = float(clip_bound)
        self.n_segments = int(n_segments)

        x_bound = torch.linspace(-self.clip_bound, self.clip_bound, self.n_segments + 1)
        y_bound = torch.exp(x_bound)
        width = (2.0 * self.clip_bound) / self.n_segments
        slope = (y_bound[1:] - y_bound[:-1]) / width
        bias = y_bound[:-1] - slope * x_bound[:-1]

        self.register_buffer("x_bound", x_bound)
        self.register_buffer("slope", slope)
        self.register_buffer("bias", bias)
        self.register_buffer("zero", torch.tensor(0.0))
        self.register_buffer("upper", torch.exp(torch.tensor(self.clip_bound)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bound = self.x_bound.to(device=x.device, dtype=x.dtype)
        slope = self.slope.to(device=x.device, dtype=x.dtype)
        bias = self.bias.to(device=x.device, dtype=x.dtype)
        zero = self.zero.to(device=x.device, dtype=x.dtype)
        upper = self.upper.to(device=x.device, dtype=x.dtype)

        idx = torch.bucketize(x, x_bound[1:-1])
        approx = slope[idx] * x + bias[idx]
        return torch.where(x < -self.clip_bound, zero, torch.where(x > self.clip_bound, upper, approx))


class NLSpikeSiLU(nn.Module):
    """Float reference for the NLSpike SiLU replacement."""

    def __init__(self, clip_bound: float = 5.0, n_segments: int = 64, q_bits: int = 8):
        super().__init__()
        self.clip_bound = float(clip_bound)
        self.q_bits = int(q_bits)
        self.exp = PiecewiseLinearExp(clip_bound=clip_bound, n_segments=n_segments)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exp_neg = self.exp(-x).to(dtype=x.dtype)
        y = x / (1.0 + exp_neg)
        y = torch.where(x < -self.clip_bound, torch.zeros_like(y), y)
        return quantize_floor(y, self.q_bits)


class NLSpikeSoftmax(nn.Module):
    """Float reference for the NLSpike Softmax replacement."""

    def __init__(
        self,
        dim: int = -1,
        clip_bound: float = 5.0,
        n_segments: int = 64,
        q_bits: int = 8,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.dim = dim
        self.clip_bound = float(clip_bound)
        self.q_bits = int(q_bits)
        self.eps = float(eps)
        self.exp = PiecewiseLinearExp(clip_bound=clip_bound, n_segments=n_segments)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        shifted = logits - logits.max(dim=self.dim, keepdim=True).values + self.clip_bound
        exp_values = self.exp(shifted).to(dtype=logits.dtype)
        probs = exp_values / exp_values.sum(dim=self.dim, keepdim=True).clamp_min(self.eps)
        return quantize_floor(probs, self.q_bits)


def _cordic_gain(n_iter: int) -> float:
    gain = 1.0
    for k in range(n_iter):
        gain *= math.sqrt(1.0 + 2.0 ** (-2 * k))
    return gain


def _cordic_hypot_pair(x: torch.Tensor, y: torch.Tensor, n_iter: int) -> torch.Tensor:
    xi = x.clone()
    yi = y.clone()
    one = torch.ones((), dtype=xi.dtype, device=xi.device)
    minus_one = -one

    for k in range(n_iter):
        direction = torch.where(yi >= 0, one, minus_one)
        x_shift = yi / (1 << k)
        y_shift = xi / (1 << k)
        xi = xi + direction * x_shift
        yi = yi - direction * y_shift

    return xi.abs() / xi.new_tensor(_cordic_gain(n_iter))


def cordic_l2(x: torch.Tensor, eps: float = 0.0, n_iter: int = 8) -> torch.Tensor:
    """Approximate the last-dimension L2 norm with a CORDIC-style tree."""
    if x.size(-1) == 0:
        raise ValueError("last dimension must be non-empty")

    values = x.abs()
    if eps > 0:
        eps_value = torch.full_like(values[..., :1], math.sqrt(eps))
        values = torch.cat([values, eps_value], dim=-1)

    values = values.sort(dim=-1).values
    while values.size(-1) > 1:
        dim = values.size(-1)
        half = dim // 2
        left = values[..., : 2 * half : 2]
        right = values[..., 1 : 2 * half : 2]
        merged = _cordic_hypot_pair(left, right, n_iter=n_iter)
        if dim % 2 == 1:
            values = torch.cat([merged, values[..., -1:]], dim=-1)
        else:
            values = merged
    return values.squeeze(-1)


def nlspike_rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    n_iter: int = 8,
    q_bits: int = 8,
) -> torch.Tensor:
    """Float reference for NLSpike RMSNorm."""
    hidden_size = x.size(-1)
    norm = cordic_l2(x, eps=eps * hidden_size, n_iter=n_iter)
    y = x * (math.sqrt(hidden_size) / norm).unsqueeze(-1)
    if weight is not None:
        y = y * weight
    return quantize_floor(y, q_bits)


class NLSpikeRMSNorm(nn.Module):
    """RMSNorm module wrapper for the NLSpike float reference."""

    def __init__(
        self,
        hidden_size: Optional[int] = None,
        eps: float = 1e-5,
        n_iter: int = 8,
        q_bits: int = 8,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.eps = float(eps)
        self.n_iter = int(n_iter)
        self.q_bits = int(q_bits)
        if hidden_size is not None and elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nlspike_rms_norm(
            x,
            weight=self.weight,
            eps=self.eps,
            n_iter=self.n_iter,
            q_bits=self.q_bits,
        )
