from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class MHAConfig:
    """Configuration for MHA."""
    batch: int
    seq_len: int
    heads: int
    dim: int           # per-head dimension (Hd)
    causal: bool
    dtype: torch.dtype = torch.float16
    dropout_p: float = 0.0

def attn_flops_forward(cfg: MHAConfig) -> float:
    """Return forward FLOPs for attention core (QK^T and P·V).

    FLOPs ≈ 4 * B * H * S^2 * Hd
    - QK^T: 2 * B * H * S * S * Hd
    - Softmax omitted (treated as lower-order constant here).
    - P·V:  2 * B * H * S * S * Hd
    """
    B, H, S, Hd = cfg.batch, cfg.heads, cfg.seq_len, cfg.dim
    return 4.0 * B * H * (S ** 2) * Hd


def attn_flops_forward_backward(cfg: MHAConfig) -> float:
    """Approximate FLOPs for forward+backward of attention core.

    Empirical rule of thumb: backward is ~2× forward (dV, dK/dQ, softmax bwd).
    We use: FWD_BWD ≈ 3.5 × FWD.
    """
    return 3.5 * attn_flops_forward(cfg)


def make_inputs(batch: int, seq_len: int, heads: int, dim: int,
                 device: torch.device, dtype: torch.dtype,
                 seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic random Q, K, V and dout for a given shape."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=dtype, generator=g)
    k = torch.randn_like(q, memory_format=torch.contiguous_format)
    v = torch.randn_like(q, memory_format=torch.contiguous_format)
    dout = torch.randn_like(q, memory_format=torch.contiguous_format)
    return q, k, v, dout
