#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mha_fa3.py

Benchmark Multi-Head Attention (FlashAttention-3 core) with random inputs,
refactored into a class that exposes forward/backward and a profile() API.

- Inputs: batch, seq_len, heads, dim_per_head, causal
- Generates random Q/K/V on CUDA (or accepts user-provided tensors).
- Measures latency (ms) and TFLOPs of the FA-3 kernel forward and backward.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Optional, Tuple

import torch

from mha_utils import MHAConfig, attn_flops_forward, attn_flops_forward_backward, make_inputs


# ================================ Utilities ================================ #


def _require_fa3() -> None:
    """Ensure FA-3 python interface exists and device is Hopper (SM >= 90)."""
    try:
        import flash_attn_interface  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "flash_attn_interface not found. Install flash-attn>=2.8.3 and build FA-3."
        ) from e

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. FA-3 requires H100/H800.")

    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(
            f"Compute capability {major}.x found. FA-3 requires Hopper (SM 90+)."
        )


# ================================ Kernel Class ============================= #


class FA3MHAKernel:
    """FlashAttention-3 MHA kernel wrapper with fwd/bwd and profiling."""

    def __init__(self,
                 batch: int,
                 seq_len: int,
                 heads: int,
                 dim: int,
                 causal: bool,
                 dtype: torch.dtype = torch.float16,
                 device: str = "cuda:0") -> None:
        """Initialize the kernel configuration and runtime context.

        Args:
          batch: Batch size (B).
          seq_len: Sequence length (S).
          heads: Number of heads (H).
          dim: Per-head dimension (Hd).
          causal: Use causal mask if True.
          dtype: torch.float16 or torch.bfloat16.
          device: CUDA device string.
        """
        _require_fa3()

        self.cfg = MHAConfig(
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            dim=dim,
            causal=causal,
            dtype=dtype,
            dropout_p=0.0,
        )
        self.device = torch.device(device)

        # Cache for last tensors used in forward to support backward().
        self._last_q: Optional[torch.Tensor] = None
        self._last_k: Optional[torch.Tensor] = None
        self._last_v: Optional[torch.Tensor] = None
        self._last_out: Optional[torch.Tensor] = None

    # ----------------------------- Core APIs ----------------------------- #

    def forward(self,
                q: Optional[torch.Tensor] = None,
                k: Optional[torch.Tensor] = None,
                v: Optional[torch.Tensor] = None,
                softmax_scale: Optional[float] = None) -> torch.Tensor:
        """Compute forward attention output.

        Args:
          q: Query tensor [B, S, H, Hd]. If None, random tensors are generated.
          k: Key tensor [B, S, H, Hd]. If None, random tensors are generated.
          v: Value tensor [B, S, H, Hd]. If None, random tensors are generated.
          softmax_scale: Scale for softmax. If None, defaults to 1/sqrt(Hd).

        Returns:
          Output tensor [B, S, H, Hd].
        """
        import flash_attn_interface as fai

        if q is None or k is None or v is None:
            q, k, v, _ = make_inputs(self.cfg.batch, self.cfg.seq_len, self.cfg.heads, self.cfg.dim, self.device, self.cfg.dtype)

        # Ensure dtype/device and disable grad for pure inference.
        q = q.to(device=self.device, dtype=self.cfg.dtype)
        k = k.to(device=self.device, dtype=self.cfg.dtype)
        v = v.to(device=self.device, dtype=self.cfg.dtype)
        q.requires_grad_(False)
        k.requires_grad_(False)
        v.requires_grad_(False)

        out = fai.flash_attn_func(q, k, v,
                                  softmax_scale=softmax_scale,
                                  causal=self.cfg.causal)
        # Cache last I/O for backward() convenience.
        self._last_q, self._last_k, self._last_v = q, k, v
        self._last_out = out
        return out

    def backward(self,
                 dout: Optional[torch.Tensor] = None,
                 softmax_scale: Optional[float] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a forward pass (if needed) and compute gradients.

        Args:
          dout: Upstream gradient of shape [B, S, H, Hd]. If None, uses mean-square loss.
          softmax_scale: Scale for softmax. If None, defaults to 1/sqrt(Hd).

        Returns:
          Tuple of gradients (dQ, dK, dV) matching [B, S, H, Hd].
        """
        import flash_attn_interface as fai

        # Prepare inputs: either from the last forward or freshly generated.
        if self._last_q is None or self._last_k is None or self._last_v is None:
            q, k, v, dout = make_inputs(self.cfg.batch, self.cfg.seq_len, self.cfg.heads, self.cfg.dim, self.device, self.cfg.dtype)
        else:
            q, k, v = self._last_q.clone(), self._last_k.clone(), self._last_v.clone()

        # Enable gradients.
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        out = fai.flash_attn_func(q, k, v,
                                  softmax_scale=softmax_scale,
                                  causal=self.cfg.causal)

        if dout is None:
            # Use a simple scalar loss to exercise backward path.
            loss = out.float().pow(2).mean()
        else:
            # d(loss)/d(out) = dout; contract to a scalar loss via dot-product.
            if dout.shape != out.shape:
                raise ValueError(f"dout shape {dout.shape} != out shape {out.shape}")
            loss = (out * dout.to(out.dtype)).float().mean()

        loss.backward()

        d_q = q.grad.detach().clone()
        d_k = k.grad.detach().clone()
        d_v = v.grad.detach().clone()

        # Reset grads to avoid accumulation if this method is called repeatedly.
        for t in (q, k, v):
            if t.grad is not None:
                t.grad.zero_()

        return d_q, d_k, d_v

    # ----------------------------- Profiling ----------------------------- #

    def profile(self,
                q: Optional[torch.Tensor] = None,
                k: Optional[torch.Tensor] = None,
                v: Optional[torch.Tensor] = None,
                dout: Optional[torch.Tensor] = None,
                warmup: int = 50,
                iters: int = 300,
                do_backward: bool = False,
                softmax_scale: Optional[float] = None
                ) -> Dict[str, Optional[float]]:
        """Profile forward and optional backward.

        Args:
          warmup: Warmup iterations before measuring.
          iters: Measured iterations.
          do_backward: If True, also measure forward+backward.
          softmax_scale: Optional softmax scale.

        Returns:
          Dict with:
            - fwd_latency_ms
            - fwd_tflops
            - bwd_latency_ms (None if do_backward=False)
            - bwd_tflops     (None if do_backward=False)
        """
        import flash_attn_interface as fai

        cfg = self.cfg
        device = self.device

        if q is None or k is None or v is None or dout is None:
            q, k, v, dout = make_inputs(cfg.batch, cfg.seq_len, cfg.heads, cfg.dim, device, cfg.dtype)
        else:
            q = q.to(device=device, dtype=cfg.dtype)
            k = k.to(device=device, dtype=cfg.dtype)
            v = v.to(device=device, dtype=cfg.dtype)
            dout = dout.to(device=device, dtype=cfg.dtype)

        # Fix input for fair timing
        q_f = q.detach()
        k_f = k.detach()
        v_f = v.detach()
        dout_f = dout.detach()

        # Local helper to run a single forward.
        def _fwd_once(q_, k_, v_) -> torch.Tensor:
            return fai.flash_attn_func(q_, k_, v_,
                                       softmax_scale=softmax_scale,
                                       causal=cfg.causal)

        # Warmup forward-only.
        for _ in range(max(1, warmup)):
            out = _fwd_once(q_f, k_f, v_f)
            torch.cuda.synchronize()

        # Measure forward-only.
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        total_fwd_ms = 0.0
        for _ in range(iters):
            starter.record()
            out = _fwd_once(q_f, k_f, v_f)
            ender.record()
            torch.cuda.synchronize()
            total_fwd_ms += starter.elapsed_time(ender)
        avg_fwd_ms = total_fwd_ms / iters

        # Compute forward TFLOPs.
        fwd_flops = attn_flops_forward(cfg)
        fwd_tflops = fwd_flops / (avg_fwd_ms * 1e-3) / 1e12

        # Optionally measure backward (time of fwd+bwd minus fwd-only).
        avg_bwd_ms: Optional[float] = None
        bwd_tflops: Optional[float] = None
        if do_backward:
            # Recreate inputs with grad enabled to isolate the bwd loop clearly.
            q_b, k_b, v_b = q.clone().detach().requires_grad_(True), \
                            k.clone().detach().requires_grad_(True), \
                            v.clone().detach().requires_grad_(True)

            # Warmup fwd+bwd.
            for _ in range(max(1, warmup)):
                out = _fwd_once(q_b, k_b, v_b)
                loss = (out * dout).float().mean()
                loss.backward()
                torch.cuda.synchronize()

            # Measure fwd+bwd.
            total_fb_ms = 0.0
            for _ in range(iters):
                starter.record()
                out = _fwd_once(q_b, k_b, v_b)
                loss = (out * dout).float().mean()
                loss.backward()
                ender.record()
                torch.cuda.synchronize()
                total_fb_ms += starter.elapsed_time(ender)
                for t in (q_b, k_b, v_b):
                    if t.grad is not None:
                        t.grad.zero_()

            avg_fb_ms = total_fb_ms / iters
            # Backward time is approximated by subtracting forward-only time.
            avg_bwd_ms = max(0.0, avg_fb_ms - avg_fwd_ms)

            # Use the 3×FWD model to estimate total FLOPs, then isolate BWD FLOPs.
            fwd_bwd_flops = attn_flops_forward_backward(cfg)
            bwd_flops = max(0.0, fwd_bwd_flops - fwd_flops)
            bwd_tflops = bwd_flops / (avg_bwd_ms * 1e-3) / 1e12 if avg_bwd_ms > 0 else None

        return {
            "fwd_latency_ms": float(avg_fwd_ms),
            "fwd_tflops": float(fwd_tflops),
            "bwd_latency_ms": None if avg_bwd_ms is None else float(avg_bwd_ms),
            "bwd_tflops": None if bwd_tflops is None else float(bwd_tflops),
        }

# =================================== CLI =================================== #


def run_fa3_benchmark(
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
    causal: bool,
    *,
    dtype: str | torch.dtype = "fp16",
    warmup: int = 50,
    iters: int = 300,
    bwd: bool = False,
    device: str = "cuda:0",
    verbose: bool = True,
) -> Dict[str, Optional[float]]:
    """Run the FA-3 MHA micro-benchmark programmatically.

    This function wraps the CLI logic in `main()` so it can be imported and
    called from other scripts or notebooks.

    Args:
      batch: Batch size (B).
      seq_len: Sequence length (S).
      heads: Number of heads (H).
      dim: Per-head dimension (Hd).
      causal: Whether to apply causal masking.
      dtype: "fp16"/"bf16" or a torch.dtype (torch.float16/torch.bfloat16).
      warmup: Number of warmup iterations before timing.
      iters: Number of measured iterations.
      bwd: If True, also time forward+backward and report backward-only latency.
      device: CUDA device, e.g., "cuda:0".
      verbose: If True, print a human-readable summary.

    Returns:
      A dictionary with keys:
        - "fwd_latency_ms": Average forward latency in milliseconds.
        - "fwd_tflops": Forward throughput in TFLOPs.
        - "bwd_latency_ms": Backward latency (None if bwd=False).
        - "bwd_tflops": Backward throughput (None if bwd=False).
    """
    # Normalize dtype input.
    if isinstance(dtype, str):
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype.lower()]

    # Build kernel and profile.
    kernel = FA3MHAKernel(
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        dim=dim,
        causal=bool(causal),
        dtype=dtype,          # type: ignore[arg-type]
        device=device
    )

    # Device info (respect an explicit device index if provided).
    dev = torch.device(device)
    dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
    dev_name = torch.cuda.get_device_name(dev_idx)
    cc_major, cc_minor = torch.cuda.get_device_capability(dev_idx)

    d_model = heads * dim
    q, k, v, dout = make_inputs(kernel.cfg.batch, kernel.cfg.seq_len, kernel.cfg.heads, kernel.cfg.dim, torch.device("cuda:0"), kernel.cfg.dtype)
    metrics = kernel.profile(
        q=q, k=k, v=v,
        dout=dout,
        warmup=warmup,
        iters=iters,
        do_backward=bool(bwd),
    )

    if verbose:
        print(
            f"[Config] B={batch} S={seq_len} H={heads} Hd={dim} D={d_model} "
            f"Causal={bool(causal)} DType={dtype}"
        )
        print(f"[Device] {dev_name} | CC ({cc_major}, {cc_minor})")

        print(
            f"[Result-FWD] avg_latency = {metrics['fwd_latency_ms']:.3f} ms | "
            f"throughput = {metrics['fwd_tflops']:.2f} TFLOPs"
        )
        if bwd:
            bms = metrics["bwd_latency_ms"]
            bt = metrics["bwd_tflops"]
            bms_str = f"{bms:.3f} ms" if bms is not None else "N/A"
            bt_str = f"{bt:.2f} TFLOPs" if bt is not None else "N/A"
            print(f"[Result-BWD] avg_latency = {bms_str} | throughput = {bt_str}")

        print(f"[Info] softmax_scale default = 1/sqrt(Hd) = {1.0 / math.sqrt(dim):.6f}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlashAttention-3 MHA class-based micro-benchmark."
    )
    parser.add_argument("--batch", type=int, required=True, help="Batch size (B).")
    parser.add_argument("--seq_len", type=int, required=True, help="Sequence length (S).")
    parser.add_argument("--heads", type=int, required=True, help="Number of heads (H).")
    parser.add_argument("--dim", type=int, required=True, help="Per-head dimension (Hd).")
    parser.add_argument("--causal", action="store_true", help="Enable causal mask.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Computation dtype.",
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=300, help="Measured iterations.")
    parser.add_argument("--bwd", action="store_true", help="Include backward timing.")
    args = parser.parse_args()

    # Delegate to the programmatic API (prints by default).
    run_fa3_benchmark(
        batch=args.batch,
        seq_len=args.seq_len,
        heads=args.heads,
        dim=args.dim,
        causal=bool(args.causal),
        dtype=args.dtype,
        warmup=args.warmup,
        iters=args.iters,
        bwd=bool(args.bwd),
        device="cuda:0",
        verbose=True,
    )


if __name__ == "__main__":
    main()
