#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mha_torch.py

Benchmark for Multi-Head Attention using PyTorch native SDPA.
The goal is to invoke the most efficient kernel backend on supported GPUs,
and provide forward, backward, and profile interfaces similar to FA3,
for easy integration and comparison.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from contextlib import nullcontext

from mha_utils import MHAConfig, attn_flops_forward, attn_flops_forward_backward, make_inputs


# ================================ Utilities ================================ #

def _require_cuda() -> None:
    """Ensure CUDA is available and device is ready."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device detected. Please run in a CUDA-enabled environment.")


def _sdpa_fastest_ctx():
    """Return a context that prioritizes the fastest SDPA backend.
    Prefer FLASH_ATTENTION, then EFFICIENT_ATTENTION (PyTorch 2.5+).
    If new API is unavailable, fallback to old API; otherwise, use nullcontext.
    """
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        # New API: specify backend by priority
        return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
    except Exception:
        # Fallback for older versions (deprecated warning on new versions but usable)
        try:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True
            )
        except Exception:
            return nullcontext()


# ================================ Kernel Class ============================= #

class TorchMHAKernel:
    """Wrapper class for PyTorch SDPA-based MHA."""

    def __init__(self,
                 batch: int,
                 seq_len: int,
                 heads: int,
                 dim: int,
                 causal: bool,
                 dtype: torch.dtype = torch.float16,
                 device: str = "cuda:0") -> None:
        """Initialize kernel config and runtime environment.

        Args:
          batch: batch size B
          seq_len: sequence length S
          heads: number of attention heads H
          dim: per-head dimension Hd
          causal: whether to use causal mask
          dtype: torch.float16 or torch.bfloat16
          device: device string, e.g. "cuda:0"
        """
        _require_cuda()

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

    # ----------------------------- Core APIs ----------------------------- #

    @torch.inference_mode(False)
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                softmax_scale: Optional[float] = None) -> torch.Tensor:
        """Run forward computation.

        Note:
          Input must be [B, S, H, Hd]. No caching, caller manages input/output.

        Args:
          q: Query tensor [B, S, H, Hd]
          k: Key tensor [B, S, H, Hd]
          v: Value tensor [B, S, H, Hd]
          softmax_scale: softmax scaling factor. If not provided, use 1/sqrt(Hd)

        Returns:
          Output tensor [B, S, H, Hd]
        """
        cfg = self.cfg

        q = q.to(device=self.device, dtype=cfg.dtype)
        k = k.to(device=self.device, dtype=cfg.dtype)
        v = v.to(device=self.device, dtype=cfg.dtype)

        # PyTorch SDPA supports [B, H, S, D] or [B, S, H, D]
        # Here we use [B, H, S, D] to match backend requirements
        q_bhsd = q.transpose(1, 2)   # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)

        # is_causal controls lower-triangular mask. scale can override default scaling
        with _sdpa_fastest_ctx():
            out_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=cfg.causal,
                scale=softmax_scale
            )
        out = out_bhsd.transpose(1, 2).contiguous()  # Back to [B, S, H, D]
        return out

    @torch.inference_mode(False)
    def backward(self,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 v: torch.Tensor,
                 dout: Optional[torch.Tensor] = None,
                 softmax_scale: Optional[float] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one forward and backward pass, return dQ, dK, dV.

        Note:
          If dout is not provided, use MSE loss to construct scalar loss.

        Args:
          q: Query tensor [B, S, H, Hd]
          k: Key tensor [B, S, H, Hd]
          v: Value tensor [B, S, H, Hd]
          dout: upstream gradient [B, S, H, Hd]. If not provided, use scalar loss
          softmax_scale: softmax scaling factor

        Returns:
          Tuple (dQ, dK, dV), all shape [B, S, H, Hd]
        """
        cfg = self.cfg

        q = q.to(device=self.device, dtype=cfg.dtype).detach().requires_grad_(True)
        k = k.to(device=self.device, dtype=cfg.dtype).detach().requires_grad_(True)
        v = v.to(device=self.device, dtype=cfg.dtype).detach().requires_grad_(True)

        q_bhsd = q.transpose(1, 2)
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)

        with _sdpa_fastest_ctx():
            out_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=cfg.causal,
                scale=softmax_scale
            )
        out = out_bhsd.transpose(1, 2).contiguous()

        if dout is None:
            loss = out.float().pow(2).mean()
        else:
            if dout.shape != out.shape:
                raise ValueError(f"dout shape {dout.shape} does not match out shape {out.shape}")
            loss = (out * dout.to(out.dtype)).float().mean()

        loss.backward()

        d_q = q.grad.detach().clone()
        d_k = k.grad.detach().clone()
        d_v = v.grad.detach().clone()

        # Clear gradients to avoid accumulation on repeated calls
        for t in (q, k, v):
            if t.grad is not None:
                t.grad.zero_()

        return d_q, d_k, d_v

    # ----------------------------- Profiling ----------------------------- #

    @torch.inference_mode(False)
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
        """Benchmark interface, return latency and TFLOPs.

        Note:
          If q, k, v are not provided, generate fixed random tensors from cfg.
          CUDA events and synchronization are used for accurate timing.

        Args:
          q: Query [B, S, H, Hd]
          k: Key [B, S, H, Hd]
          v: Value [B, S, H, Hd]
          dout: upstream gradient [B, S, H, Hd]. Used if do_backward is True
          warmup: warmup rounds
          iters: timing rounds
          do_backward: whether to measure backward
          softmax_scale: softmax scaling factor

        Returns:
          Dictionary with keys:
            - fwd_latency_ms
            - fwd_tflops
            - bwd_latency_ms
            - bwd_tflops
        """
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

        def _fwd_once(q_, k_, v_) -> torch.Tensor:
            # Convert to [B, H, S, D]
            with _sdpa_fastest_ctx():
                out = F.scaled_dot_product_attention(
                    q_.transpose(1, 2), k_.transpose(1, 2), v_.transpose(1, 2),
                    attn_mask=None, dropout_p=0.0,
                    is_causal=cfg.causal, scale=softmax_scale
                )
            return out.transpose(1, 2).contiguous()

        # Warmup forward
        for _ in range(max(1, warmup)):
            out = _fwd_once(q_f, k_f, v_f)
            torch.cuda.synchronize()

        # Forward timing
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

        # Compute forward TFLOPs
        fwd_flops = attn_flops_forward(cfg)
        fwd_tflops = fwd_flops / (avg_fwd_ms * 1e-3) / 1e12

        # Optional backward
        avg_bwd_ms: Optional[float] = None
        bwd_tflops: Optional[float] = None
        if do_backward:
            # Warmup forward + backward
            for _ in range(max(1, warmup)):
                q_b = q_f.clone().detach().requires_grad_(True)
                k_b = k_f.clone().detach().requires_grad_(True)
                v_b = v_f.clone().detach().requires_grad_(True)
                out = _fwd_once(q_b, k_b, v_b)
                loss = (out * dout_f).float().mean()
                loss.backward()
                torch.cuda.synchronize()

            # Timing forward + backward
            total_fb_ms = 0.0
            for _ in range(iters):
                q_b = q_f.clone().detach().requires_grad_(True)
                k_b = k_f.clone().detach().requires_grad_(True)
                v_b = v_f.clone().detach().requires_grad_(True)

                starter.record()
                out = _fwd_once(q_b, k_b, v_b)
                loss = (out * dout_f).float().mean()
                loss.backward()
                ender.record()
                torch.cuda.synchronize()
                total_fb_ms += starter.elapsed_time(ender)

            avg_fb_ms = total_fb_ms / iters
            # Backward time is total time minus forward-only time
            avg_bwd_ms = max(0.0, avg_fb_ms - avg_fwd_ms)

            # Backward FLOPs estimate
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

def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch SDPA MHA benchmark.")
    parser.add_argument("--batch", type=int, required=True, help="Batch size B")
    parser.add_argument("--seq_len", type=int, required=True, help="Sequence length S")
    parser.add_argument("--heads", type=int, required=True, help="Number of attention heads H")
    parser.add_argument("--dim", type=int, required=True, help="Per-head dimension Hd")
    parser.add_argument("--causal", action="store_true", help="Whether to use causal mask")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="Computation precision")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup rounds")
    parser.add_argument("--iters", type=int, default=300, help="Timing rounds")
    parser.add_argument("--bwd", action="store_true", help="Whether to measure backward")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    kernel = TorchMHAKernel(
        batch=args.batch,
        seq_len=args.seq_len,
        heads=args.heads,
        dim=args.dim,
        causal=bool(args.causal),
        dtype=dtype,
        device="cuda:0",
    )

    # If no external input is provided, generate fixed random input here
    q, k, v, dout = make_inputs(kernel.cfg.batch, kernel.cfg.seq_len, kernel.cfg.heads, kernel.cfg.dim, torch.device("cuda:0"), kernel.cfg.dtype)
    metrics = kernel.profile(
        q=q, k=k, v=v,
        dout=dout,
        warmup=args.warmup,
        iters=args.iters,
        do_backward=bool(args.bwd),
    )

    d_model = args.heads * args.dim
    print(f"[Config] B={args.batch} S={args.seq_len} H={args.heads} Hd={args.dim} D={d_model} "
          f"Causal={bool(args.causal)} DType={dtype}")
    print(f"[Device] {torch.cuda.get_device_name(0)} | CC {torch.cuda.get_device_capability()}")

    print(f"[Result-FWD] avg_latency = {metrics['fwd_latency_ms']:.3f} ms | "
          f"throughput = {metrics['fwd_tflops']:.2f} TFLOPs")
    if args.bwd:
        bms = metrics["bwd_latency_ms"]
        bt = metrics["bwd_tflops"]
        bms_str = f"{bms:.3f} ms" if bms is not None else "N/A"
        bt_str = f"{bt:.2f} TFLOPs" if bt is not None else "N/A"
        print(f"[Result-BWD] avg_latency = {bms_str} | throughput = {bt_str}")

    print(f"[Info] softmax_scale default = 1/sqrt(Hd) = {1.0 / math.sqrt(args.dim):.6f}")


if __name__ == "__main__":
    main()
