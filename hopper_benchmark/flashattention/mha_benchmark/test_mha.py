#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_mha_compare.py

Read MHA shape parameters from a CSV, run forward/backward with both FA-3
and PyTorch SDPA implementations, verify numerical consistency, profile
latency and TFLOPs, print a console table (latency & TFLOPs only), and save
results to an .xlsx file.

Required CSV columns (case/underscore insensitive; see alias mapping):
  batch, seq_len, heads, dim, causal

Example:
batch,seq_len,heads,dim,causal
16,4096,64,128,TRUE
32,2048,32,128,false
"""

from __future__ import annotations
import argparse
import csv
import os
from typing import Dict, List, Tuple

import torch
import pandas as pd

from mha_utils import make_inputs

# Try importing the FA3 wrapper (prefer fa3_bench_mha.py, fallback to fa_bench_mha.py)
try:
    from mha_fa3 import FA3MHAKernel as FA3Kernel  # type: ignore
except Exception:
    raise ImportError(
        "Failed to import FA3MHAKernel. Ensure fa3_bench_mha.py or fa_bench_mha.py "
        "is present in the same directory and importable."
    )

# Import the PyTorch SDPA wrapper
try:
    from mha_pytorch import TorchMHAKernel  # type: ignore
except Exception:
    raise ImportError(
        "Failed to import TorchMHAKernel. Ensure torch_bench_mha.py is present in the "
        "same directory and importable."
    )

# ================================ Utilities ================================ #

def _sniff_open_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Detect delimiter and read CSV; return list of row dicts and original headers."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel  # fallback to comma
        reader = csv.DictReader(f, dialect=dialect)
        rows = [dict(r) for r in reader]
        headers = reader.fieldnames or []
    return rows, headers


def _norm_key(k: str) -> str:
    """Normalize column key: lowercase and strip spaces/underscores."""
    return k.strip().lower().replace(" ", "").replace("_", "")


def _parse_bool(v: str) -> bool:
    """Parse a boolean-like string to bool."""
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def _extract_shape(row: Dict[str, str]) -> Tuple[int, int, int, int, bool]:
    """Extract (batch, seq_len, heads, dim, causal) from a CSV row; supports aliases."""
    alias = {
        "batch": ["batch", "bs", "b"],
        "seq_len": ["seqlen", "seq_len", "s", "len", "length"],
        "heads": ["heads", "headnum", "head", "h", "head_num"],
        "dim": ["dim", "hd", "dmodelperhead", "perheaddim", "dimperhead"],
        "causal": ["causal", "iscasual", "iscausal", "mask"],
    }

    def pick(keys: List[str], required: bool = True) -> str:
        for k in keys:
            kk = _norm_key(k)
            for orig, val in row.items():
                if _norm_key(orig) == kk:
                    return val
        if required:
            raise KeyError(f"Missing required columns: {keys}")
        return ""

    batch = int(pick(alias["batch"]))
    seq_len = int(pick(alias["seq_len"]))
    heads = int(pick(alias["heads"]))
    dim = int(pick(alias["dim"]))
    causal = _parse_bool(pick(alias["causal"], required=False))
    return batch, seq_len, heads, dim, causal


def _max_abs_rel_diff(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> Tuple[float, float]:
    """Return (max absolute error, max relative error)."""
    a32 = a.detach().float()
    b32 = b.detach().float()
    diff = (a32 - b32).abs()
    max_abs = float(diff.max().item())
    denom = b32.abs().clamp_min(eps)
    rel = (diff / denom).abs()
    max_rel = float(rel.max().item())
    return max_abs, max_rel


# ================================ Main Logic ================================ #

def run_compare(input_params: str, warmup: int, iters: int, output: str,
                dtype: str = "fp16", device: str = "cuda:0",
                atol: float = 1e-2, rtol: float = 2e-2) -> None:
    """Run full comparison and profiling, then print and persist results."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device detected.")

    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype.lower()]
    dev = torch.device(device)

    output_path = output if output.lower().endswith(".xlsx") else f"{output}.xlsx"

    rows, _ = _sniff_open_csv(input_params)
    if not rows:
        raise ValueError("Input CSV is empty.")

    # Print device info
    dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
    print(f"[Device] {torch.cuda.get_device_name(dev_idx)} | CC {torch.cuda.get_device_capability(dev_idx)}")
    print(f"[Args] warmup={warmup} iters={iters} dtype={dt} device={device}")
    print()

    results_for_excel: List[Dict[str, object]] = []

    # Console table header (latency & TFLOPs only)
    header_print = (
        "Idx  B     S      H    Hd   Causal  "
        "FA3 Fwd(ms)  FA3 Fwd(TF)  FA3 Bwd(ms)  FA3 Bwd(TF)  "
        "Torch Fwd(ms)  Torch Fwd(TF)  Torch Bwd(ms)  Torch Bwd(TF)"
    )
    print(header_print)
    print("-" * len(header_print))

    for idx, row in enumerate(rows):
        # Parse shape parameters
        batch, seq_len, heads, dim, causal = _extract_shape(row)

        # Generate deterministic inputs for this configuration
        q, k, v, dout = make_inputs(batch, seq_len, heads, dim, dev, dt, seed=1234 + idx)

        # Instantiate both kernels
        fa3 = FA3Kernel(batch=batch, seq_len=seq_len, heads=heads, dim=dim,
                        causal=causal, dtype=dt, device=device)
        torch_mha = TorchMHAKernel(batch=batch, seq_len=seq_len, heads=heads, dim=dim,
                                   causal=causal, dtype=dt, device=device)

        # Forward pass
        out_fa3 = fa3.forward(q, k, v, softmax_scale=None)
        out_torch = torch_mha.forward(q, k, v, softmax_scale=None)

        # Consistency checks (not printed; still enforced)
        ok_fwd = torch.allclose(out_fa3, out_torch, atol=atol, rtol=rtol)
        if not ok_fwd:
            out_abs, out_rel = _max_abs_rel_diff(out_fa3, out_torch)
            raise AssertionError(
                f"[FWD Mismatch] idx={idx} B={batch} S={seq_len} H={heads} Hd={dim} "
                f"max_abs={out_abs:.4e} max_rel={out_rel:.4e}"
            )

        # Backward pass
        dQ_fa3, dK_fa3, dV_fa3 = fa3.backward(dout=dout, softmax_scale=None)
        dQ_t, dK_t, dV_t = torch_mha.backward(q, k, v, dout=dout, softmax_scale=None)

        ok_bwd = (
            torch.allclose(dQ_fa3, dQ_t, atol=atol, rtol=rtol)
            and torch.allclose(dK_fa3, dK_t, atol=atol, rtol=rtol)
            and torch.allclose(dV_fa3, dV_t, atol=atol, rtol=rtol)
        )
        if not ok_bwd:
            dq_abs, _ = _max_abs_rel_diff(dQ_fa3, dQ_t)
            dk_abs, _ = _max_abs_rel_diff(dK_fa3, dK_t)
            dv_abs, _ = _max_abs_rel_diff(dV_fa3, dV_t)
            raise AssertionError(
                f"[BWD Mismatch] idx={idx} B={batch} S={seq_len} H={heads} Hd={dim} "
                f"dQ_abs={dq_abs:.4e} dK_abs={dk_abs:.4e} dV_abs={dv_abs:.4e}"
            )

        # Profiling both sides (forward and backward)
        fa3_metrics = fa3.profile(warmup=warmup, iters=iters, do_backward=True)
        torch_metrics = torch_mha.profile(q=q, k=k, v=v, dout=dout,
                                          warmup=warmup, iters=iters, do_backward=True)

        rec = {
            "idx": idx,
            "batch": batch,
            "seq_len": seq_len,
            "heads": heads,
            "dim": dim,
            "causal": causal,
            "fa3_fwd_ms": round(float(fa3_metrics["fwd_latency_ms"]), 4),
            "fa3_fwd_tflops": round(float(fa3_metrics["fwd_tflops"]), 4),
            "fa3_bwd_ms": round(float(fa3_metrics["bwd_latency_ms"]), 4) if fa3_metrics["bwd_latency_ms"] is not None else None,
            "fa3_bwd_tflops": round(float(fa3_metrics["bwd_tflops"]), 4) if fa3_metrics["bwd_tflops"] is not None else None,
            "torch_fwd_ms": round(float(torch_metrics["fwd_latency_ms"]), 4),
            "torch_fwd_tflops": round(float(torch_metrics["fwd_tflops"]), 4),
            "torch_bwd_ms": round(float(torch_metrics["bwd_latency_ms"]), 4) if torch_metrics["bwd_latency_ms"] is not None else None,
            "torch_bwd_tflops": round(float(torch_metrics["bwd_tflops"]), 4) if torch_metrics["bwd_tflops"] is not None else None,
        }
        results_for_excel.append(rec)

        # Console print (latency & TFLOPs only)
        print(
            f"{idx:<4d}{batch:<6d}{seq_len:<7d}{heads:<5d}{dim:<5d}{str(causal):<8s}"
            f"{rec['fa3_fwd_ms']!s:<12s}{rec['fa3_fwd_tflops']!s:<13s}"
            f"{rec['fa3_bwd_ms']!s:<12s}{rec['fa3_bwd_tflops']!s:<13s}"
            f"{rec['torch_fwd_ms']!s:<14s}{rec['torch_fwd_tflops']!s:<14s}"
            f"{rec['torch_bwd_ms']!s:<14s}{rec['torch_bwd_tflops']!s:<13s}"
        )

    # Persist results to XLSX (latency & TFLOPs table)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    df = pd.DataFrame(results_for_excel, columns=[
        "idx", "batch", "seq_len", "heads", "dim", "causal",
        "fa3_fwd_ms", "fa3_fwd_tflops", "fa3_bwd_ms", "fa3_bwd_tflops",
        "torch_fwd_ms", "torch_fwd_tflops", "torch_bwd_ms", "torch_bwd_tflops",
    ])
    df.to_excel(output_path, index=False)
    print(f"[Saved] Results written to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FA-3 vs PyTorch MHA correctness & performance test")
    parser.add_argument("--input_params", type=str, required=True, help="Path to input shapes CSV")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Measured iterations")
    parser.add_argument("--output", type=str, required=True, help="Output name ('.xlsx' appended if missing)")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="Computation dtype")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device, e.g., cuda:0")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for allclose")
    parser.add_argument("--rtol", type=float, default=2e-2, help="Relative tolerance for allclose")
    args = parser.parse_args()

    run_compare(
        input_params=args.input_params,
        warmup=args.warmup,
        iters=args.iters,
        output=args.output,
        dtype=args.dtype,
        device=args.device,
        atol=args.atol,
        rtol=args.rtol,
    )


if __name__ == "__main__":
    main()
