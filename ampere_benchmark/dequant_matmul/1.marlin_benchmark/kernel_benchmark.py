import sys

import numpy as np
import torch
import marlin

import time

groupsize = -1
shapes = [
    (1, 16384, 16384),
    (1, 43008, 14336),
    (1, 14336, 14336),
    (1, 57344, 14336),
    (1, 14336, 57344),
    (1, 9216, 9216),
    (1, 36864, 9216),
    (1, 9216, 36864),
    (1, 22016, 8192),
    (1, 8192, 22016),
    (1, 8192, 8192),
    (1, 28672, 8192),
    (1, 8192, 28672),
    (16384, 16384, 16384),
    (8192, 43008, 14336),
    (8192, 14336, 14336),
    (8192, 57344, 14336),
    (8192, 14336, 57344),
    (8192, 9216, 9216),
    (8192, 36864, 9216),
    (8192, 9216, 36864),
    (8192, 22016, 8192),
    (8192, 8192, 22016),
    (8192, 8192, 8192),
    (8192, 28672, 8192),
    (8192, 8192, 28672),
]


def benchmark(f, warmup=1, iter=10):
    for i in range(warmup + iter):
        f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.0)
    return res


def benchmark_quant(A, B, C, s, thread_k, thread_n, sms):
    workspace = torch.zeros(C.shape[1] // 128 * 16, device=torch.device("cuda:0"))
    res = benchmark(lambda: marlin.mul(A, B, C, s, workspace, thread_k, thread_n, sms))
    return {
        "s": res,
        "TFLOP/s": 2 * A.numel() * C.shape[1] / res / 10**12,
        "GB/s": (2 * A.numel() + 4 * B.numel() + 2 * C.numel() + 2 * s.numel())
        / res
        / 10**9,
    }


def get_problem(m, n, k, groupsize=-1):
    if groupsize == -1:
        groupsize = k
    dev = torch.device("cuda:0")
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B = torch.randint(low=-(2**31), high=2**31, size=(k * n // 8,), device=dev)
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    s = torch.zeros((k // groupsize, n), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return A, B, C, B_ref, s

# Pass the SM count for known GPUs to avoid the kernel having to query this information (this is very minor)
gpu = torch.cuda.get_device_name(0)
if "A100" in gpu:
    SMS = 108
elif "A10" in gpu:
    SMS = 72
elif "3090" in gpu:
    SMS = 82
elif "A6000" in gpu:
    SMS = 84
else:
    SMS = -1

for m, n, k in shapes:
    A, B, C, B_ref, s = get_problem(m, n, k, groupsize)
    res_q = benchmark_quant(A, B, C, s, -1, -1, SMS)
    print(f"{m}_{n}_{k}: {res_q['s'] * 1e3 :.4f} ms")
