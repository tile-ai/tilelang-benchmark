# export BNB_CUDA_VERSION=12.0
import os
os.environ["BNB_CUDA_VERSION"] = "120"
from bitsandbytes import functional as F

import torch
import time
import bitsandbytes as bnb

llm_shapes = [
    [1, 16384, 16384],
    [1, 43008, 14336],
    [1, 14336, 14336],
    [1, 57344, 14336],
    [1, 14336, 57344],
    [1, 9216, 9216],
    [1, 36864, 9216],
    [1, 9216, 36864],
    [1, 22016, 8192],
    [1, 8192, 22016],
    [1, 8192, 8192],
    [1, 28672, 8192],
    [1, 8192, 28672],
    [16384, 16384, 16384],
    [8192, 43008, 14336],
    [8192, 14336, 14336],
    [8192, 57344, 14336],
    [8192, 14336, 57344],
    [8192, 9216, 9216],
    [8192, 36864, 9216],
    [8192, 9216, 36864],
    [8192, 22016, 8192],
    [8192, 8192, 22016],
    [8192, 8192, 8192],
    [8192, 28672, 8192],
    [8192, 8192, 28672],
]

for M, N, K in llm_shapes:

    A = torch.randn(M, K, device="cuda").half()
    B = torch.empty(N, K, dtype=torch.float16, device="cuda")
    torch.nn.init.xavier_uniform_(B)

    # state -> [absmax, input_shape, A.dtype, blocksize, None, quant_type, datatype] absmax->scale
    B_nf4, state_nf4 = F.quantize_nf4(B, blocksize=128)

    out = bnb.matmul_4bit(A, B_nf4.t(), quant_state=state_nf4)

    # print(out)

    # torch gpu benchmark

    warm_up_iterations = 10
    iterations = 100

    for _ in range(warm_up_iterations):
        bnb.matmul_4bit(A, B_nf4.t(), quant_state=state_nf4)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iterations):
        bnb.matmul_4bit(A, B_nf4.t(), quant_state=state_nf4)
    torch.cuda.synchronize()
    end = time.time()

    average_time = (end - start) / iterations
    print(f"{M}_{N}_{K}: {average_time * 1e3:.4f} ms")
