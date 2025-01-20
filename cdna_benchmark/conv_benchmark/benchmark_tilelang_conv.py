import torch
from tvm import tl
import tvm.tl.language as T
from tvm.tl.autotuner import *
import itertools
import argparse
from functools import partial

def get_configs():
    block_M = [32, 64, 128, 256]
    block_N = [32, 64, 128, 256]
    block_K = [32, 64]
    num_stages = [0, 1, 2]
    thread_num = [128, 256]
    
    _configs = list(itertools.product(block_M, block_N, block_K, num_stages, thread_num))

    configs = [
        {'block_M': c[0], 'block_N': c[1], 'block_K': c[2], 'num_stages': c[3], 'thread_num': c[4]}
        for c in _configs
    ]
    return configs

def convolution(N, C, H, W, F, K, S, D, P):
    KH, KW = K, K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1



    dtype = "float16"
    accum_dtype = "float"
    out_dtype = "float16"
    
    def ref_program(A, B):
        stride, padding, dilation = S, P, D
        A = A.permute(0, 3, 1, 2)  # N, H, W, C -> N, C, H, W
        B = B.permute(0, 3, 1, 2)  # F, H, W, C -> F, C, H, W
        C = torch.conv2d(A, B, stride=stride, padding=padding, dilation=dilation)
        C = C.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
        C = C.to(getattr(torch, out_dtype))
        return C

    k_pack = 2
    coalesced_width = None
    @autotune(configs=get_configs(), keys=['block_M', 'block_N', 'block_K', 'num_stages', 'thread_num'], warmup=10, rep=10)
    @jit(out_idx=[2], supply_type=tl.TensorSupplyType.Integer, ref_prog=ref_program, skip_check=True, profiler="tvm", target="hip")
    def kernel(block_M=None, block_N=None, block_K=None, num_stages=None, thread_num=None):

        @T.prim_func
        def main(
            data: T.Buffer((N, H, W, C), dtype),
            kernel: T.Buffer((F, KH, KW, C), dtype),
            out: T.Buffer((N, OH, OW, F), out_dtype),
        ):
            with T.Kernel(T.ceildiv(F, block_N), T.ceildiv(N * OH * OW, block_M), threads=thread_num) as (
                bx,
                by,
            ):
                data_shared = T.alloc_shared((block_M, block_K), dtype)
                kernel_shared = T.alloc_shared((block_N, block_K), dtype)
                out_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                out_shared = T.alloc_shared((block_M, block_N), out_dtype)

                kernel_flat = T.Buffer((F, KH * KW * C), dtype, kernel.data)
                out_flat = T.Buffer((N * OH * OW, F), out_dtype, out.data)

         
                T.clear(out_local)
                for k_iter in T.Pipelined(T.ceildiv(KH * KW * C, block_K), num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K, coalesced_width=coalesced_width):
                        k = k_iter * block_K + j
                        m = by * block_M + i
                        access_h = m % (OH * OW) // OW * S + k // (KW * C) * D - P
                        access_w = m % OW * S + k // C % KW * D - P
                        in_bound = (
                            (access_h >= 0)
                            and (access_w >= 0)
                            and (access_h < H)
                            and (access_w < W)
                        )
                        data_shared[i, j] = T.if_then_else(
                            in_bound, data[m // (OH * OW), access_h, access_w, k % C], 0
                        )
                    T.copy(kernel_flat[bx * block_N, k_iter * block_K], kernel_shared, coalesced_width=coalesced_width)
                    T.gemm(data_shared, kernel_shared, out_local, transpose_B=True, k_pack=k_pack)
                T.copy(out_local, out_shared)
                T.copy(out_shared, out_flat[by * block_M, bx * block_N])

        return main
    return kernel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=128, help='n')
    parser.add_argument('--c', type=int, default=128, help='c')
    parser.add_argument('--h', type=int, default=64, help='h')
    parser.add_argument('--w', type=int, default=64, help='w')
    parser.add_argument('--f', type=int, default=128, help='f')
    parser.add_argument('--k', type=int, default=3, help='k')
    parser.add_argument('--s', type=int, default=1, help='s')
    parser.add_argument('--d', type=int, default=1, help='d')
    parser.add_argument('--p', type=int, default=1, help='p')
    args = parser.parse_args()
    N, C, H, W, F, K, S, D, P = args.n, args.c, args.h, args.w, args.f, args.k, args.s, args.d, args.p
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
    total_flops = 2 * N * C * OH * OW * F * K * K
    best_latency, best_config, ref_latency = convolution(N, C, H, W, F, K, S, D, P)
    print(f"Best latency: {best_latency}")
    print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
    print(f"Best config: {best_config}")
