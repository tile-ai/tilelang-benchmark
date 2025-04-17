# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import argparse
import itertools
import logging

import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import autotune, jit
from tvm import tir
# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _tir_u8_to_u2_to_u8(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    assert nbit == 2
    assert dtype == "int8"
    assert val.dtype == "uint8"
    
    return (val >> (pos * tir.const(nbit, "uint8"))) & tir.const((1 << nbit) - 1, "uint8")



def get_configs(M, N, K, with_roller=False):
    """
    Generate a list of configuration dictionaries that will be used for tuning.
    
    Parameters
    ----------
    with_roller : bool
        Whether to enable bitblas roller to deduce search spaces

    Returns
    -------
    list of dict
        Each configuration dict includes various block sizes, pipeline stages,
        thread numbers, and other parameters to explore during autotuning.
    """
    if with_roller:
        from tilelang.carver.template import MatmulTemplate
        from tilelang.carver.arch import CUDA
        from tilelang.carver.roller.rasterization import NoRasterization
        arch = CUDA("cuda")
        topk = 10

        carve_template = MatmulTemplate(
            M=M,
            N=N,
            K=K,
            in_dtype="float16",
            out_dtype="float16",
            accum_dtype="float",
        ).with_arch(arch)

        func = carve_template.equivalent_function()
        assert func is not None, "Function is None"

        roller_hints = carve_template.recommend_hints(topk=topk)

        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")

        configs = []
        for hint in roller_hints:
            config = {}
            block_m, block_n = hint.block
            warp_m, warp_n = hint.warp
            # block_rows, block_cols represents warp partitioning
            block_rows, block_cols = block_m // warp_m, block_n // warp_n
            config["block_M"] = block_m
            config["block_N"] = block_n
            config["block_K"] = hint.rstep[0]
            config["num_stages"] = hint.pipeline_stage
            config["thread_num"] = block_rows * block_cols * 32
            config["policy"] = T.GemmWarpPolicy.from_warp_partition(block_rows, block_cols)
            config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization
            configs.append(config)
        for config in configs:
            print(config)
    else:

        block_M = [64, 128, 256]
        block_N = [64, 128, 256]
        block_K = [32, 64, 128, 256]
        num_stages = [0, 1, 2, 3]
        thread_num = [128, 256]
        policy = [T.GemmWarpPolicy.Square]
        enable_rasterization = [True, False]
        
        # block_M = [64]
        # block_N = [64]
        # block_K = [128]
        # num_stages = [2]
        # thread_num = [128]
        # policy = [T.GemmWarpPolicy.Square]
        # enable_rasterization = [True]

        _configs = list(
            itertools.product(
                block_M,
                block_N,
                block_K,
                num_stages,
                thread_num,
                policy,
                enable_rasterization,
            ))

        configs = [
            {
                "block_M": c[0],
                "block_N": c[1],
                "block_K": c[2],
                "num_stages": c[3],
                "thread_num": c[4],
                "policy": c[5],
                "enable_rasteration": c[6],  # keep param name for backward-compat
            } for c in _configs
        ]
    return configs


def matmul(M, N, K, with_roller):
    """
    Create an autotuned matrix multiplication kernel for matrices of shape:
      - A: (M, K)
      - B: (N, K)
      - C: (M, N)

    Parameters
    ----------
    M : int
        The dimension M of the matrix multiplication.
    N : int
        The dimension N of the matrix multiplication.
    K : int
        The dimension K of the matrix multiplication.

    Returns
    -------
    (best_latency, best_config, ref_latency)
        best_latency : float
            The best latency found among the tuned configurations.
        best_config : dict
            The parameter configuration that yielded best_latency.
        ref_latency : float
            The baseline latency of the reference program (for computing speedup).
    """

    # Decorate the kernel with autotune & jit, specifying:
    #  - Tuning config list
    #  - Profiling keys
    #  - Warmup and repetition counts for better measurement
    #  - A reference program for correctness verification
    #  - The "tvm" profiler backend
    #  - HIP as the compilation target (modify as needed for your hardware)

    @autotune(
        configs=get_configs(M, N, K, with_roller),
        warmup=3,
        rep=20,
    )
    @jit(
        out_idx=[2],
        supply_type=tl.TensorSupplyType.Integer,
        target="auto",
    )
    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        thread_num=None,
        policy=None,
        enable_rasteration=None,
    ):
        """
        The actual kernel to compute C = A @ B^T.

        Parameters
        ----------
        block_M : int
            Block size in M dimension.
        block_N : int
            Block size in N dimension.
        block_K : int
            Block size in K dimension.
        num_stages : int
            Number of pipelined stages (for asynchronous load).
        thread_num : int
            Number of threads to use per block.
        enable_rasteration : bool
            Whether to enable rasterization (swizzling) optimization.
        k_pack : int
            K dimension packing factor to improve memory coalescing.

        Returns
        -------
        Function
            A TVM Tensor Language function (T.prim_func) that computes matmul.
        """
        # Use half-precision for input data to reduce memory bandwidth,
        # accumulate in float for better numerical accuracy
        dtype = "int8"
        accum_dtype = "int32"
        num_bits = 2
        num_elems_per_byte = 8 // num_bits
        storage_dtype = "uint8"
        B_shape = (N, K // num_elems_per_byte)
        B_shared_shape = (block_N, block_K // num_elems_per_byte)
        B_dequantize_shared_shape = (block_N, block_K)

        @T.prim_func
        def main(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor(B_shape, storage_dtype),
                C: T.Tensor((N, M), dtype),
        ):
            """
            The compiled TVM function for block-level matrix multiplication.

            - We divide the entire (M, N) domain into blocks of shape
              (block_M, block_N).
            - Each block has its own allocated shared memory for sub-blocks
              of A and B.
            - The partial results go into C_local, and then we copy them back
              to global memory C.
            """
            # Bind x-dimension to block index in N,
            #     y-dimension to block index in M.
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):

                # Allocate shared memory for A sub-block of shape (block_M, block_K)
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                # Allocate shared memory for B sub-block of shape (block_N, block_K)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
                B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
                B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, dtype)
                B_dequantize_prev_local = T.alloc_fragment(B_dequantize_shared_shape, dtype)
                # Allocate a local fragment for intermediate accumulation
                C_local = T.alloc_fragment((block_N, block_M), accum_dtype)
                # Allocate a shared memory for C sub-block of shape (block_M, block_N)
                C_shared = T.alloc_shared((block_N, block_M), dtype)

                # Enable (or disable) swizzling optimization
                T.use_swizzle(panel_size=10, enable=enable_rasteration)

                # Clear out the accumulation buffer
                T.clear(C_local)

                # Loop over sub-blocks in K dimension, pipelined by num_stages
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    # Load a sub-block of A from global memory into A_shared
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    # Load a sub-block of B from global memory into B_shared
                    T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)

                    T.copy(B_shared, B_local)
                    for i, j in T.Parallel(block_N, block_K):
                        B_dequantize_local[i, j] = _tir_u8_to_u2_to_u8(
                            num_bits,
                            B_local[i, j // num_elems_per_byte],
                            j % num_elems_per_byte,
                            dtype=dtype,
                        )
                    T.copy(B_dequantize_local, B_dequantize_prev_local)
                    T.gemm(
                        B_dequantize_prev_local,
                        A_shared,
                        C_local,
                        transpose_B=True,
                        policy=policy,
                    )
                # Write back the results from C_local to the global memory C
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[bx * block_N, by * block_M])

        return main

    return kernel()


if __name__ == "__main__":
    # Parse command-line arguments for matrix dimensions
    parser = argparse.ArgumentParser(description="Autotuned MatMul Benchmark")
    parser.add_argument("--m", type=int, default=16384, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=16384, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=16384, help="Matrix dimension K")
    parser.add_argument(
        "--with_roller",
        action="store_true",
        help="Whether to enable BitBLAS roller for search space",
    )
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k
    with_roller = args.with_roller

    # Compute total floating-point operations to measure throughput
    total_flops = 2 * M * N * K

    # matmul(...) returns (best_latency, best_config, ref_latency)
    best_result = matmul(M, N, K, with_roller)
    best_latency = best_result.latency
    best_config = best_result.config
    ref_latency = best_result.ref_latency

    # Print out the benchmark results
    print(f"Best latency (s): {best_latency}")
    print(f"Best TFlops: {total_flops / best_latency * 1e-9:.3f}")
    print(f"Best config: {best_config}")
