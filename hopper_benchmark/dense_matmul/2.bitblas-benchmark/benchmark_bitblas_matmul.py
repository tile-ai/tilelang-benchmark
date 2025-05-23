# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from bitblas.utils.target_detector import auto_detect_nvidia_target
from bitblas import Matmul, MatmulConfig
import argparse
from bitblas.base.roller.policy import TensorCorePolicy, DefaultPolicy
from bitblas.base.arch import CUDA
from bitblas.gpu.matmul_analysis import get_tensorized_func_and_tags
from bitblas.base.utils import apply_and_build
from bitblas import set_log_level
set_log_level("DEBUG")
# Initialize the parser
parser = argparse.ArgumentParser(
    description="Benchmark BitBLAS int4 on a specific target."
)

# Add arguments to the parser

parser.add_argument(
    "--target",
    type=str,
    default=auto_detect_nvidia_target(),
    help="Specify the target device for benchmarking.",
)

parser.add_argument(
    "--M",
    type=int,
    default=16384,
    help="Number of rows in matrix A.",
)

parser.add_argument(
    "--N",
    type=int,
    default=16384,
    help="Number of rows in matrix A.",
)

parser.add_argument(
    "--K",
    type=int,
    default=16384,
    help="Number of rows in matrix A.",
)

parser.add_argument(
    "--A_dtype",
    type=str,
    default="float16",
    choices=[
        "float16",
        "float32",
        "float64",
        "int32",
        "int8",
    ],  # Assuming these are the valid choices
    help="Data type of activation A.",
)
parser.add_argument(
    "--W_dtype",
    type=str,
    default="float16",
    choices=[
        "float16",
        "float32",
        "float64",
        "int32",
        "int8",
        "int4",
        "int2",
        "int1",
        "nf4",
        "fp4_e2m1",
    ],  # Assuming these are the valid choices
    help="Data type of weight W.",
)
parser.add_argument(
    "--accum_dtype",
    type=str,
    default="float16",
    choices=["float16", "int32"],  # Assuming these are the valid choices
    help="Data type for accumulation.",
)
parser.add_argument(
    "--out_dtype",
    type=str,
    default="float16",
    choices=[
        "float16",
        "float32",
        "int32",
        "int8",
    ],  # Assuming these are the valid choices
    help="Data type for output.",
)
parser.add_argument(
    "--layout",
    type=str,
    default="nt",
    choices=["nt", "nn"],  # Assuming these are the valid choices
    help="Matrix layout, 'nt' for non-transpose A and transpose W.",
)
parser.add_argument(
    "--with_bias", action="store_true", help="Include bias in the benchmark."
)
parser.add_argument(
    "--with_scaling",
    action="store_true",
    help="Include scaling factor in the quantization.",
)
parser.add_argument(
    "--group_size", type=int, default=None, help="Group size for grouped quantization."
)
parser.add_argument(
    "--with_zeros", action="store_true", help="Include zeros in the quantization."
)
parser.add_argument(
    "--zeros_mode",
    type=str,
    default=None,
    choices=[
        "original",
        "rescale",
        "quantized",
    ],  # Replace with actual modes if applicable
    help="Specify the mode for calculating zeros.",
)

# Parse the arguments
args = parser.parse_args()

# Assign arguments to variables
target = args.target
M, N, K = args.M, args.N, args.K
group_size = args.group_size
A_dtype = args.A_dtype
W_dtype = args.W_dtype
accum_dtype = args.accum_dtype
out_dtype = args.out_dtype
layout = args.layout
with_bias = args.with_bias
group_size = args.group_size
with_scaling = args.with_scaling
with_zeros = args.with_zeros
zeros_mode = args.zeros_mode

test_shapes = [
    # square test
    (
        MatmulConfig,
        Matmul,
        (
            M,
            N,
            K,
            A_dtype,
            W_dtype,
            out_dtype,
            accum_dtype,
            layout,
            with_bias,
            group_size,
            with_scaling,
            with_zeros,
            zeros_mode,
        ),
    ),
]

benchmark_sets = []
benchmark_sets.extend(test_shapes)

# fmt:on

benchmark_results = {}
for config, operator, input_args in benchmark_sets:
    config = config(*input_args)
    matmul = operator(config, target=target, enable_tuning=True, backend="tir")
    kernel_latency = matmul.profile_latency()

    profile_config = {
        f"{operator.__name__}-{'-'.join([str(i) for i in input_args])}": kernel_latency
    }

    benchmark_results.update(profile_config)
    

for key, value in benchmark_results.items():
    print(
        f"Input arguments: {key}, Static latency: {value}"
    )
