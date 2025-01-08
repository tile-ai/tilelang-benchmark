import re

shapes = [
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


A_dtype, W_dtype, out_dtype, accum_dtype = "float16", "int4", "float16", "float16"
A_dtype, W_dtype, out_dtype, accum_dtype = "float16", "int2", "float16", "float16"
A_dtype, W_dtype, out_dtype, accum_dtype = "int8", "int2", "int32", "int32"
# A_dtype, W_dtype, out_dtype, accum_dtype = "float16", "nf4", "float16", "float16"


for M, N, K in shapes:
    log_file = f"./benchmark_logs/benchmark_{M}_{N}_{K}_{A_dtype}_{W_dtype}_{out_dtype}_{accum_dtype}.log"
    with open(log_file, "r") as f:
        lines = f.readlines()
        # match latency ms
        pattern = re.compile(r"(\d+\.\d+)")
        latency = float(pattern.findall(lines[-1])[0])
        print(f"{M}_{N}_{K}: {latency:.2f} ms")