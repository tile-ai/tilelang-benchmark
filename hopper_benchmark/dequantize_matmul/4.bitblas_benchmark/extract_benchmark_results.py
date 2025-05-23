import re

shapes = [
    [1, 1024, 8192],
    [1, 8192, 8192],
    [1, 28672, 8192],
    [1, 8192, 28672],
    [1, 1024, 8192],
    [1, 8192, 8192],
    [1, 28672, 8192],
    [1, 8192, 28672],
    [1, 8192, 24576],
    [1, 8192, 8192],
    [1, 24576, 8192],
    [1, 8192, 24576],
    [1, 8192, 8192],
    [1, 24576, 8192],
    [1024, 1024, 8192],
    [1024, 8192, 8192],
    [1024, 28672, 8192],
    [1024, 8192, 28672],
    [8192, 1024, 8192],
    [8192, 8192, 8192],
    [8192, 28672, 8192],
    [8192, 8192, 28672],
    [1024, 8192, 24576],
    [1024, 8192, 8192],
    [1024, 24576, 8192],
    [8192, 8192, 24576],
    [8192, 8192, 8192],
    [8192, 24576, 8192],
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
        print(f"{M}_{N}_{K}: {latency:.4f} ms")