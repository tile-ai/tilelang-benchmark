import re

shapes = [
    [1, 16384, 16384],
    [16, 16384, 16384],
    [32, 16384, 16384],
    [64, 16384, 16384],
    [128, 16384, 16384],
    [256, 16384, 16384],
    [1, 8192, 8192],
    [16, 8192, 8192],
    [32, 8192, 8192],
    [64, 8192, 8192],
    [128, 8192, 8192],
    [256, 8192, 8192],
    [1, 28672, 8192],
    [16, 28672, 8192],
    [32, 28672, 8192],
    [64, 28672, 8192],
    [128, 28672, 8192],
    [256, 28672, 8192],
    [1, 8192, 28672],
    [16, 8192, 28672],
    [32, 8192, 28672],
    [64, 8192, 28672],
    [128, 8192, 28672],
    [256, 8192, 28672],
]



A_dtype, W_dtype, out_dtype, accum_dtype = "float16", "int4", "float16", "float16"
A_dtype, W_dtype, out_dtype, accum_dtype = "int8", "int2", "int32", "int32"


for M, N, K in shapes:
    log_file = f"./benchmark_logs/benchmark_{M}_{N}_{K}_{A_dtype}_{W_dtype}_{out_dtype}_{accum_dtype}.log"
    with open(log_file, "r") as f:
        lines = f.readlines()
        # match latency ms
        pattern = re.compile(r"(\d+\.\d+)")
        latency = float(pattern.findall(lines[-1])[0])
        print(f"{M}_{N}_{K}: {latency:.2f} ms")
