import re

shapes = [
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


for M, N, K in shapes:
    # log_file = f"./benchmark_logs/benchmark_{M}_{N}_{K}_int8_int8_int32_int32.log"
    log_file = f"./benchmark_logs/benchmark_{M}_{N}_{K}_float16_float16_float16_float32.log"
    with open(log_file, "r") as f:
        lines = f.readlines()
        # match latency ms
        pattern = re.compile(r"(\d+\.\d+)")
        for line in lines:
            if "Best latency" in line:
                latency = float(pattern.findall(line)[0])
                print(f"{M}_{N}_{K}: {latency:.3f} ms")
