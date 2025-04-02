import os
import re
triton_data = []


def get_and_print_triton(log):
    data = None
    with open(log) as f:
        content = f.read()
        data = float(re.findall(r"\d+\.\d+", content)[-2])
        print(data)
    return data


for i, (m, n, k) in enumerate(
    [
        [4096, 1024, 4096],
        [4096, 4096, 4096],
        [4096, 14336, 4096],
        [4096, 4096, 14336],
        [8192, 1024, 4096],
        [8192, 4096, 4096],
        [8192, 14336, 4096],
        [8192, 4096, 14336],
        [4096, 1024, 8192],
        [4096, 8192, 8192],
        [4096, 28672, 8192],
        [4096, 8192, 28672],
        [8192, 1024, 8192],
        [8192, 8192, 8192],
        [8192, 28672, 8192],
        [8192, 8192, 28672],
    ]
):
    log_path = f"./logs/benchmark_tilelang_m{m}_n{n}_k{k}_float16.log"
    if not os.path.exists(log_path):
        continue
    data = get_and_print_triton(log_path)
    triton_data.append(data)

