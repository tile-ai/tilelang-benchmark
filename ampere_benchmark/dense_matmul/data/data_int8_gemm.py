# fmt: off
import os
import re
dirpath = os.path.dirname(os.path.abspath(__file__))
matmul_providers = ["M0","M1","M2","M3","M4","M5","M6","M7","M8","M9","M10","M11","M12"]

matmul_times_data = [
    ('cuBLAS-W$_{INT8}$A$_{INT8}$', [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
    ('Triton-W$_{INT8}$A$_{INT8}$', [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
    ('BitBLAS-W$_{INT8}$A$_{INT8}$',  [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
]

# parse the results from cublas
cublas_data = matmul_times_data[0][1]
def get_and_print_cublas(m, n, k, log):
    data = None
    with open(log) as f:
        lines = f.readlines()
        for line in lines:
            if f"{m},{n},{k}" in line:
                data = float(re.findall(r"\d+\.\d+", line)[-1])
                print(data)
    return data

for i, (m, n, k) in enumerate([
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
    ]):
    log_path = f"{dirpath}/../0.cublas-benchmark/benchmark_results.log"
    if not os.path.exists(log_path):
        continue
    data = get_and_print_cublas(m, n, k, log_path)
    cublas_data[i] = data
matmul_times_data[0] = (matmul_times_data[0][0], cublas_data)

triton_data = matmul_times_data[1][1]
def get_and_print_triton(log):
    data = None
    with open(log) as f:
        content = f.read()
        data = float(re.findall(r"\d+\.\d+", content)[-2])
        print(data)
    return data

for i, (m, n, k) in enumerate([
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
    ]):
    log_path = f"{dirpath}/../1.triton-benchmark/logs/benchmark_tilelang_m_{m}_n_{n}_k_{k}_int8.log"
    if not os.path.exists(log_path):
        continue
    print(log_path)
    data = get_and_print_triton(log_path)
    triton_data[i] = data
matmul_times_data[1] = (matmul_times_data[1][0], triton_data)


bitblas_data = matmul_times_data[2][1]
def get_and_print_bitblas(log):
    data = None
    with open(log) as f:
        content = f.read()
        data = float(re.findall(r"\d+\.\d+", content)[-1])
        print(data)
    return data

for i, (m, n, k) in enumerate([
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
    ]):
    log_path = f"{dirpath}/../2.bitblas-benchmark/benchmark_logs/benchmark_{m}_{n}_{k}_int8_int8_int32_int32.log"
    if not os.path.exists(log_path):
        continue
    print(log_path)
    data = get_and_print_bitblas(log_path)
    bitblas_data[i] = data
matmul_times_data[2] = (matmul_times_data[2][0], bitblas_data)

print(matmul_times_data)
