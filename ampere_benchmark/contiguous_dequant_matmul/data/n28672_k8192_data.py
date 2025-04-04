# fmt: off
import os
import re
dirpath = os.path.dirname(os.path.abspath(__file__))
matmul_providers = ["1", "16", "32", "64", "128", "256"]

matmul_times_data = [
    ('cuBLAS-W$_{FP16}$A$_{FP16}$', [-1, -1, -1, -1, -1, -1]),
    ('CUTLASS-W$_{INT4}$A$_{FP16}$', [-1, -1, -1, -1, -1, -1]),
    ('Marlin-W$_{INT4}$A$_{FP16}$', [-1, -1, -1, -1, -1, -1]),
    ('BitBLAS-W$_{INT4}$A$_{FP16}$', [-1, -1, -1, -1, -1, -1]),
    ('BitBLAS-W$_{INT2}$A$_{INT8}$', [-1, -1, -1, -1, -1, -1]),
]

# parse the results from cublas
cublas_data = matmul_times_data[0][1]
def get_and_print_cublas(m, n, k, log):
    data = None
    with open(log) as f:
        lines = f.readlines()
        for line in lines:
            if f"{m},{n},{k}" in line:
                data = float(re.findall(r"\d+\.\d+", line)[-2])
                print(data)
    return data

for i, (m, n, k) in enumerate([
        [1, 28672, 8192],
        [16, 28672, 8192],
        [32, 28672, 8192],
        [64, 28672, 8192],
        [128, 28672, 8192],
        [256, 28672, 8192],
    ]):
    log_path = f"{dirpath}/../0.cublas-benchmark/benchmark_results.log"
    if not os.path.exists(log_path):
        continue
    data = get_and_print_cublas(m, n, k, log_path)
    cublas_data[i] = data
matmul_times_data[0] = (matmul_times_data[0][0], cublas_data)

cutlass_data = matmul_times_data[1][1]
def get_and_print_cutlass(m, n, k, log):
    data = None
    with open(log) as f:
        lines = f.readlines()
        for line in lines:
            if f"{m}_{n}_{k}" in line:
                data = float(re.findall(r"\d+\.\d+", line)[-1])
                print(data)
    return data

for i, (m, n, k) in enumerate([
        [1, 28672, 8192],
        [16, 28672, 8192],
        [32, 28672, 8192],
        [64, 28672, 8192],
        [128, 28672, 8192],
        [256, 28672, 8192],
    ]):
    log_path = f"{dirpath}/../2.cutlass_fpa_intb_benchmark/cutlass_fpa_intb.log"
    if not os.path.exists(log_path):
        continue
    data = get_and_print_cutlass(m, n, k, log_path)
    cutlass_data[i] = data
matmul_times_data[1] = (matmul_times_data[1][0], cutlass_data)


marlin_data = matmul_times_data[2][1]
def get_and_print_marlin(log):
    data = None
    with open(log) as f:
        content = f.read()
        data = float(re.findall(r"\d+\.\d+", content)[-1])
        print(data)
    return data

for i, (m, n, k) in enumerate([
        [1, 28672, 8192],
        [16, 28672, 8192],
        [32, 28672, 8192],
        [64, 28672, 8192],
        [128, 28672, 8192],
        [256, 28672, 8192],
    ]):
    log_path = f"{dirpath}/../1.marlin_benchmark/kernel_benchmark.log"
    print(log_path)
    if not os.path.exists(log_path):
        continue
    data = get_and_print_cutlass(m, n, k, log_path)
    marlin_data[i] = data
matmul_times_data[2] = (matmul_times_data[2][0], marlin_data)

bitblas_fp16_int4_data = matmul_times_data[3][1]
def get_and_print_bitblas(log):
    data = None
    with open(log) as f:
        content = f.read()
        data = float(re.findall(r"\d+\.\d+", content)[-1])
        print(data)
    return data

for i, (m, n, k) in enumerate([
        [1, 28672, 8192],
        [16, 28672, 8192],
        [32, 28672, 8192],
        [64, 28672, 8192],
        [128, 28672, 8192],
        [256, 28672, 8192],
    ]):
    log_path = f"{dirpath}/../3.bitblas_benchmark/benchmark_logs/benchmark_{m}_{n}_{k}_float16_int4_float16_float16.log"
    if not os.path.exists(log_path):
        continue
    data = get_and_print_bitblas(log_path)
    bitblas_fp16_int4_data[i] = data
matmul_times_data[3] = (matmul_times_data[3][0], bitblas_fp16_int4_data)


bitblas_int2_int8_data = matmul_times_data[4][1]
def get_and_print_bitblas(log):
    data = None
    with open(log) as f:
        content = f.read()
        data = float(re.findall(r"\d+\.\d+", content)[-1])
        print(data)
    return data

for i, (m, n, k) in enumerate([
        [1, 28672, 8192],
        [16, 28672, 8192],
        [32, 28672, 8192],
        [64, 28672, 8192],
        [128, 28672, 8192],
        [256, 28672, 8192],
    ]):
    log_path = f"{dirpath}/../3.bitblas_benchmark/benchmark_logs/benchmark_{m}_{n}_{k}_int8_int2_int32_int32.log"
    if not os.path.exists(log_path):
        continue
    data = get_and_print_bitblas(log_path)
    bitblas_int2_int8_data[i] = data

matmul_times_data[4] = (matmul_times_data[4][0], bitblas_int2_int8_data)

print(matmul_times_data)
