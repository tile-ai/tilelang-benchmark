mkdir -p ./logs

export PYTHONPATH=/home/srga/lei/tilelang-benchmark/cdna_benchmark/dequantize_matmul/1.triton-benchmark/gemlite:$PYTHONPATH

python ./benchmark_triton_matmul_float16.py --m 1 --n 1024 --k 8192 2>&1 | tee ./logs/benchmark_tilelang_m1_n1024_k8192_float16.log
python ./benchmark_triton_matmul_float16.py --m 1 --n 8192 --k 8192 2>&1 | tee ./logs/benchmark_tilelang_m1_n8192_k8192_float16.log
python ./benchmark_triton_matmul_float16.py --m 1 --n 28672 --k 8192 2>&1 | tee ./logs/benchmark_tilelang_m1_n28672_k8192_float16.log
python ./benchmark_triton_matmul_float16.py --m 1 --n 8192 --k 28672 2>&1 | tee ./logs/benchmark_tilelang_m1_n8192_k28672_float16.log
