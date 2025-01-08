mkdir -p ./logs

python ./benchmark_triton_matmul_float16.py --m 4096 --n 1024 --k 4096 2>&1 | tee ./logs/benchmark_tilelang_m4096_n1024_k4096_float16.log
python ./benchmark_triton_matmul_float16.py --m 4096 --n 4096 --k 4096 2>&1 | tee ./logs/benchmark_tilelang_m4096_n4096_k4096_float16.log
python ./benchmark_triton_matmul_float16.py --m 4096 --n 14336 --k 4096 2>&1 | tee ./logs/benchmark_tilelang_m4096_n14336_k4096_float16.log
python ./benchmark_triton_matmul_float16.py --m 4096 --n 4096 --k 14336 2>&1 | tee ./logs/benchmark_tilelang_m4096_n4096_k14336_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 1024 --k 4096 2>&1 | tee ./logs/benchmark_tilelang_m8192_n1024_k4096_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 4096 --k 4096 2>&1 | tee ./logs/benchmark_tilelang_m8192_n4096_k4096_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 14336 --k 4096 2>&1 | tee ./logs/benchmark_tilelang_m8192_n14336_k4096_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 4096 --k 14336 2>&1 | tee ./logs/benchmark_tilelang_m8192_n4096_k14336_float16.log
python ./benchmark_triton_matmul_float16.py --m 4096 --n 1024 --k 8192 2>&1 | tee ./logs/benchmark_tilelang_m4096_n1024_k8192_float16.log
python ./benchmark_triton_matmul_float16.py --m 4096 --n 8192 --k 8192 2>&1 | tee ./logs/benchmark_tilelang_m4096_n8192_k8192_float16.log
python ./benchmark_triton_matmul_float16.py --m 4096 --n 28672 --k 8192 2>&1 | tee ./logs/benchmark_tilelang_m4096_n28672_k8192_float16.log
python ./benchmark_triton_matmul_float16.py --m 4096 --n 8192 --k 28672 2>&1 | tee ./logs/benchmark_tilelang_m4096_n8192_k28672_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 1024 --k 8192 2>&1 | tee ./logs/benchmark_tilelang_m8192_n1024_k8192_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 8192 --k 8192 2>&1 | tee ./logs/benchmark_tilelang_m8192_n8192_k8192_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 28672 --k 8192 2>&1 | tee ./logs/benchmark_tilelang_m8192_n28672_k8192_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 8192 --k 28672 2>&1 | tee ./logs/benchmark_tilelang_m8192_n8192_k28672_float16.log
