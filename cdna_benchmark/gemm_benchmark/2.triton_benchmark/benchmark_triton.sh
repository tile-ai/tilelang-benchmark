# python ./benchmark_triton_matmul.py --m 2048 --n 2048 --k 2048 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_2048_2048_2048.log
# python ./benchmark_triton_matmul.py --m 4096 --n 4096 --k 4096 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_4096_4096_4096.log
# python ./benchmark_triton_matmul.py --m 8192 --n 8192 --k 8192 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_8192_8192.log
# python ./benchmark_triton_matmul.py --m 16384 --n 16384 --k 16384 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_16384_16384_16384.log
# python ./benchmark_triton_matmul.py --m 8192 --n 8192 --k 1024 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_8192_1024.log
# python ./benchmark_triton_matmul.py --m 8192 --n 8192 --k 2048 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_8192_2048.log
# python ./benchmark_triton_matmul.py --m 8192 --n 8192 --k 4096 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_8192_4096.log

mkdir -p ./logs/triton_gemm

# python ./benchmark_triton_matmul.py --m 16384 --n 16384 --k 16384 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_16384_16384_16384.log
# python ./benchmark_triton_matmul.py --m 8192 --n 43008 --k 14336 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_43008_14336.log
# python ./benchmark_triton_matmul.py --m 8192 --n 14336 --k 14336 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_14336_14336.log
# python ./benchmark_triton_matmul.py --m 8192 --n 57344 --k 14336 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_57344_14336.log
# python ./benchmark_triton_matmul.py --m 8192 --n 14336 --k 57344 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_14336_57344.log
# python ./benchmark_triton_matmul.py --m 8192 --n 9216 --k 9216 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_9216_9216.log
# python ./benchmark_triton_matmul.py --m 8192 --n 36864 --k 9216 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_36864_9216.log
# python ./benchmark_triton_matmul.py --m 8192 --n 9216 --k 36864 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_9216_36864.log
# python ./benchmark_triton_matmul.py --m 8192 --n 22016 --k 8192 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_22016_8192.log
# python ./benchmark_triton_matmul.py --m 8192 --n 8192 --k 22016 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_8192_22016.log
# python ./benchmark_triton_matmul.py --m 8192 --n 8192 --k 8192 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_8192_8192.log
# python ./benchmark_triton_matmul.py --m 8192 --n 28672 --k 8192 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_28672_8192.log
# python ./benchmark_triton_matmul.py --m 8192 --n 8192 --k 28672 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_8192_28672.log


# M	N	K
# 8192	1024	8192
# 8192	8192	8192
# 8192	28672	8192
# 8192	8192	28672

python ./benchmark_triton_matmul.py --m 8192 --n 1024 --k 8192 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_1024_8192.log
python ./benchmark_triton_matmul.py --m 8192 --n 8192 --k 8192 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_8192_8192.log
python ./benchmark_triton_matmul.py --m 8192 --n 28672 --k 8192 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_28672_8192.log
python ./benchmark_triton_matmul.py --m 8192 --n 8192 --k 28672 2>&1 | tee ./logs/triton_gemm/run_gemm_triton_8192_8192_28672.log
