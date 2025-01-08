mkdir -p ./logs

python ./benchmark_triton_matmul_float16.py --m 1 --n 16384 --k 16384 2>&1 | tee ./logs/benchmark_tilelang_m_1_n_16384_k_16384_float16.log
python ./benchmark_triton_matmul_float16.py --m 1 --n 43008 --k 14336 2>&1 | tee ./logs/benchmark_tilelang_m_1_n_43008_k_14336_float16.log
python ./benchmark_triton_matmul_float16.py --m 1 --n 14336 --k 14336 2>&1 | tee ./logs/benchmark_tilelang_m_1_n_14336_k_14336_float16.log
python ./benchmark_triton_matmul_float16.py --m 1 --n 57344 --k 14336 2>&1 | tee ./logs/benchmark_tilelang_m_1_n_57344_k_14336_float16.log
python ./benchmark_triton_matmul_float16.py --m 1 --n 14336 --k 57344 2>&1 | tee ./logs/benchmark_tilelang_m_1_n_14336_k_57344_float16.log
python ./benchmark_triton_matmul_float16.py --m 1 --n 9216 --k 9216 2>&1 | tee ./logs/benchmark_tilelang_m_1_n_9216_k_9216_float16.log
python ./benchmark_triton_matmul_float16.py --m 1 --n 36864 --k 9216 2>&1 | tee ./logs/benchmark_tilelang_m_1_n_36864_k_9216_float16.log
python ./benchmark_triton_matmul_float16.py --m 1 --n 9216 --k 36864 2>&1 | tee ./logs/benchmark_tilelang_m_1_n_9216_k_36864_float16.log
python ./benchmark_triton_matmul_float16.py --m 1 --n 22016 --k 8192 2>&1 | tee ./logs/benchmark_tilelang_m_1_n_22016_k_8192_float16.log
python ./benchmark_triton_matmul_float16.py --m 1 --n 8192 --k 22016 2>&1 | tee ./logs/benchmark_tilelang_m_1_n_8192_k_22016_float16.log
python ./benchmark_triton_matmul_float16.py --m 1 --n 8192 --k 8192 2>&1 | tee ./logs/benchmark_tilelang_m_1_n_8192_k_8192_float16.log
python ./benchmark_triton_matmul_float16.py --m 1 --n 28672 --k 8192 2>&1 | tee ./logs/benchmark_tilelang_m_1_n_28672_k_8192_float16.log
python ./benchmark_triton_matmul_float16.py --m 1 --n 8192 --k 28672 2>&1 | tee ./logs/benchmark_tilelang_m_1_n_8192_k_28672_float16.log
python ./benchmark_triton_matmul_float16.py --m 16384 --n 16384 --k 16384 2>&1 | tee ./logs/benchmark_tilelang_m_16384_n_16384_k_16384_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 43008 --k 14336 2>&1 | tee ./logs/benchmark_tilelang_m_8192_n_43008_k_14336_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 14336 --k 14336 2>&1 | tee ./logs/benchmark_tilelang_m_8192_n_14336_k_14336_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 57344 --k 14336 2>&1 | tee ./logs/benchmark_tilelang_m_8192_n_57344_k_14336_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 14336 --k 57344 2>&1 | tee ./logs/benchmark_tilelang_m_8192_n_14336_k_57344_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 9216 --k 9216 2>&1 | tee ./logs/benchmark_tilelang_m_8192_n_9216_k_9216_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 36864 --k 9216 2>&1 | tee ./logs/benchmark_tilelang_m_8192_n_36864_k_9216_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 9216 --k 36864 2>&1 | tee ./logs/benchmark_tilelang_m_8192_n_9216_k_36864_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 22016 --k 8192 2>&1 | tee ./logs/benchmark_tilelang_m_8192_n_22016_k_8192_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 8192 --k 22016 2>&1 | tee ./logs/benchmark_tilelang_m_8192_n_8192_k_22016_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 8192 --k 8192 2>&1 | tee ./logs/benchmark_tilelang_m_8192_n_8192_k_8192_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 28672 --k 8192 2>&1 | tee ./logs/benchmark_tilelang_m_8192_n_28672_k_8192_float16.log
python ./benchmark_triton_matmul_float16.py --m 8192 --n 8192 --k 28672 2>&1 | tee ./logs/benchmark_tilelang_m_8192_n_8192_k_28672_float16.log
