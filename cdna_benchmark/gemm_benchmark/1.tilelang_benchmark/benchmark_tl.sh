export HIP_VISIBLE_DEVICES=1

# export PYTHONPATH=/home/srga/lei/tilelang:$PYTHONPATH
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export TVM_HOME=/home/srga/lei/tilelang/3rdparty/tvm
# export PYTHONPATH=$TVM_HOME/python:/home/srga/lei/tilelang:$PYTHONPATH

# python ./benchmark_tilelang_matmul.py --m 2048 --n 2048 --k 2048 2>&1 | tee run_gemm_tilelang_2048_2048_2048.log
# python ./benchmark_tilelang_matmul.py --m 4096 --n 4096 --k 4096 2>&1 | tee run_gemm_tilelang_4096_4096_4096.log
# python ./benchmark_tilelang_matmul.py --m 8192 --n 8192 --k 8192 2>&1 | tee run_gemm_tilelang_8192_8192_8192.log
# python ./benchmark_tilelang_matmul.py --m 16384 --n 16384 --k 16384 2>&1 | tee run_gemm_tilelang_16384_16384_16384.log
# python ./benchmark_tilelang_matmul.py --m 8192 --n 8192 --k 1024 2>&1 | tee run_gemm_tilelang_8192_8192_1024.log
# python ./benchmark_tilelang_matmul.py --m 8192 --n 8192 --k 2048 2>&1 | tee run_gemm_tilelang_8192_8192_2048.log
# python ./benchmark_tilelang_matmul.py --m 8192 --n 8192 --k 4096 2>&1 | tee run_gemm_tilelang_8192_8192_4096.log

# M	N	K
# 8192	1024	8192
# 8192	8192	8192
# 8192	28672	8192
# 8192	8192	28672
mkdir -p ./logs/tl_gemm
python ./benchmark_tilelang_matmul.py --m 8192 --n 1024 --k 8192 2>&1 | tee ./logs/tl_gemm/run_gemm_tilelang_8192_1024_8192.log
python ./benchmark_tilelang_matmul.py --m 8192 --n 8192 --k 8192 2>&1 | tee ./logs/tl_gemm/run_gemm_tilelang_8192_8192_8192.log
python ./benchmark_tilelang_matmul.py --m 8192 --n 28672 --k 8192 2>&1 | tee ./logs/tl_gemm/run_gemm_tilelang_8192_28672_8192.log
python ./benchmark_tilelang_matmul.py --m 8192 --n 8192 --k 28672 2>&1 | tee ./logs/tl_gemm/run_gemm_tilelang_8192_8192_28672.log
