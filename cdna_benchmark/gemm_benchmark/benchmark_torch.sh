python /home/aiscuser/lei/tl_experiments/benchmark_torch_matmul.py --m 2048 --n 2048 --k 2048 2>&1 | tee /home/aiscuser/lei/tl_experiments/logs/torch_gemm/run_gemm_torch_2048_2048_2048.log
python /home/aiscuser/lei/tl_experiments/benchmark_torch_matmul.py --m 4096 --n 4096 --k 4096 2>&1 | tee /home/aiscuser/lei/tl_experiments/logs/torch_gemm/run_gemm_torch_4096_4096_4096.log
python /home/aiscuser/lei/tl_experiments/benchmark_torch_matmul.py --m 8192 --n 8192 --k 8192 2>&1 | tee /home/aiscuser/lei/tl_experiments/logs/torch_gemm/run_gemm_torch_8192_8192_8192.log
python /home/aiscuser/lei/tl_experiments/benchmark_torch_matmul.py --m 16384 --n 16384 --k 16384 2>&1 | tee /home/aiscuser/lei/tl_experiments/logs/torch_gemm/run_gemm_torch_16384_16384_16384.log
python /home/aiscuser/lei/tl_experiments/benchmark_torch_matmul.py --m 8192 --n 8192 --k 1024 2>&1 | tee /home/aiscuser/lei/tl_experiments/logs/torch_gemm/run_gemm_torch_8192_8192_1024.log
python /home/aiscuser/lei/tl_experiments/benchmark_torch_matmul.py --m 8192 --n 8192 --k 2048 2>&1 | tee /home/aiscuser/lei/tl_experiments/logs/torch_gemm/run_gemm_torch_8192_8192_2048.log
python /home/aiscuser/lei/tl_experiments/benchmark_torch_matmul.py --m 8192 --n 8192 --k 4096 2>&1 | tee /home/aiscuser/lei/tl_experiments/logs/torch_gemm/run_gemm_torch_8192_8192_4096.log
