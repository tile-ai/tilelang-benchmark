
################        op  datatype  layout  verify  init  log  repeat  M___ N___ K___  StrideA StrideB StrideC
# ./bin/ckProfiler      gemm         1       1       1     1    0       5  3840 4096 4096     4096    4096    4096
mkdir -p ./ck_benchmark_result
# ./composable_kernel/build/bin/ckProfiler      gemm         1       1       1     1    0       5   8192 1024 8192     8192 8192 8192 2>&1 | tee ./ck_benchmark_result/gemm_8192_1024_8192.txt
# ./composable_kernel/build/bin/ckProfiler      gemm         1       1       1     1    0       5   8192 8192 8192     8192 8192 8192 2>&1 | tee ./ck_benchmark_result/gemm_8192_8192_8192.txt
# ./composable_kernel/build/bin/ckProfiler      gemm         1       1       1     1    0       5   8192 28762 8192     8192 8192 8192 2>&1 | tee ./ck_benchmark_result/gemm_8192_28762_8192.txt
# ./composable_kernel/build/bin/ckProfiler      gemm         1       1       1     1    0       5   8192 8192 28762     28762 28762 28762 2>&1 | tee ./ck_benchmark_result/gemm_8192_8192_28762.txt

./composable_kernel/build/bin/tile_example_gemm_basic -m=8192 -n=1024 -k=8192 2>&1 | tee ./ck_benchmark_result/gemm_8192_1024_8192.txt
./composable_kernel/build/bin/tile_example_gemm_basic -m=8192 -n=8192 -k=8192 2>&1 | tee ./ck_benchmark_result/gemm_8192_8192_8192.txt
./composable_kernel/build/bin/tile_example_gemm_basic -m=8192 -n=28762 -k=8192 -stride_a=8192 -stride_b=28762 -stride_c=28762 2>&1 | tee ./ck_benchmark_result/gemm_8192_28762_8192.txt
./composable_kernel/build/bin/tile_example_gemm_basic -m=8192 -n=8192 -k=28762 -stride_a=28762 -stride_b=8192 -stride_c=8192 2>&1 | tee ./ck_benchmark_result/gemm_8192_8192_28762.txt

./composable_kernel/build/bin/tile_example_fmha_fwd -b=64 -h=64 -s=1024 -d=128 -v=0 2>&1 | tee ./ck_benchmark_result/fmha_fwd_64_64_1024_128.txt
./composable_kernel/build/bin/tile_example_fmha_fwd -b=64 -h=64 -s=2048 -d=128 -v=0 2>&1 | tee ./ck_benchmark_result/fmha_fwd_64_64_2048_128.txt
./composable_kernel/build/bin/tile_example_fmha_fwd -b=64 -h=64 -s=4096 -d=128 -v=0 2>&1 | tee ./ck_benchmark_result/fmha_fwd_64_64_4096_128.txt
./composable_kernel/build/bin/tile_example_fmha_fwd -b=64 -h=64 -s=8192 -d=128 -v=0 2>&1 | tee ./ck_benchmark_result/fmha_fwd_64_64_8192_128.txt
