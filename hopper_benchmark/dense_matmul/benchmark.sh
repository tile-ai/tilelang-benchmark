#!/bin/bash

cd 0.cublas-benchmark
./compile_and_run.sh
cd ..

cd 1.triton-benchmark
./benchmark_float16.sh
cd ..

cd 2.tilelang-benchmark
./benchmark_bitblas_matmul.sh
cd ..
