#!/bin/bash

cd 0.cublas-benchmark
./compile_and_run.sh
cd ..

cd 1.bitblas_benchmark
./benchmark_bitblas_matmul.sh
cd ..
