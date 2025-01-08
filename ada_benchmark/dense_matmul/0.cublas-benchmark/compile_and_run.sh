#!/bin/bash
mkdir -p build
cd build
cmake ..
make -j
cd ..
./build/cublas_benchmark 2>&1 | tee benchmark_results.log
