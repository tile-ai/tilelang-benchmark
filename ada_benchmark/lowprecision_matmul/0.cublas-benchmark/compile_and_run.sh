#!/bin/bash
rm -r build
mkdir -p build
cd build
cmake ..
make clean
make -j
cd ..
./build/cublas_benchmark 2>&1 | tee benchmark_results.log
