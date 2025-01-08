#!/bin/bash
git clone https://github.com/apache/tvm --recursive cutlass_fpa_intb_tvm
cd cutlass_fpa_intb_tvm
git checkout 2bf3a0a4287069ac55ee3304c285b08592d3d1bc
git submodule update --init --recursive
mkdir -p build
cd build
cp ../config.cmake .
echo "set(USE_CUDA ON)" >> config.cmake
echo "set(USE_LLVM ON)" >> config.cmake
echo "set(USE_CUBLAS ON)" >> config.cmake
echo "set(USE_CUTLASS ON)" >> config.cmake
cmake -DCMAKE_CUDA_ARCHITECTURES="80" ..
make -j 16

cd ../..

