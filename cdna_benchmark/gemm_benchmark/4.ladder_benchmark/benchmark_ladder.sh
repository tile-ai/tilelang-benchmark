# !/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
export HIP_VISIBLE_DEVICES=1
export LADDER_HOME=/home/aiscuser/lei/Ladder
export LADDER_TVM_HOME=$LADDER_HOME/3rdparty/tvm
export LADDER_CUTLASS_HOME=$LADDER_HOME/3rdparty/cutlass
# export PYTHONPATH=$LADDER_HOME/python
export PYTHONPATH=/home/aiscuser/lei/welder/python
export PYTHONPATH=$LADDER_TVM_HOME/python:$PYTHONPATH
export CPLUS_INCLUDE_PATH=$LADDER_CUTLASS_HOME/include

/home/aiscuser/miniconda3/bin/python /home/aiscuser/lei/tl_benchmark/gemm_benchmark/benchmark_ladder_gemm.py 2>&1 | tee ladder_gemm.log

# /home/aiscuser/miniconda3/bin/python /home/aiscuser/lei/tl_benchmark/gemm_benchmark/benchmark_ladder_gemm_fp4.py 2>&1 | tee ladder_gemm_fp4.log
