#!/bin/bash
export PYTHONPATH=$(pwd)/cutlass_fpa_intb_tvm/python
mkdir ./tmp
python -u ./cutlass_fpa_intb.py 2>&1 | tee cutlass_fpa_intb.log
