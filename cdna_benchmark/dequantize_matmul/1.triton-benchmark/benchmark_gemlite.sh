mkdir -p logs
export PYTHONPATH=/home/srga/lei/tilelang-benchmark/cdna_benchmark/dequantize_matmul/1.triton-benchmark/gemlite:$PYTHONPATH
python3 benchmark_gemlite.py --in_features 1024 --out_features 8192 > logs/gemlite_1024_8192.log
python3 benchmark_gemlite.py --in_features 8192 --out_features 8192 > logs/gemlite_8192_8192.log
python3 benchmark_gemlite.py --in_features 28672 --out_features 8192 > logs/gemlite_28672_8192.log
python3 benchmark_gemlite.py --in_features 8192 --out_features 28672 > logs/gemlite_8192_28672.log
