# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
export PYTHONPATH=/root/tilelang:$PYTHONPATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID

rm -rf benchmark_logs
mkdir -p benchmark_logs

python benchmark_mla_decode_amd_tilelang.py --batch 128 --kv_ctx 1024 --auto_tune 2>&1 | tee benchmark_logs/benchmark_mla_decode_amd_tilelang_b128_k1024.log
python benchmark_mla_decode_amd_tilelang.py --batch 128 --kv_ctx 2048 --auto_tune 2>&1 | tee benchmark_logs/benchmark_mla_decode_amd_tilelang_b128_k2048.log
python benchmark_mla_decode_amd_tilelang.py --batch 128 --kv_ctx 4096 --auto_tune 2>&1 | tee benchmark_logs/benchmark_mla_decode_amd_tilelang_b128_k4096.log
python benchmark_mla_decode_amd_tilelang.py --batch 128 --kv_ctx 8192 --auto_tune 2>&1 | tee benchmark_logs/benchmark_mla_decode_amd_tilelang_b128_k8192.log
python benchmark_mla_decode_amd_tilelang.py --batch 128 --kv_ctx 16384 --auto_tune 2>&1 | tee benchmark_logs/benchmark_mla_decode_amd_tilelang_b128_k16384.log

python benchmark_mla_decode_amd_tilelang.py --batch 64 --kv_ctx 1024 --auto_tune 2>&1 | tee benchmark_logs/benchmark_mla_decode_amd_tilelang_b64_k1024.log
python benchmark_mla_decode_amd_tilelang.py --batch 64 --kv_ctx 2048 --auto_tune 2>&1 | tee benchmark_logs/benchmark_mla_decode_amd_tilelang_b64_k2048.log
python benchmark_mla_decode_amd_tilelang.py --batch 64 --kv_ctx 4096 --auto_tune 2>&1 | tee benchmark_logs/benchmark_mla_decode_amd_tilelang_b64_k4096.log
python benchmark_mla_decode_amd_tilelang.py --batch 64 --kv_ctx 8192 --auto_tune 2>&1 | tee benchmark_logs/benchmark_mla_decode_amd_tilelang_b64_k8192.log
python benchmark_mla_decode_amd_tilelang.py --batch 64 --kv_ctx 16384 --auto_tune 2>&1 | tee benchmark_logs/benchmark_mla_decode_amd_tilelang_b64_k16384.log
