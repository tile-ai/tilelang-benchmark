export CUDA_VISIBLE_DEVICES=1

# bs	head_num	kv_heads	seq_len	kv_seq_len	dim
# 64	64	64	1024	1024	128
python benchmark_torch_mha.py --batch 64 --heads 64 --seq_q 1024 --seq_kv 1024 --dim 128
python benchmark_torch_mha.py --batch 64 --heads 64 --seq_q 2048 --seq_kv 2048 --dim 128
python benchmark_torch_mha.py --batch 64 --heads 64 --seq_q 4096 --seq_kv 4096 --dim 128
python benchmark_torch_mha.py --batch 64 --heads 64 --seq_q 8192 --seq_kv 8192 --dim 128

