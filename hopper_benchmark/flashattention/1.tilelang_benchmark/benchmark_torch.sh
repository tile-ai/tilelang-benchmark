export CUDA_VISIBLE_DEVICES=1

# bs	head_num	kv_heads	seq_len	kv_seq_len	dim
# 64	64	64	1024	1024	128
# 64	64	64	8192	8192	128
# 64	64	64	1	1024	128
# 64	64	64	1	8192	128
python benchmark_torch_mha.py --batch 64 --heads 64 --seq_q 1024 --seq_kv 1024 --dim 128
python benchmark_torch_mha.py --batch 64 --heads 64 --seq_q 8192 --seq_kv 8192 --dim 128
python benchmark_torch_mha.py --batch 64 --heads 64 --seq_q 1 --seq_kv 1024 --dim 128
python benchmark_torch_mha.py --batch 64 --heads 64 --seq_q 1 --seq_kv 8192 --dim 128
