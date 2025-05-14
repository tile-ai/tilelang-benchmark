# bs	head_num	kv_heads	seq_len	kv_seq_len	dim	Ratio
# 64	64	64	1024	1024	128	50%
# 64	64	64	2048	2048	128	50%
# 64	64	64	4096	4096	128	50%
# 64	64	64	8192	8192	128	50%

mkdir -p logs
python benchmark_torch_bsa.py --batch_size 64 --seqlen 1024 --heads 64 --dim 128 --sparsity 0.5 --block_size 128 --causal True 2>&1 | tee logs/bsa_1024.log
python benchmark_torch_bsa.py --batch_size 64 --seqlen 2048 --heads 64 --dim 128 --sparsity 0.5 --block_size 128 --causal True 2>&1 | tee logs/bsa_2048.log
python benchmark_torch_bsa.py --batch_size 64 --seqlen 4096 --heads 64 --dim 128 --sparsity 0.5 --block_size 128 --causal True 2>&1 | tee logs/bsa_4096.log
python benchmark_torch_bsa.py --batch_size 64 --seqlen 8192 --heads 64 --dim 128 --sparsity 0.5 --block_size 128 --causal True 2>&1 | tee logs/bsa_8192.log
