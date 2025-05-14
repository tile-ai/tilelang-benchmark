# bs	head_num	kv_heads	seq_len	kv_seq_len	dim	Ratio
# 64	64	64	1024	1024	128	50%
# 64	64	64	2048	2048	128	50%
# 64	64	64	4096	4096	128	50%
# 64	64	64	8192	8192	128	50%

python benchmark_block_sparse_attn.py --batch_size 64 --seqlen 1024 --nheads 64 --dim 128 --sparsity 0.5 --block_size 128 --causal True 
python benchmark_block_sparse_attn.py --batch_size 64 --seqlen 2048 --nheads 64 --dim 128 --sparsity 0.5 --block_size 128 --causal True 
python benchmark_block_sparse_attn.py --batch_size 64 --seqlen 4096 --nheads 64 --dim 128 --sparsity 0.5 --block_size 128 --causal True 
python benchmark_block_sparse_attn.py --batch_size 64 --seqlen 8192 --nheads 64 --dim 128 --sparsity 0.5 --block_size 128 --causal True 

