# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py

import torch

from block_sparse_attn import (
    block_sparse_attn_func,
    flash_attn_varlen_func,
)

from utils import (
    time_fwd,
    flops,
    efficiency,
)

def generate_base_sparsity_mask(max_seqlen_q, max_seqlen_k, round_base, m_block_dim, n_block_dim, sparsity, causal=False, device="cuda"):
    def round_to_multiple(x, base):
        return ((x + base - 1) // base) * base
    nrow, ncol = round_to_multiple(max_seqlen_q, round_base) // m_block_dim, round_to_multiple(max_seqlen_k, round_base) // n_block_dim
    base_mask = torch.zeros(1, nrow, ncol, device=device, dtype=torch.bool)
    total_block_num = 0

    density = 1.0 - sparsity
    if not density == 0.0 and not density == 1.0:
        for i in range(nrow): # do in reverse order
            idx = nrow - i - 1
            if causal:
                available_col_num = max(0, ncol - i)
                total_block_num += available_col_num
                num_one = max(1, int(density * available_col_num))
                base_mask[0][idx, torch.randperm(available_col_num)[:num_one]] = True
            else:
                available_col_num = ncol
                total_block_num += available_col_num
                num_one = max(1, int(density * available_col_num))
                base_mask[0][idx, torch.randperm(available_col_num)[:num_one]] = True
    elif density == 1.0:
        base_mask[0] = torch.ones_like(base_mask[0])
        total_block_num = nrow * ncol
    else:
        total_block_num = nrow * ncol
    
    calculated_block_num = base_mask.sum().item()
    real_sparsity = 1.0 - calculated_block_num / total_block_num
    return base_mask, real_sparsity
   
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--nheads", type=int, default=64)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--dropout_p", type=float, default=0.0)
    parser.add_argument("--causal", type=bool, default=True)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--block_sparse_repeats", type=int, default=1)
    args = parser.parse_args()
    
    repeats = args.repeats
    block_sparse_repeats = args.block_sparse_repeats
    batch_size = args.batch_size
    seqlen = args.seqlen
    nheads = args.nheads
    dim = args.dim
    sparsity = args.sparsity
    block_size = args.block_size
    dropout_p = args.dropout_p
    causal = args.causal

    device = 'cuda:0'
    dtype = torch.float16

    method = ("Block_Sparse_Flash2")
    time_f = {}
    speed_f = {}
    
    all_results = {}
    results = {}
    shape = (batch_size * seqlen, nheads, dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device)
    base_f = time_fwd(flash_attn_varlen_func, q, k, v, cu_seqlens, cu_seqlens, seqlen, seqlen, dropout_p, None, causal, repeats=repeats, verbose=False)
    base_speed = efficiency(flops(batch_size, seqlen, dim, nheads, causal, mode="fwd"), base_f)
    results["base"] = [[base_f], [base_speed]]

    sum_sparsity, sum_speed, sum_latency = 0, 0, 0
    for _ in range(block_sparse_repeats):
        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device)
        head_mask_type = torch.tensor([1] * nheads, device=device, dtype=torch.int32)
        base_blockmask, real_sparsity = generate_base_sparsity_mask(seqlen, seqlen, block_size, block_size, block_size, sparsity, causal = causal, device=device)
        base_blockmask = base_blockmask.unsqueeze(0).repeat(batch_size, nheads, 1, 1)
        config = (causal, dim, nheads, batch_size, seqlen, sparsity, real_sparsity)
        f = time_fwd(block_sparse_attn_func, q, k, v, cu_seqlens, cu_seqlens, head_mask_type, None, base_blockmask, seqlen, seqlen, dropout_p, is_causal=causal, exact_streaming=False, repeats=repeats, verbose=False)
        time_f[config, method] = f
        print(f"### causal={causal}, headdim={dim}, nheads = {nheads}, batch_size={batch_size}, seqlen={seqlen}, real_sparsity={real_sparsity} ###")
        speed_f[config, method] = efficiency(flops(batch_size, seqlen, dim, nheads, causal, mode="fwd"), time_f[config, method])
        print(
            f"{method}"
            f"fwd: {speed_f[config, method]:.2f} TFLOPs/s, {(time_f[config, method]*1000):.2f} ms, "
            f"fwd base: {base_speed:.2f} TFLOPs/s, {base_f*1000:.2f} ms"
            ) 
        sum_sparsity += real_sparsity
        sum_speed += speed_f[config, method]
        sum_latency += time_f[config, method]
