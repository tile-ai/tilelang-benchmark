import argparse
import torch
import time
from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_func
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--h', type=int, default=12, help='Number of heads')
    parser.add_argument('--n_ctx', type=int, default=2048, help='Context size')
    parser.add_argument('--d_head', type=int, default=128, help='Head dimension')
    parser.add_argument('--casual', type=bool, default=False, help='Casual flag')
    args = parser.parse_args()
    BATCH = args.batch
    H = args.h
    N_CTX = args.n_ctx
    HEAD_DIM = args.d_head
    casual = args.casual
    
    dtype = torch.float16
    device = torch.device('cuda')

    qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    def ref_program(qkv, casual=False):
        flash_attn_func(qkv, causal=casual)

    # Warm-up
    for _ in range(5):
        _ = ref_program(qkv, casual)

    
    iters = 10

    # Measure latency
    torch.cuda.synchronize()  # Ensure all GPU operations are complete
    start_time = time.time()
    for _ in range(iters):
        _ = ref_program(qkv, casual)
    torch.cuda.synchronize()  # Ensure all GPU operations are complete
    ref_latency = (time.time() - start_time) / iters
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if casual:
        total_flops *= 0.5
    
    tflops = total_flops / ref_latency * 1e-12
    
    print(f"tflops: {tflops}")