import argparse
import torch
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=16384, help='M dimension of the matrix')
    parser.add_argument('--n', type=int, default=16384, help='N dimension of the matrix')
    parser.add_argument('--k', type=int, default=16384, help='K dimension of the matrix')
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k
    total_flops = 2 * M * N * K
    
    iters = 10

    # Initialize random tensors
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    
    model = torch.nn.Linear(K, N, bias=False).cuda().half()
    
    def ref_program(A):
        return model(A)

    # Warm-up
    for _ in range(5):
        _ = ref_program(A)

    # Measure latency
    torch.cuda.synchronize()  # Ensure all GPU operations are complete
    start_time = time.time()
    for _ in range(iters):
        _ = ref_program(A)
    torch.cuda.synchronize()  # Ensure all GPU operations are complete
    ref_latency = (time.time() - start_time) / iters

    # Calculate TFlops
    ref_tflops = total_flops / ref_latency * 1e-12
    print(f"Matrix dimensions: M={M}, N={N}, K={K}")
    print(f"Reference latency: {ref_latency:.6f} seconds")
    print(f"Reference TFlops: {ref_tflops:.2f}")
