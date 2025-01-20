latency = [
    # 0.119,
    # 0.375,
    # 2.577,
    # 20.679,
    # 0.526,
    # 0.777,
    # 1.32,
    
    0.105,
    0.373,
    2.807,
    23.516,
    0.51,
    0.817,
    1.498,
]

shapes = [
    [2048, 2048, 2048],
    [4096, 4096, 4096],
    [8192, 8192, 8192],
    [16384, 16384, 16384],
    [8192, 8192, 1024],
    [8192, 8192, 2048],
    [8192, 8192, 4096],
]


for i in range(len(latency)):
    M, N, K = shapes[i]
    total_flops = 2 * M * N * K
    tflop = total_flops / latency[i] * 1e-9
    print(f"Matrix dimensions: M={M}, N={N}, K={K} Reference TFlops: {tflop:.2f}")
