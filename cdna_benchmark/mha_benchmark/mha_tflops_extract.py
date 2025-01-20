import os
import re

shapes = [
    [1, 12, 1024, 64, True],
    [1, 8, 512, 64, True],
    [1, 12, 512, 64, True],
    [1, 16, 512, 64, True],
    [1, 32, 4096, 128, True],
    [1, 40, 4096, 128, True],
    [1, 64, 4096, 128, True],
    [1, 16, 2048, 64, True],
    [1, 40, 2048, 128, True],
    [1, 96, 2048, 128, True],
    [1, 6, 1024, 64, True],
    [1, 12, 1024, 64, True],
    [1, 16, 1024, 64, True],
    [64, 12, 1024, 64, True],
    [64, 8, 512, 64, True],
    [64, 12, 512, 64, True],
    [64, 16, 512, 64, True],
    [64, 32, 4096, 128, True],
    [64, 40, 4096, 128, True],
    [64, 64, 4096, 128, True],
    [64, 16, 2048, 64, True],
    [64, 40, 2048, 128, True],
    [64, 96, 2048, 128, True],
    [64, 6, 1024, 64, True],
    [64, 12, 1024, 64, True],
    [64, 16, 1024, 64, True],
]

def extract_tflops(data):
    # Use regular expressions to match Best TFlops and Ref TFlops
    best_tflops_pattern = r"Best TFlops:\s*([\d\.]+)"
    ref_tflops_pattern = r"Ref TFlops:\s*([\d\.]+)"

    # Search for matches in the input string
    best_tflops_match = re.search(best_tflops_pattern, data)
    ref_tflops_match = re.search(ref_tflops_pattern, data)

    # Convert the extracted values to float
    best_tflops = float(best_tflops_match.group(1)) if best_tflops_match else None
    ref_tflops = float(ref_tflops_match.group(1)) if ref_tflops_match else None

    return best_tflops, ref_tflops

best_tflops_map = []
ref_tflops_map = []
for b, h, n_ctx, d_head, casual in shapes:
    casual = "true" if casual else "false"
    path = f"./run_tilelang_mha_b{b}_h_{h}_n_ctx_{n_ctx}_d_head_{d_head}_casual_{casual}.log"
    if not os.path.exists(path):
        key = f"{b}_{h}_{n_ctx}_{d_head}_{casual}"
        best_tflops_map.append((key, 0.0))        
        continue
    with open(path, "r") as f:
        data = f.read()
        best_tflops, ref_tflops = extract_tflops(data)
    key = f"{b}_{h}_{n_ctx}_{d_head}_{casual}"
    best_tflops_map.append((key, best_tflops))        

for key, best in best_tflops_map:

    if best is None:
        best = 0.0
    print(f"Shape: {key}, Best TFlops: {best}")
