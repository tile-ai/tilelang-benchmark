import re

shapes = [
    [2048, 2048, 2048],
    [4096, 4096, 4096],
    [8192, 8192, 8192],
    [16384, 16384, 16384],
    [8192, 8192, 1024],
    [8192, 8192, 2048],
    [8192, 8192, 4096],
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

best_tflops_map = {}
ref_tflops_map = {}
for m, n, k in shapes:
    with open(f"/home/aiscuser/lei/tl_benchmark/gemm_benchmark/run_gemm_tilelang_{m}_{n}_{k}.log", "r") as f:
        data = f.read()
        best_tflops, ref_tflops = extract_tflops(data)
        key = f"{m}_{n}_{k}"
        best_tflops_map[key] = best_tflops
        ref_tflops_map[key] = ref_tflops
        
for key in best_tflops_map:
    print(f"Shape: {key}, Best TFlops: {best_tflops_map[key]}, Ref TFlops: {ref_tflops_map[key]}")
