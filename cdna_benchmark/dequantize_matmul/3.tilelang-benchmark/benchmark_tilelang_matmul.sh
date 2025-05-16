#!/bin/bash

# Create a directory to store logs
mkdir -p benchmark_logs

# Define all shape combinations, keeping only the numerical values
shapes=(
    "1 1024 8192"
    "1 8192 8192"
    "1 28672 8192"
    "1 8192 28672"
)

# Define all dtype combinations
dtypes=(
    "float16 float16 float16 float32"
    # "int8 int8 int32 int32"
)

# Iterate through all shape combinations
for shape in "${shapes[@]}"; do
    # Split shape into m, n, k values
    read -r m n k <<< "$shape"

    # Iterate through all dtype combinations
    for dtype_combo in "${dtypes[@]}"; do
        # Split dtype combination into A_dtype, W_dtype, out_dtype, accum_dtype
        read -r A_dtype W_dtype out_dtype accum_dtype <<< "$dtype_combo"

        # Generate a log file name based on shape values, dtypes, and timestamp
        log_file="benchmark_logs/benchmark_${m}_${n}_${k}_${A_dtype}_${W_dtype}_${out_dtype}_${accum_dtype}.log"

        # Display current shape and dtype combination being processed
        echo "Running benchmark for shape: m=${m}, n=${n}, k=${k}"
        
        # Construct the command to run the benchmark script
        # cmd="python ./benchmark_tilelang_matmul_fp16xfp4.py --m ${m} --n ${n} --k ${k}"
        cmd="python ./benchmark_tilelang_matmul_fp16xint4.py --m ${m} --n ${n} --k ${k}"
        echo "Running command: $cmd"
        
        # Execute the command and save output to the log file
        bash -c "$cmd 2>&1 | tee ${log_file}"
        
        # Confirm log file creation
        echo "Logs written to ${log_file}"
    done
done
