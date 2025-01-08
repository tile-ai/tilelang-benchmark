#!/bin/bash

# Create a directory to store logs
mkdir -p benchmark_logs

# Define all shape combinations, keeping only the numerical values
shapes=(
    "1 16384 16384"
    "16 16384 16384"
    "32 16384 16384"
    "64 16384 16384"
    "128 16384 16384"
    "256 16384 16384"
    "1 8192 8192"
    "16 8192 8192"
    "32 8192 8192"
    "64 8192 8192"
    "128 8192 8192"
    "256 8192 8192"
    "1 28672 8192"
    "16 28672 8192"
    "32 28672 8192"
    "64 28672 8192"
    "128 28672 8192"
    "256 28672 8192"
    "1 8192 28672"
    "16 8192 28672"
    "32 8192 28672"
    "64 8192 28672"
    "128 8192 28672"
    "256 8192 28672"
)

# Define all dtype combinations
dtypes=(
    "float16 int4 float16 float16"
    "int8 int2 int32 int32"
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
        echo "Running benchmark for shape: m=${m}, n=${n}, k=${k}, A_dtype=${A_dtype}, W_dtype=${W_dtype}, out_dtype=${out_dtype}, accum_dtype=${accum_dtype}"
        
        # Construct the command to run the benchmark script
        cmd="python ./benchmark_bitblas_matmul.py --M ${m} --N ${n} --K ${k} --A_dtype ${A_dtype} --W_dtype ${W_dtype} --out_dtype ${out_dtype} --accum_dtype ${accum_dtype}"
        echo "Running command: $cmd"
        
        # Execute the command and save output to the log file
        bash -c "$cmd 2>&1 | tee ${log_file}"
        
        # Confirm log file creation
        echo "Logs written to ${log_file}"
    done
done
