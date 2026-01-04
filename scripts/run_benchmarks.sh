#!/bin/bash

# Create results directory for this GPU
mkdir -p results/rtx3080

# Compile
make

# List of matrix sizes
sizes=(256 512 1024 2048 4096 8192 16384)

# Number of runs per size
num_runs=5

# Output file
output_file="results/rtx3080/sgemm.csv"

# Add header
echo "N,CuBLAS,UncoalescedNaive,CoalescedNaive,Tiled,RegisterTiled" > "$output_file"

# Loop over matrix sizes
for N in "${sizes[@]}"; do
    echo "Running N=$N for $num_runs runs..."
    ./sgemm $N $num_runs >> "$output_file"
done

# Clean build artifacts
make clean