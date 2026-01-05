#!/bin/bash

# Create results directory for this GPU
mkdir -p results/rtx3080

# Compile
make

# List of matrix sizes
sizes=(250 500 1000 1500 2000 3000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000)

# Number of runs per size
num_runs=1

# Output file
output_file="results/rtx3080/sgemm.csv"

# Add header
echo "N,CuBLAS,UncoalescedNaive,CoalescedNaive,Tiled" > "$output_file"

# Loop over matrix sizes
for N in "${sizes[@]}"; do
    echo "Running N=$N for $num_runs runs..."
    ./sgemm $N $num_runs >> "$output_file"
done

# Clean build artifacts
make clean