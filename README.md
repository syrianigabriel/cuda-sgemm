# CUDA SGEMM Benchmark

This project allows you to benchmark different implementations of single-precision general matrix-matrix multiplication (SGEMM) on your own NVIDIA GPU using CUDA. You can test naive, coalesced, tiled, register-tiled, and cuBLAS kernels and see how performance scales with matrix size.

## Features

- Multiple SGEMM implementations:
  - **Uncoalesced Naive** — simple row-column multiplication.
  - **Coalesced Naive** — optimized memory access.
  - **Tiled (shared memory)** — block-level tiling for higher reuse.
  - **Register-Tiled** — per-thread register tiling.
  - **cuBLAS** — NVIDIA’s optimized library.
- Accurate GPU timing with CudaTimer.
- Results exported to CSV for plotting and analysis.
- Compute performance in **GFLOPS**.

## Requirements

- NVIDIA GPU with CUDA support.
- CUDA Toolkit installed.
- C++17 compatible compiler.
- Python 3 (for plotting results).

## Building

```bash
# Clone repository
git clone https://github.com/syrianigabriel/cuda-sgemm.git
cd cuda-sgemm
```

Now, make sure to set -arch=sm_XX in the Makefile to match your GPU’s compute capability (use nvidia-smi to check). Then,

```bash
# Build with make
make
```

## Running Benchmarks

You can run benchmarks for a specific matrix size and number of runs:

```bash
./sgemm <matrix_size> [num_runs]
```

- `<matrix_size>` — dimension N for NxN matrices.  
- `[num_runs]` — optional, number of repetitions for averaging.

**Tip:**
To test multiple sizes at once, edit `scripts/run_benchmarks.sh` and modify the `sizes=(...)` array.

## Output

The program prints a CSV line:

```
N,CuBLAS,UncoalescedNaive,CoalescedNaive,Tiled
```

- Each value is the average kernel execution time in milliseconds.  
- You can redirect output to a CSV file:

```bash
./sgemm 1024 5 >> results/t4/sgemm.csv
```

## Sample Results on NVIDIA T4

| N      | CuBLAS   | UncoalescedNaive | CoalescedNaive | Tiled     |
|--------|----------|-----------------|----------------|-----------|
| 128    | 0.255661 | 0.147226        | 0.0471552      | 0.0390144 |
| 256    | 0.301958 | 0.86391         | 0.179942       | 0.128429  |
| 512    | 0.403264 | 6.24232         | 1.17416        | 0.766336  |
| 1024   | 1.1716   | 37.1659         | 6.78271        | 4.26188   |
| 2048   | 3.65073  | 159.023         | 29.7906        | 21.7178   |
| 4096   | 27.9319  | 1170.06         | 297.243        | 195.311   |
| 8192   | 308.78   | 10126.1         | 2640.2         | 1733.93   |
| 16384  | 2106.67  | 102477          | 39194.4        | 16763.9   |
