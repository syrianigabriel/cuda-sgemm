#include <cuda_runtime.h>
#include "sgemm_launches.h"

constexpr int TILE_WIDTH = 16;

__global__ void block_tiled_sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    // Thread information
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Corresponding to entry C[i, j]
    int i = by * blockDim.y + ty;
    int j = bx * blockDim.x + tx;

    // Declare block-level shared memory
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    // Result of C[i,j]
    float sum = 0;

    // Split dot product into tiles
    for (int phase = 0; phase < ceil((float) N/TILE_WIDTH); phase++)
    {
        // A (M x K) tiles move from left to right
        if (i < M && (phase*TILE_WIDTH+tx) < K)
            sh_A[ty][tx] = A[i*K + (phase*TILE_WIDTH+tx)];
        else
            sh_A[ty][tx] = 0.0f;

        // B (K x N) tiles move from top to bottom
        if (j < N && (phase*TILE_WIDTH+ty) < K)
            sh_B[ty][tx] = B[(phase*TILE_WIDTH+ty)*N + j];
        else
            sh_B[ty][tx] = 0.0f;
        
        __syncthreads();

        // Compute partial dot product of matrices in shared memory
        for (int k = 0; k < TILE_WIDTH; k++)
            sum += sh_A[ty][k] * sh_B[k][tx];
        __syncthreads();
    }

    // Fill in value in result matrix
    if (i < M && j < N)
        C[i*N + j] = sum;
}

void launch_block_tiled_sgemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    block_tiled_sgemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}