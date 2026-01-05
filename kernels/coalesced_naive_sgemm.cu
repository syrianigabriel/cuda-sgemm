#include <cuda_runtime.h>
#include "sgemm_launches.h"

constexpr int TILE_WIDTH = 16;

__global__ void coalesced_naive_sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // thread is within bounds
    if (row < M && col < N)
    {
        float total = 0;
        for (int k = 0; k < N; k++)
        {
            total += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = total;
    }
}

void launch_coalesced_naive_sgemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    coalesced_naive_sgemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}