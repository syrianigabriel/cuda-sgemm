#include <cuda_runtime.h>
#include "sgemm_launches.h"

constexpr int TILE_WIDTH = 16;

__global__ void double_buffered_sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{

}

void launch_double_buffered_sgemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    double_buffered_sgemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}