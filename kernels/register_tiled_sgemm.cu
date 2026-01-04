#include <cuda_runtime.h>
#include "sgemm_launches.h"

constexpr int TILE_WIDTH = 16;
constexpr int RM = 2;
constexpr int RN = 2;

__global__ void register_tiled_sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K)
{
    // Block and thread info
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Per-thread register block
    float acc[RM][RN] = {0.0f};

    // Top-left element of C for this thread’s block
    const int row = (by * blockDim.y + ty) * RM;
    const int col = (bx * blockDim.x + tx) * RN;

    // Shared memory tiles
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    // Number of phases
    const int phases = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < phases; phase++)
    {
        // Load A tile into shared memory
        for (int r = 0; r < RM; r++)
        {
            int a_row = row + r;
            int a_col = phase * TILE_WIDTH + tx;
            sh_A[ty + r][tx] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }

        // Load B tile into shared memory
        for (int c = 0; c < RN; c++)
        {
            int b_row = phase * TILE_WIDTH + ty;
            int b_col = col + c;
            sh_B[ty][tx + c] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }

        __syncthreads();

        // Compute tile
        for (int k = 0; k < TILE_WIDTH; k++)
        {
            float regA[RM];
            float regB[RN];

            // Load registers
            for (int r = 0; r < RM; r++)
                regA[r] = sh_A[ty + r][k];

            for (int c = 0; c < RN; c++)
                regB[c] = sh_B[k][tx + c];

            // Multiply-accumulate
            for (int r = 0; r < RM; r++)
                for (int c = 0; c < RN; c++)
                    acc[r][c] += regA[r] * regB[c];
        }

        __syncthreads();
    }

    // Write results back to global memory
    for (int r = 0; r < RM; r++)
        for (int c = 0; c < RN; c++)
            if (row + r < M && col + c < N)
                C[(row + r) * N + (col + c)] = acc[r][c];
}

void launch_register_tiled_sgemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(
        (N + block.x * RN - 1) / (block.x * RN),
        (M + block.y * RM - 1) / (block.y * RM)
    );

    register_tiled_sgemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
