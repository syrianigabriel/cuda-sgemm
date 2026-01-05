#include <cuda_runtime.h>

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

    // Top-left element of C for this thread's block
    const int row = (by * blockDim.y + ty) * RM;
    const int col = (bx * blockDim.x + tx) * RN;

    // Shared memory tiles
    __shared__ float sh_A[TILE_WIDTH + RM - 1][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH + RN - 1];

    // Number of phases
    const int phases = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int phase = 0; phase < phases; phase++)
    {
        // Load A tile into shared memory
        #pragma unroll
        for (int r = 0; r < RM; r++)
        {
            int a_row = row + r;
            int a_col = phase * TILE_WIDTH + tx;
            if (a_row < M && a_col < K)
                sh_A[ty * RM + r][tx] = A[a_row * K + a_col];
            else
                sh_A[ty * RM + r][tx] = 0.0f;
        }

        // Load B tile into shared memory
        #pragma unroll
        for (int c = 0; c < RN; c++)
        {
            int b_row = phase * TILE_WIDTH + ty;
            int b_col = col + c;
            if (b_row < K && b_col < N)
                sh_B[ty][tx * RN + c] = B[b_row * N + b_col];
            else
                sh_B[ty][tx * RN + c] = 0.0f;
        }

        __syncthreads();

        // Compute tile
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++)
        {
            float regA[RM];
            float regB[RN];

            // Load registers
            #pragma unroll
            for (int r = 0; r < RM; r++)
                regA[r] = sh_A[ty * RM + r][k];
            
            #pragma unroll
            for (int c = 0; c < RN; c++)
                regB[c] = sh_B[k][tx * RN + c];

            // Multiply-accumulate
            #pragma unroll
            for (int r = 0; r < RM; r++)
            {
                #pragma unroll
                for (int c = 0; c < RN; c++)
                    acc[r][c] += regA[r] * regB[c];
            }
        }

        __syncthreads();
    }

    // Write results back to global memory
    #pragma unroll
    for (int r = 0; r < RM; r++)
    {
        #pragma unroll
        for (int c = 0; c < RN; c++)
        {
            int out_row = row + r;
            int out_col = col + c;
            if (out_row < M && out_col < N)
                C[out_row * N + out_col] = acc[r][c];
        }
    }
}

void launch_register_tiled_sgemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    
    // Calculate grid dimensions - FIXED: Proper ceiling division
    int grid_x = (N + (block.x * RN) - 1) / (block.x * RN);
    int grid_y = (M + (block.y * RM) - 1) / (block.y * RM);
    
    // Ensure at least 1 block in each dimension
    grid_x = max(grid_x, 1);
    grid_y = max(grid_y, 1);
    
    dim3 grid(grid_x, grid_y);

    register_tiled_sgemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}