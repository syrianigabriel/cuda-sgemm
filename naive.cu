#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_NUM 10
#define MIN_NUM -10

using namespace std;

void cudaCheck(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%d) at %s:%d\n",
                cudaGetErrorString(err), err, file, line);
        exit(err);
    }
}

#define CUDA_CHECK(call) cudaCheck(call, __FILE__, __LINE__)

__global__ void sq_mat_mul_kernel_naive(float* A, float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // thread is within bounds
    if (row < N && col < N)
    {
        float total = 0;
        for (int k = 0; k < N; k++)
        {
            total += A[row * N + k] * B[k * N + col];
        }
        C[row*N + col] = total;
    }
}

int main(int argc, char const *argv[])
{
    if (argc < 4) {
        cout << "Usage: " << argv[0] << " <matrix_size> <block_x> <block_y>\n";
        return 1;
    }

    int N = atoi(argv[1]);
    int block_x = atoi(argv[2]);
    int block_y = atoi(argv[3]);


    // initialize matrices A, B, C of size NxN
    float* A = (float*) malloc(N * N * sizeof(float));
    float* B = (float*) malloc(N * N * sizeof(float));
    float* C = (float*) malloc(N * N * sizeof(float));

    // fill matrices with random values
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i*N + j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
            B[i*N + j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
        }
    }

    // initalize device pointers (VRAM)
    float* d_A;
    float* d_B;
    float* d_C;

    // allocate memory on device and store addresses in d_X ptrs
    CUDA_CHECK(cudaMalloc((void**) &d_A, N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &d_B, N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &d_C, N*N*sizeof(float)));

    // copy A and B matrices from host to device
    CUDA_CHECK(cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice));

    // block size (x, y, z)
    dim3 dimBlock(block_x, block_y, 1);
    dim3 dimGrid((N + block_x - 1)/block_x, (N + block_y - 1)/block_y);

    float time;
    cudaEvent_t start;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, 0));
    // execute on GPU
    sq_mat_mul_kernel_naive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(end, 0));

    // wait for all kernels to finish
    CUDA_CHECK(cudaEventSynchronize(end));

    cudaEventElapsedTime(&time, start, end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));

    // copy result back to host memory
    CUDA_CHECK(cudaMemcpy(C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost));
    cout << N << " " << time << "\n";

    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}