#include "sgemm_api.h"
#include "timer.h"
#include <stdexcept>
#include <string>
#include <iostream>
#include <vector>
#include "helpers.h"

#define MAX_NUM 10
#define MIN_NUM -10
#define TILE_WIDTH 16

int main(int argc, char const *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> [num_runs]" << std::endl;
        return 1;
    }

    int num_runs = (argc >= 3) ? std::atoi(argv[2]) : 1;

    int N = atoi(argv[1]);

    // initialize matrices A, B, C of size NxN
    std::vector<float> A(N*N), B(N*N), C(N*N);

    // fill matrices with random values
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i*N + j] = MIN_NUM + static_cast<float>(rand()) / RAND_MAX * (MAX_NUM - MIN_NUM);
            B[i*N + j] = MIN_NUM + static_cast<float>(rand()) / RAND_MAX * (MAX_NUM - MIN_NUM);
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
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));

    CudaTimer timer;
    float cublas_total = 0, tiled_total = 0, coalesced_naive_total = 0, uncoalesced_naive_total = 0, register_tiled_total = 0;

    for (int run = 0; run < num_runs; run++) 
    {
        timer.start();
        sgemm(d_A, d_B, d_C, N, N, N, SgemmEnum::CuBLAS);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        cublas_total += timer.stop();

        timer.start();
        sgemm(d_A, d_B, d_C, N, N, N, SgemmEnum::UncoalescedNaive);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        uncoalesced_naive_total += timer.stop();

        timer.start();
        sgemm(d_A, d_B, d_C, N, N, N, SgemmEnum::CoalescedNaive);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        coalesced_naive_total += timer.stop();

        timer.start();
        sgemm(d_A, d_B, d_C, N, N, N, SgemmEnum::Tiled);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        tiled_total += timer.stop();

        // timer.start();
        // sgemm(d_A, d_B, d_C, N, N, N, SgemmEnum::RegisterTiled);
        // CUDA_CHECK(cudaGetLastError());
        // register_tiled_total += timer.stop();
    }

    std::cout << N << ","
          << cublas_total / num_runs << ","
          << uncoalesced_naive_total / num_runs << ","
          << coalesced_naive_total / num_runs << ","
          << tiled_total / num_runs
        //   << register_tiled_total / num_runs
          << std::endl; 

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}