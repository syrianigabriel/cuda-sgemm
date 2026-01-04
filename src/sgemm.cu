#include "sgemm_api.h"
#include "sgemm_launches.h"
#include <cublas_v2.h>
#include <stdexcept>

static void cpu_sgemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
        {
            float sum = 0;
            for (int k = 0; k < K; ++k)
                sum += A[i*K + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}

void sgemm(const float* A, const float* B, float* C, int M, int N, int K, SgemmEnum type)
{
    if (!A || !B || !C)
    {
        throw std::runtime_error("Null pointer passed to SGEMM!");
    }

    switch (type)
    {
        case SgemmEnum::CoalescedNaive:
            launch_coalesced_naive_sgemm(A, B, C, M, N, K);
            break;
        case SgemmEnum::UncoalescedNaive:
            launch_uncoalesced_naive_sgemm(A, B, C, M, N, K);
            break;
        case SgemmEnum::Tiled:
            launch_block_tiled_sgemm(A, B, C, M, N, K);
            break;
        case SgemmEnum::CPU:
            cpu_sgemm(A, B, C, M, N, K);
            break;
        case SgemmEnum::CuBLAS:
        {
            cublasHandle_t handle;
            cublasCreate(&handle);
            const float alpha = 1.0f;
            const float beta = 0.0f;

            cublasSgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K,
                        &alpha,
                        B, N,
                        A, K,
                        &beta,
                        C, N);

            cublasDestroy(handle);
            break;
        }
        default:
            throw std::runtime_error("Unsupported SGEMM kernel!");
    }
}