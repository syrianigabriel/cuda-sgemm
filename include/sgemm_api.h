#pragma once

enum class SgemmEnum
{
    CoalescedNaive,
    UncoalescedNaive,
    Tiled,
    RegisterTiled,
    CuBLAS,
    CPU
};

void sgemm(const float* A, const float* B, float* C, int M, int N, int K, SgemmEnum type);