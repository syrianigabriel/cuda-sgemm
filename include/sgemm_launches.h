#pragma once

void launch_coalesced_naive_sgemm(const float* A, const float* B, float* C, int M, int N, int K);

void launch_block_tiled_sgemm(const float* A, const float* B, float* C, int M, int N, int K);

void launch_register_tiled_sgemm(const float* A, const float* B, float* C, int M, int N, int K);

void launch_uncoalesced_naive_sgemm(const float* A, const float* B, float* C, int M, int N, int K);