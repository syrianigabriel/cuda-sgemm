#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_NUM 10
#define MIN_NUM -10

float* matrix_multiply_cpu(float* A, float* B, float* C, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float total = 0;
            for (int k = 0; k < N; k++)
            {
                total += A[i*N + k] * B[k*N +j];
            }
            C[i*N + j] = total;
        }
    }
    return C;
}

int main(int argc, char const *argv[])
{
    srand(time(NULL));
    int N;
    if (argc > 1)
    {
        N = atoi(argv[1]);
    }
    else
    {
        printf("Usage: %s <size_of_array> \n", argv[0]);
        return 1;
    }

    float* A = (float*) malloc(N * N * sizeof(float));
    float* B = (float*) malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i*N + j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
            B[i*N + j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
        }
    }
    
    float* C = (float*) malloc(N * N * sizeof(float));

    clock_t start = clock();
    matrix_multiply_cpu(A, B, C, N);
    clock_t end = clock();
    double time_taken_cpu = (double)(end - start) / CLOCKS_PER_SEC;

    printf("%d %f\n", N, time_taken_cpu);

    free(A);
    free(B);
    free(C);

    return 0;
}