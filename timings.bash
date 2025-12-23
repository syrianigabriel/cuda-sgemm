#!/bin/bash

for N in $(seq 100 200 1000); do
    ./cpu_matrix_mul $N
done > cpu_timings.txt

for N in $(seq 2000 1000 10000); do
    ./cpu_matrix_mul $N
done >> cpu_timings.txt