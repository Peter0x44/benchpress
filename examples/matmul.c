#include <stdint.h>

#define ITERATIONS 1000000000LL
#define WARMUP_ITERATIONS 1000000LL

void init_matrix(float *mat) {
    for (int i = 0; i < 16; i++) {
        mat[i] = (float)(i + 1);
    }
}

// BENCHFUNC: Matrix multiplication to be compiled with different compilers
BENCHFUNC void mat4x4_mul(float * restrict A, float * restrict B, float * restrict C) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += A[i * 4 + k] * B[k * 4 + j];
            }
            C[i * 4 + j] = sum;
        }
    }
}

// WARMUP: Warmup before benchmarking
WARMUP void run_benchmark_warmup(void) {
    __attribute__((aligned(32))) float A[16];
    __attribute__((aligned(32))) float B[16];
    __attribute__((aligned(32))) float C[16];
    
    init_matrix(A);
    init_matrix(B);
    
    for (int64_t i = 0; i < WARMUP_ITERATIONS; i++) {
        mat4x4_mul(A, B, C);
    }
}

// BENCHMARK: Run benchmark iterations
BENCHMARK void run_benchmark(void) {
    __attribute__((aligned(32))) float A[16];
    __attribute__((aligned(32))) float B[16];
    __attribute__((aligned(32))) float C[16];
    
    init_matrix(A);
    init_matrix(B);
    
    for (int64_t i = 0; i < ITERATIONS; i++) {
        mat4x4_mul(A, B, C);
    }
}
