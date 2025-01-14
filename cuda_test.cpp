#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CACHE_SIZE (1 << 20) // Example cache size
#define MAX_ITERATIONS 1024
#define BASE_BITS 20

__device__ int count_trailing_zeros_64(uint64_t n) {
    return (n == 0) ? 64 : __ffsll(n) - 1;
}

__global__ void convergence_test_iterative(uint64_t *results, uint64_t *powers_of_3, int *cache, int n_start, int n_end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = n_start + idx;

    if (n >= n_end) return;

    uint64_t n0 = n;
    int delay = 0;
    unsigned int iteration_count = 0;

    while (n > 1) {
        if (n < CACHE_SIZE && cache[n] != -1) {
            delay += cache[n];
            break;
        }

        if (iteration_count >= MAX_ITERATIONS) {
            printf("Exceeded maximum iterations\n");
            break;
        }

        n = n + 1;
        int a = count_trailing_zeros_64(n);
        n >>= a;
        n *= powers_of_3[a];
        n = n - 1;
        int b = count_trailing_zeros_64(n);
        n >>= b;

        delay += a + b;
        iteration_count++;
    }

    if (n0 < CACHE_SIZE) {
        cache[n0] = delay;
    }
    results[idx] = delay;
}

void initialize_powers_of_3(uint64_t *powers_of_3_host, int max_power) {
    powers_of_3_host[0] = 1;
    for (int i = 1; i <= max_power; ++i) {
        powers_of_3_host[i] = powers_of_3_host[i - 1] * 3;
    }
}

int main() {
    const int n_start = 1;
    const int n_end = (1 << BASE_BITS);
    const int num_elements = n_end - n_start;

    // Host allocations
    uint64_t *results_host = new uint64_t[num_elements];
    uint64_t *powers_of_3_host = new uint64_t[65]; // Powers of 3 up to 3^64
    int *cache_host = new int[CACHE_SIZE];
    
    std::fill(cache_host, cache_host + CACHE_SIZE, -1);
    initialize_powers_of_3(powers_of_3_host, 64);

    // Device allocations
    uint64_t *results_device;
    uint64_t *powers_of_3_device;
    int *cache_device;

    cudaMalloc(&results_device, num_elements * sizeof(uint64_t));
    cudaMalloc(&powers_of_3_device, 65 * sizeof(uint64_t));
    cudaMalloc(&cache_device, CACHE_SIZE * sizeof(int));

    cudaMemcpy(powers_of_3_device, powers_of_3_host, 65 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cache_device, cache_host, CACHE_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    const int threads_per_block = 256;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    convergence_test_iterative<<<num_blocks, threads_per_block>>>(results_device, powers_of_3_device, cache_device, n_start, n_end);

    cudaMemcpy(results_host, results_device, num_elements * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(results_device);
    cudaFree(powers_of_3_device);
    cudaFree(cache_device);

    delete[] results_host;
    delete[] powers_of_3_host;
    delete[] cache_host;

    std::cout << "Computation complete." << std::endl;
    return 0;
}
