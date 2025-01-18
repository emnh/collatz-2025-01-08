#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <cstdint>

#define CACHE_SIZE (1 << 20)
#define MAX_ITERATIONS 1024
#define BASE_BITS 32

__device__ int count_trailing_zeros_64(uint64_t n) {
    return (n == 0) ? 64 : __ffsll(n) - 1;
}

__device__ bool is_mandatory(uint64_t nL, int base_bits) {
    uint64_t b = static_cast<uint64_t>(1) << base_bits; // Start with b = 2^d
    uint64_t c = nL;

    while (b % 2 == 0) {
        if (b % 2 == 0 && c % 2 == 0) {
            b /= 2;
            c /= 2;
        } else if (c % 2 == 1) {
            b *= 3;
            c = 3 * c + 1;
        }

        if (b <= ((static_cast<uint64_t>(1) << base_bits) - 1)) {
            return false;
        }
    }
    return true;
}

__global__ void generate_S_gpu(uint64_t *S_table, int *valid_count, int base_bits, uint64_t max_nL) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_nL) return;

    // Check if this `nL` is mandatory
    if (is_mandatory(idx, base_bits)) {
        int pos = atomicAdd(valid_count, 1);
        S_table[pos] = idx;
    }
}

std::vector<uint64_t> generate_S_on_gpu(int base_bits) {
    const uint64_t max_nL = static_cast<uint64_t>(1) << base_bits;

    // Host allocations
    std::vector<uint64_t> S_host(max_nL); // Temporary buffer
    int valid_count_host = 0;

    // Device allocations
    uint64_t *S_device;
    int *valid_count_device;
    cudaMalloc(&S_device, max_nL * sizeof(uint64_t));
    cudaMalloc(&valid_count_device, sizeof(int));
    cudaMemset(valid_count_device, 0, sizeof(int));

    // Launch kernel
    const int threads_per_block = 256;
    const int num_blocks = (max_nL + threads_per_block - 1) / threads_per_block;

    generate_S_gpu<<<num_blocks, threads_per_block>>>(S_device, valid_count_device, base_bits, max_nL);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(&valid_count_host, valid_count_device, sizeof(int), cudaMemcpyDeviceToHost);
    S_host.resize(valid_count_host); // Resize to the actual number of valid entries
    cudaMemcpy(S_host.data(), S_device, valid_count_host * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(S_device);
    cudaFree(valid_count_device);

    return S_host;
}

__global__ void convergence_test_iterative(uint64_t *results, uint64_t *powers_of_3, int *cache, uint64_t *S_table, int S_size, int n_start, int n_end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= S_size) return;  // Ensure threads don't exceed the S table size

    uint64_t mL = S_table[idx];
    uint64_t n = (static_cast<uint64_t>(n_start) << BASE_BITS) + mL;

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
    const int base_bits = BASE_BITS;

    // Generate S table using GPU
    auto start_S = std::chrono::high_resolution_clock::now();
    auto S = generate_S_on_gpu(base_bits);
    auto end_S = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_S = end_S - start_S;

    std::cout << "Generated S table with size: " << S.size() << " in "
              << elapsed_S.count() << " seconds." << std::endl;

    const uint64_t max_unfiltered = static_cast<uint64_t>(1) << base_bits;

    // Host allocations
    uint64_t *results_host = new uint64_t[S.size()];
    uint64_t *powers_of_3_host = new uint64_t[65];
    int *cache_host = new int[CACHE_SIZE];

    std::fill(cache_host, cache_host + CACHE_SIZE, -1);
    initialize_powers_of_3(powers_of_3_host, 64);

    // Device allocations
    uint64_t *results_device;
    uint64_t *powers_of_3_device;
    int *cache_device;
    uint64_t *S_device;

    cudaMalloc(&results_device, S.size() * sizeof(uint64_t));
    cudaMalloc(&powers_of_3_device, 65 * sizeof(uint64_t));
    cudaMalloc(&cache_device, CACHE_SIZE * sizeof(int));
    cudaMalloc(&S_device, S.size() * sizeof(uint64_t));

    cudaMemcpy(powers_of_3_device, powers_of_3_host, 65 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cache_device, cache_host, CACHE_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(S_device, S.data(), S.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    const int threads_per_block = 256;
    const int num_blocks = (S.size() + threads_per_block - 1) / threads_per_block;

    convergence_test_iterative<<<num_blocks, threads_per_block>>>(results_device, powers_of_3_device, cache_device, S_device, S.size(), 1, max_unfiltered);
    cudaDeviceSynchronize();

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    cudaMemcpy(results_host, results_device, S.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(results_device);
    cudaFree(powers_of_3_device);
    cudaFree(cache_device);
    cudaFree(S_device);

    delete[] results_host;
    delete[] powers_of_3_host;
    delete[] cache_host;

    // Output benchmark results
    double numbers_per_second = max_unfiltered / elapsed.count();
    std::cout << "Processed " << max_unfiltered << " numbers (unfiltered) in "
              << elapsed.count() << " seconds." << std::endl;
    std::cout << "Processing rate: " << numbers_per_second << " numbers/second." << std::endl;
    std::cout << "Filtered S table size: " << S.size() << std::endl;

    return 0;
}
