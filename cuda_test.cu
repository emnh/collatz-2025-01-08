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

__global__ void convergence_test_iterative(uint64_t *results, uint64_t *powers_of_3, int *cache, uint64_t *S_table, uint64_t S_size, uint64_t n_start, uint64_t n_end) {
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

std::vector<uint64_t> generate_S(int base_bits) {
    std::vector<uint64_t> S;
    const uint64_t max_nL = static_cast<uint64_t>(1) << base_bits;

    for (uint64_t nL = 0; nL < max_nL; ++nL) {
        if (nL % (max_nL / 1000) == 0) {
            std::cout << "Progress: " << std::fixed << std::setprecision(2) << (static_cast<double>(nL) / max_nL) * 100 << "%\r";
            std::cout << std::defaultfloat;
            std::cout.flush();
        }

        uint64_t b = static_cast<uint64_t>(1) << base_bits;
        uint64_t c = nL;

        bool mandatory = true;
        while (b % 2 == 0) {
            if (b % 2 == 0 && c % 2 == 0) {
                b /= 2;
                c /= 2;
            } else if (c % 2 == 1) {
                b *= 3;
                c = 3 * c + 1;
            }

            if (b <= ((static_cast<uint64_t>(1) << base_bits) - 1)) {
                mandatory = false;
                break;
            }
        }

        if (mandatory) {
            S.push_back(nL);
        }
    }
    
    std::cout << std::endl;

    std::cout << "S size: " << S.size() << std::endl;
    return S;
}

void initialize_powers_of_3(uint64_t *powers_of_3_host, int max_power) {
    powers_of_3_host[0] = 1;
    for (int i = 1; i <= max_power; ++i) {
        powers_of_3_host[i] = powers_of_3_host[i - 1] * 3;
    }
}

int main() {
    const uint64_t base_bits = BASE_BITS;
    const uint64_t max_unfiltered = static_cast<uint64_t>(1) << base_bits;
    const uint64_t n_start = 1;
    const uint64_t n_end = max_unfiltered;

    // Generate S table
    auto S_host = generate_S(base_bits);
    int S_size = S_host.size();

    // Host allocations
    uint64_t *results_host = new uint64_t[S_size];
    uint64_t *powers_of_3_host = new uint64_t[65];
    int *cache_host = new int[CACHE_SIZE];

    std::fill(cache_host, cache_host + CACHE_SIZE, -1);
    initialize_powers_of_3(powers_of_3_host, 64);

    // Device allocations
    uint64_t *results_device;
    uint64_t *powers_of_3_device;
    int *cache_device;
    uint64_t *S_device;

    cudaMalloc(&results_device, S_size * sizeof(uint64_t));
    cudaMalloc(&powers_of_3_device, 65 * sizeof(uint64_t));
    cudaMalloc(&cache_device, CACHE_SIZE * sizeof(int));
    cudaMalloc(&S_device, S_size * sizeof(uint64_t));

    cudaMemcpy(powers_of_3_device, powers_of_3_host, 65 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cache_device, cache_host, CACHE_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(S_device, S_host.data(), S_size * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    const int threads_per_block = 256;
    const int num_blocks = (S_size + threads_per_block - 1) / threads_per_block;

    convergence_test_iterative<<<num_blocks, threads_per_block>>>(results_device, powers_of_3_device, cache_device, S_device, S_size, n_start, n_end);
    cudaDeviceSynchronize();

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    cudaMemcpy(results_host, results_device, S_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);

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
    std::cout << "S table size: " << S_size << std::endl;

    return 0;
}
