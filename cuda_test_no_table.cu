#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <iomanip>

#define CACHE_SIZE (1ULL << 30)  // 1 GiB Cache Size
#define BASE_BITS 34             // Set to 37 for larger ranges
#define CHUNK_SIZE (1ULL << 30)  // 1 GiB of numbers per chunk
#define MAX_ITERATIONS 1024

__device__ bool is_mandatory(uint64_t nL, int base_bits) {
    __uint128_t b = static_cast<__uint128_t>(1) << base_bits; // Start with b = 2^BASE_BITS
    __uint128_t c = nL;

    while (b % 2 == 0) {
        if (b % 2 == 0 && c % 2 == 0) {
            b /= 2;
            c /= 2;
        } else if (c % 2 == 1) {
            b *= 3;
            c = 3 * c + 1;
        }

        if (b <= ((static_cast<__uint128_t>(1) << base_bits) - 1)) {
            return false;
        }
    }
    return true;
}

__device__ int count_trailing_zeros_64(uint64_t n) {
    return (n == 0) ? 64 : __ffsll(n) - 1;
}

__global__ void direct_convergence_test(uint64_t *results, uint64_t *powers_of_3, int *cache, uint64_t chunk_start, uint64_t chunk_end, int base_bits, unsigned long long *total_processed) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + chunk_start;
    if (idx >= chunk_end) return;

    // Test if the number is mandatory
    if (!is_mandatory(idx, base_bits)) return;

    // Perform convergence test
    __uint128_t n = idx;
    uint64_t n0 = static_cast<uint64_t>(n);
    int delay = 0;
    unsigned int iteration_count = 0;

    while (n > 1) {
        if (n < CACHE_SIZE && cache[static_cast<uint64_t>(n)] != -1) {
            delay += cache[static_cast<uint64_t>(n)];
            break;
        }

        if (iteration_count >= MAX_ITERATIONS) {
            printf("Exceeded maximum iterations\n");
            break;
        }

        n = n + 1;
        int a = count_trailing_zeros_64(static_cast<uint64_t>(n));
        n >>= a;
        n *= powers_of_3[a];
        n = n - 1;
        int b = count_trailing_zeros_64(static_cast<uint64_t>(n));
        n >>= b;

        delay += a + b;
        iteration_count++;
    }

    if (n0 < CACHE_SIZE) {
        cache[n0] = delay;
    }

    // Increment the total processed count
    atomicAdd(total_processed, 1);
}

void initialize_powers_of_3(uint64_t *powers_of_3_host, int max_power) {
    powers_of_3_host[0] = 1;
    for (int i = 1; i <= max_power; ++i) {
        powers_of_3_host[i] = powers_of_3_host[i - 1] * 3;
    }
}

int main() {
    const int base_bits = BASE_BITS;
    const uint64_t max_nL = static_cast<uint64_t>(1) << base_bits;
    const uint64_t chunk_size = CHUNK_SIZE;

    // Host allocations
    uint64_t *powers_of_3_host = new uint64_t[65];
    int *cache_host = new int[CACHE_SIZE];
    unsigned long long total_processed_host = 0;

    std::fill(cache_host, cache_host + CACHE_SIZE, -1);
    initialize_powers_of_3(powers_of_3_host, 64);

    // Device allocations
    uint64_t *powers_of_3_device;
    int *cache_device;
    unsigned long long *total_processed_device;

    cudaMalloc(&powers_of_3_device, 65 * sizeof(uint64_t));
    cudaMalloc(&cache_device, CACHE_SIZE * sizeof(int));
    cudaMalloc(&total_processed_device, sizeof(unsigned long long));
    cudaMemset(total_processed_device, 0, sizeof(unsigned long long));

    cudaMemcpy(powers_of_3_device, powers_of_3_host, 65 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cache_device, cache_host, CACHE_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Timer for the entire process
    auto start_total = std::chrono::high_resolution_clock::now();

    // Process in chunks
    for (uint64_t chunk_start = 0; chunk_start < max_nL; chunk_start += chunk_size) {
        uint64_t chunk_end = std::min(chunk_start + chunk_size, max_nL);

        const int threads_per_block = 256;
        const int num_blocks = ((chunk_end - chunk_start) + threads_per_block - 1) / threads_per_block;

        // Timer for chunk processing
        auto start_chunk = std::chrono::high_resolution_clock::now();

        // Launch the kernel
        direct_convergence_test<<<num_blocks, threads_per_block>>>(nullptr, powers_of_3_device, cache_device, chunk_start, chunk_end, base_bits, total_processed_device);
        cudaDeviceSynchronize();

        auto end_chunk = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_chunk = end_chunk - start_chunk;

        std::cout << "Processed chunk [" << chunk_start << ", " << chunk_end << ") in "
                  << elapsed_chunk.count() << " seconds." << std::endl;
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_total = end_total - start_total;

    // Copy total processed count back to host
    cudaMemcpy(&total_processed_host, total_processed_device, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Output performance report
    double numbers_per_second = total_processed_host / elapsed_total.count();
    std::cout << "\nPerformance Report:\n";
    std::cout << "-------------------\n";
    std::cout << "Total processed: " << total_processed_host << " numbers\n";
    std::cout << "Total time: " << elapsed_total.count() << " seconds\n";
    std::cout << "Processing rate: " << numbers_per_second << " numbers/second\n";

    // Clean up
    cudaFree(powers_of_3_device);
    cudaFree(cache_device);
    cudaFree(total_processed_device);

    delete[] powers_of_3_host;
    delete[] cache_host;

    return 0;
}
