#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <filesystem> // For file existence check

#define CACHE_SIZE (1ULL << 2)  // 1 GiB Cache Size
#define BASE_BITS 36 //36             // Set to 37 for larger ranges
#define BASE_TABLE_BITS 28
#define CHUNK_SIZE (1ULL << 30)  // 1 GiB of numbers per chunk
// #define MAX_ITERATIONS 1024

void saveArrayToFile(const std::string& file_path, const uint64_t* h_array, int size) {
    std::ofstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for writing: " << file_path << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(h_array), size * sizeof(uint64_t));
    file.close();
    std::cout << "Array saved to " << file_path << std::endl;
}

bool loadArrayFromFile(const std::string& file_path, uint64_t* h_array, int size) {
    if (!std::filesystem::exists(file_path)) {
        return false; // File does not exist
    }

    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for reading: " << file_path << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(h_array), size * sizeof(uint64_t));
    file.close();
    std::cout << "Array loaded from " << file_path << std::endl;
    return true;
}

std::string uint128_to_string(__uint128_t value) {
    // Split the 128-bit integer into two 64-bit parts
    const __uint128_t base = 10;
    std::string result;
    do {
        __uint128_t remainder = value % base;
        value /= base;
        result = static_cast<char>('0' + remainder) + result;
    } while (value != 0);
    return result;
}

void fill_table_item(uint64_t *B_table, uint64_t *C_table, uint64_t nL, int base_bits) {
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

            if (b >= ((__uint128_t) 1 << (__uint128_t) 64)) {
                // std::cout << "Overflow detected in b: " << uint128_to_string(b) << std::endl;
                printf("Overflow detected in b\n");
            }

            if (c >= ((__uint128_t) 1 << (__uint128_t) 64)) {
                // std::cout << "Overflow detected in c: " << uint128_to_string(c) << std::endl;
                printf("Overflow detected in c\n");
            }

            B_table[nL] = (uint64_t) b;
            C_table[nL] = (uint64_t) c;

            return;
        }
    }

    if (b >= ((__uint128_t) 1 << (__uint128_t) 64)) {
        // std::cout << "Overflow detected in b: " << uint128_to_string(b) << std::endl;
        printf("Overflow detected in b\n");
    }

    if (c >= ((__uint128_t) 1 << (__uint128_t) 64)) {
        // std::cout << "Overflow detected in c: " << uint128_to_string(c) << std::endl;
        printf("Overflow detected in c\n");
    }

    B_table[nL] = (uint64_t) b;
    C_table[nL] = (uint64_t) c;
}

void fill_table(uint64_t *B_table, uint64_t *C_table, int base_bits) {
    const uint64_t size = 1 << base_bits;
    for (uint64_t i = 0; i < size; ++i) {
        fill_table_item(B_table, C_table, i, base_bits);
    }
}

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

        if (b <= ((static_cast<__uint128_t>(1) << base_bits) - 1)) {
            return false;
        }
    }

    return true;
}

__global__ void generate_S_gpu_chunk(uint64_t *S_table, int *valid_count, uint64_t chunk_start, uint64_t chunk_end, int base_bits) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + chunk_start;
    if (idx >= chunk_end) return;

    if (is_mandatory(idx, base_bits)) {
        int pos = atomicAdd(valid_count, 1);
        S_table[pos] = idx;
    }
}

std::vector<uint64_t> generate_S_on_gpu(int base_bits) {
    const uint64_t max_nL = static_cast<uint64_t>(1) << base_bits;
    const uint64_t chunk_size = CHUNK_SIZE;

    std::vector<uint64_t> S_host;
    uint64_t *S_device;
    int *valid_count_device;
    cudaMalloc(&S_device, chunk_size * sizeof(uint64_t)); // Allocate memory for a single chunk
    cudaMalloc(&valid_count_device, sizeof(int));

    for (uint64_t chunk_start = 0; chunk_start < max_nL; chunk_start += chunk_size) {
        uint64_t chunk_end = std::min(chunk_start + chunk_size, max_nL);

        // Reset valid count on the device
        cudaMemset(valid_count_device, 0, sizeof(int));

        // Launch kernel
        const int threads_per_block = 256;
        const int num_blocks = ((chunk_end - chunk_start) + threads_per_block - 1) / threads_per_block;

        generate_S_gpu_chunk<<<num_blocks, threads_per_block>>>(S_device, valid_count_device, chunk_start, chunk_end, base_bits);
        cudaDeviceSynchronize();

        // Copy results back to host
        int valid_count_host = 0;
        cudaMemcpy(&valid_count_host, valid_count_device, sizeof(int), cudaMemcpyDeviceToHost);

        std::vector<uint64_t> S_chunk(valid_count_host);
        cudaMemcpy(S_chunk.data(), S_device, valid_count_host * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        // Append chunk results to S_host
        S_host.insert(S_host.end(), S_chunk.begin(), S_chunk.end());
    }

    // Clean up
    cudaFree(S_device);
    cudaFree(valid_count_device);

    return S_host;
}

__device__ int count_trailing_zeros_64(uint64_t n) {
    return (n == 0) ? 64 : __ffsll(n) - 1;
}

__device__ __uint128_t method_A(const __uint128_t n, uint64_t *powers_of_3) {
    const __uint128_t n1 = n + 1;
    int a = count_trailing_zeros_64(static_cast<uint64_t>(n1));
    const __uint128_t n2 = n1 >> a;
    const __uint128_t n3 = powers_of_3[a] * n2;
    const __uint128_t n4 = n3 - 1;
    int b = count_trailing_zeros_64(static_cast<uint64_t>(n4));
    const __uint128_t n5 = n4 >> b;
    return n5;
}

__device__ __uint128_t method_B(const __uint128_t n, uint64_t *B_table, uint64_t *C_table) {
    const __uint128_t mask = (1ULL << BASE_TABLE_BITS) - 1; // Mask for BASE_TABLE_BITS
    const __uint128_t nL = n & mask;                           // Extract LSB
    const __uint128_t nH = (n >> (128 - BASE_TABLE_BITS)) & mask; // Extract MSB
    const __uint128_t bT = B_table[nL];
    const __uint128_t cT = C_table[nL];
    const __uint128_t n2 = bT * nH + cT;
    return n2;
}

__global__ void convergence_test_iterative(uint64_t* total, uint64_t *results, uint64_t *powers_of_3, int *cache, uint64_t *B_table, uint64_t *C_table, uint64_t *S_table, int S_size, uint64_t n_start, uint64_t max_unfiltered) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= S_size) return;

    uint64_t mL = S_table[idx];
    __uint128_t n = (static_cast<__uint128_t>(n_start) << BASE_BITS) + mL;

    uint64_t n0 = static_cast<uint64_t>(n);
    // int delay = 0;
    uint64_t iteration_count = 0;



    while (n >= n0) {
        // if (n < CACHE_SIZE && cache[static_cast<uint64_t>(n)] != -1) {
        //     delay += cache[static_cast<uint64_t>(n)];
        //     break;
        // }

        // if (iteration_count >= MAX_ITERATIONS) {
        //     printf("Exceeded maximum iterations\n");
        //     break;
        // }

        n = method_A(n, powers_of_3);
        // n = method_B(n, B_table, C_table);
        // if (false) {
        //     if (n1 < n2) {
        //         n = n1;
        //     } else {
        //         n = n2;
        //     }
        // }

        // iteration_count++;
    }

    // results[idx] = iteration_count;
    // atomicAdd((unsigned long long *) total, iteration_count);

    // if (n0 < CACHE_SIZE) {
    //     cache[n0] = delay;
    // }
    // results[idx] = delay;
}

void initialize_powers_of_3(uint64_t *powers_of_3_host, int max_power) {
    powers_of_3_host[0] = 1;
    for (int i = 1; i <= max_power; ++i) {
        powers_of_3_host[i] = powers_of_3_host[i - 1] * 3;
    }
}

int main() {
    const int base_bits = BASE_BITS;

    // Generate S table using GPU in chunks
    uint64_t BC_size = 1ULL << BASE_TABLE_BITS;
    uint64_t* B_table = new uint64_t[BC_size];
    uint64_t* C_table = new uint64_t[BC_size];
    auto start_S = std::chrono::high_resolution_clock::now();
    auto S = generate_S_on_gpu(base_bits);
    auto end_S = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_S = end_S - start_S;

    std::cout << "Generated S table with size: " << S.size() << " in "
              << elapsed_S.count() << " seconds." << std::endl;

    auto start_BC = std::chrono::high_resolution_clock::now();
    fill_table(B_table, C_table, BASE_TABLE_BITS);
    auto end_BC = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_BC = end_BC - start_BC;

    std::cout << "Generated BC table with size: " << BC_size << " in "
              << elapsed_BC.count() << " seconds." << std::endl;

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
    uint64_t *B_device;
    uint64_t *C_device;
    uint64_t* total_device;
    uint64_t* total_host = new uint64_t[1];
    total_host[0] = 0;

    cudaMalloc(&total_device, sizeof(uint64_t));
    cudaMalloc(&results_device, S.size() * sizeof(uint64_t));
    cudaMalloc(&powers_of_3_device, 65 * sizeof(uint64_t));
    cudaMalloc(&cache_device, CACHE_SIZE * sizeof(int));
    cudaMalloc(&S_device, S.size() * sizeof(uint64_t));
    cudaMalloc(&B_device, BC_size * sizeof(uint64_t));
    cudaMalloc(&C_device, BC_size * sizeof(uint64_t));

    cudaMemcpy(total_device, total_host, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(powers_of_3_device, powers_of_3_host, 65 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cache_device, cache_host, CACHE_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(S_device, S.data(), S.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_table, BC_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(C_device, C_table, BC_size * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    const int threads_per_block = 256;
    const int num_blocks = (S.size() + threads_per_block - 1) / threads_per_block;

    convergence_test_iterative<<<num_blocks, threads_per_block>>>(total_device, results_device, powers_of_3_device, cache_device, B_device, C_device, S_device, S.size(), 1, max_unfiltered);
    cudaDeviceSynchronize();

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    cudaMemcpy(total_host, total_device, sizeof(uint64_t), cudaMemcpyDeviceToHost);
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
    std::cout << "Processed " << (double) max_unfiltered << " numbers (unfiltered) in "
              << elapsed.count() << " seconds." << std::endl;
    std::cout << "Processing rate: " << numbers_per_second << " numbers/second." << std::endl;
    std::cout << "Filtered S table size: " << S.size() << std::endl;
    std::cout << "Average iterations per number: " << (double) total_host[0] / (double) S.size() << std::endl;

    return 0;
}
