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
#define ITER_BITS 1
#define BASE_TABLE_BITS 24
#define CHUNK_SIZE (1ULL << 30)  // 1 GiB of numbers per chunk
// #define MAX_ITERATIONS 1024

// Function to generate a filename based on a numeric variable
std::string generateFilename(const std::string prefix, const int variable) {
    return "/dev/shm/array_" + prefix + "_" + std::to_string(variable) + ".bin";
}

// Function to save an array to a file
void saveArrayToFile(const std::string& filename, const std::vector<uint64_t>& array) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    outFile.write(reinterpret_cast<const char*>(array.data()), array.size() * sizeof(uint64_t));
}

// Function to load an array from a file
std::vector<uint64_t> loadArrayFromFile(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary | std::ios::ate);
    if (!inFile) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return {};
    }
    std::streamsize size = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    std::vector<uint64_t> array(size / sizeof(uint64_t));
    inFile.read(reinterpret_cast<char*>(array.data()), size);
    return array;
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

__global__ void convergence_test_iterative(int max_tries, uint64_t* retry, uint64_t* total, uint64_t *results, uint64_t *powers_of_3, int *cache, uint64_t *B_table, uint64_t *C_table, uint64_t *S_table, int S_size, uint64_t n_start, uint64_t max_unfiltered) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= S_size) return;

    const uint64_t mL = S_table[idx];
    uint64_t iteration_count = 0;

    for (int n_mid = 0; n_mid < (1 << ITER_BITS); n_mid++) {
        __uint128_t n = (static_cast<__uint128_t>(n_mid) << BASE_BITS) + mL;

        uint64_t n0 = static_cast<uint64_t>(n);
        // int delay = 0;

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
            // break;
        }

        // results[idx] = iteration_count;

        // if (n0 < CACHE_SIZE) {
        //     cache[n0] = delay;
        // }
        // results[idx] = delay;
    }

    // atomicAdd((unsigned long long *) total, iteration_count);
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
    
    // Generate a filename based on the numeric variable
    std::string filename = generateFilename("S_table", base_bits);
    std::vector<uint64_t> S;
    // Check if the file exists
    if (std::filesystem::exists(filename)) {
        // Load the array from the file
        std::vector<uint64_t> loadedArray = loadArrayFromFile(filename);
        S = loadedArray;
        std::cout << "Loaded S table with size: " << S.size() << " from file." << std::endl;
    } else {
        S = generate_S_on_gpu(base_bits);
        // Save the array to a file
        saveArrayToFile(filename, S);
    }

    auto end_S = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_S = end_S - start_S;

    std::cout << "Generated S table with size: " << S.size() << " in "
              << elapsed_S.count() << " seconds." << std::endl;

    auto start_BC = std::chrono::high_resolution_clock::now();
    std::string B_filename = generateFilename("B_table", BASE_TABLE_BITS);
    std::string C_filename = generateFilename("C_table", BASE_TABLE_BITS);
    // Check if the file exists
    if (std::filesystem::exists(B_filename) && std::filesystem::exists(C_filename)) {
        // Load the arrays from the files
        std::vector<uint64_t> loadedBArray = loadArrayFromFile(B_filename);
        std::vector<uint64_t> loadedCArray = loadArrayFromFile(C_filename);
        std::copy(loadedBArray.begin(), loadedBArray.end(), B_table);
        std::copy(loadedCArray.begin(), loadedCArray.end(), C_table);
        std::cout << "Loaded BC tables with size: " << BC_size << " from files." << std::endl;
    } else {
        fill_table(B_table, C_table, BASE_TABLE_BITS);
        // Save the arrays to files
        saveArrayToFile(B_filename, std::vector<uint64_t>(B_table, B_table + BC_size));
        saveArrayToFile(C_filename, std::vector<uint64_t>(C_table, C_table + BC_size));
    }
    auto end_BC = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_BC = end_BC - start_BC;

    std::cout << "Generated BC table with size: " << BC_size << " in "
              << elapsed_BC.count() << " seconds." << std::endl;

    const uint64_t max_unfiltered = static_cast<uint64_t>(1) << (base_bits + ITER_BITS);

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
    uint64_t *retry_device;
    uint64_t *retry_count_device;
    total_host[0] = 0;

    auto startCopy = std::chrono::high_resolution_clock::now();

    cudaMalloc(&retry_device, S.size() * sizeof(uint64_t));
    cudaMalloc(&retry_count_device, sizeof(uint64_t));
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

    auto endCopy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedCopy = endCopy - startCopy;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    const int threads_per_block = 256;
    const int num_blocks = (S.size() + threads_per_block - 1) / threads_per_block;
    
    // uint64_t max_iterations = 1;
    const uint64_t benchmark_iterations = 1;
    for (uint64_t max_retries = 0; max_retries < benchmark_iterations; max_retries++) {
        convergence_test_iterative<<<num_blocks, threads_per_block>>>(max_retries, retry_device, total_device, results_device, powers_of_3_device, cache_device, B_device, C_device, S_device, S.size(), 1, max_unfiltered);   
    }
    
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
    cudaFree(retry_device);

    delete[] results_host;
    delete[] powers_of_3_host;
    delete[] cache_host;

    // Output benchmark results
    std::cout << "Copy time: " << elapsedCopy.count() << " seconds." << std::endl;
    double numbers_per_second = max_unfiltered / elapsed.count();
    std::cout << "Processed " << (double) max_unfiltered << " numbers (unfiltered) in "
              << elapsed.count() << " seconds." << std::endl;
    std::cout << "Processing rate: " << numbers_per_second << " numbers/second." << std::endl;
    std::cout << "Filtered S table size: " << S.size() << std::endl;
    std::cout << "Average iterations per number: " << (double) total_host[0] / (double) (S.size() * (1 << ITER_BITS)) << std::endl;

    return 0;
}
