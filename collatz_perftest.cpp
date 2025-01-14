#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdint> // For uint128_t
#include <iomanip> // For std::fixed and std::setprecision
#include <bitset>

constexpr __uint128_t MAX_ITERATIONS = 1024;
constexpr __uint128_t K = 1 << 30;
constexpr __uint128_t CACHE_SIZE = K;
constexpr int UNINITIALIZED = -1;
//constexpr __uint128_t BASE_BITS = 20;
// constexpr unsigned int S_BITS = 30;

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

// Generate the S table based on the paper's logic for mandatory least significant bits
std::vector<__uint128_t> generate_S(const __uint128_t BASE_BITS) {
    std::vector<__uint128_t> S;
    const __uint128_t max_nL = (__uint128_t) 1 << BASE_BITS;

    for (__uint128_t nL = 0; nL < max_nL; ++nL) {
        __uint128_t b = (__uint128_t) 1 << BASE_BITS; // Start with b = 2^d
        __uint128_t c = nL;

        bool mandatory = true;
        while (b % 2 == 0) {
            if (b % 2 == 0 and c % 2 == 0) {
                b /= 2;
                c /= 2;
            } else if (c % 2 == 1) {
                b *= 3;
                c = 3 * c + 1;
            }
            
            if (b <= ((__uint128_t) 1 << BASE_BITS) - 1) {
                mandatory = false;
                break;
            }
        }

        if (mandatory) {
            // std::cout << "S: " << std::bitset<32>(nL) << ": " << b << " " << c << std::endl;
            S.push_back(nL);
        }
    }
    std::cout << "S size: " << S.size() << std::endl;

    return S;
}

// Function to initialize powers of 3 up to 3^128
std::vector<__uint128_t> initialize_powers_of_3() {
    std::vector<__uint128_t> powers(129, 1);
    for (size_t i = 1; i < 129; ++i) {
        powers[i] = powers[i - 1] * 3;
    }
    return powers;
}

// Function to count trailing zeros for 128-bit numbers using built-in ctz for each 64-bit half
inline int count_trailing_zeros_128(const __uint128_t n) {
    if (n == 0) return 128;
    const uint64_t lower = static_cast<uint64_t>(n);
    const uint64_t upper = static_cast<uint64_t>(n >> 64);

    return (lower != 0) ? __builtin_ctzll(lower) : 64 + __builtin_ctzll(upper);
}

// Function to truncate trailing zeroes and count them in binary
inline std::pair<__uint128_t, int> truncate_and_count_zeroes(__uint128_t n) {
    const int count = count_trailing_zeros_128(n); // Count trailing zeros in binary
    n >>= count; // Remove trailing zeros
    return {n, count};
}

// Iterative convergence test function with caching
int convergence_test_iterative(__uint128_t n, const std::vector<__uint128_t>& powers_of_3, int* cache, std::pair<__uint128_t, int>* intermediate) {
    int delay = 0;
    unsigned int iteration_count = 0;
    __uint128_t n0 = n;

    while (n > 1) {
        if (n < CACHE_SIZE && cache[n] != UNINITIALIZED) {
            delay += cache[n];
            break;
        }

        if (iteration_count >= MAX_ITERATIONS) {
            std::cerr << "Error: Exceeded maximum allowed iterations (" << uint128_to_string(MAX_ITERATIONS) << ")\n";
            std::exit(EXIT_FAILURE);
        }

        n = n + 1;
        const auto [truncated_n, a] = truncate_and_count_zeroes(n);
        n = truncated_n;
        n *= powers_of_3[a];
        n = n - 1;
        const auto [truncated_n2, b] = truncate_and_count_zeroes(n);
        n = truncated_n2;
        delay += a + b;

        // intermediate[iteration_count++] = {n, a + b};
    }

    // for (int i = iteration_count - 1; i >= 0; --i) {
    //     if (intermediate[i].first < CACHE_SIZE) {
    //         cache[intermediate[i].first] = delay;
    //     }
    //     delay -= intermediate[i].second;
    // }
    cache[n0] = delay;

    return delay;
}

// Recursive convergence test function
int convergence_test_recursive(const __uint128_t n0_input, const __uint128_t n_input, const std::vector<__uint128_t>& powers_of_3, int* cache, int depth = 0) {
    if (n_input < n0_input) {
        return 0;
    }
    
    if (n_input < CACHE_SIZE && cache[n_input] != UNINITIALIZED) {
        return cache[n_input];
    }

    if (depth >= MAX_ITERATIONS) {
        std::cerr << "Error: Exceeded maximum allowed recursion depth (" << uint128_to_string(MAX_ITERATIONS) << ")\n";
        std::exit(EXIT_FAILURE);
    }

    if (n_input == 1) {
        return 0;
    }

    const __uint128_t n_incremented = n_input + 1;
    const auto [truncated_n, a] = truncate_and_count_zeroes(n_incremented);
    const __uint128_t n_multiplied = truncated_n * powers_of_3[a] - 1;
    const auto [truncated_n2, b] = truncate_and_count_zeroes(n_multiplied);

    const int delay = a + b + convergence_test_recursive(n0_input, truncated_n2, powers_of_3, cache, depth + 1);
    // if (truncated_n2 < CACHE_SIZE) {
    //     cache[truncated_n2] = delay;
    // }
    if (n_input < CACHE_SIZE) {
        cache[n_input] = delay;
    }

    return delay;
}

int main2(__uint128_t BASE_BITS = 20) {
    // constexpr unsigned int K = 1 << BASE_BITS;
    const auto powers_of_3 = initialize_powers_of_3();
    const auto S = generate_S(BASE_BITS);
    const auto start_total = std::chrono::high_resolution_clock::now();

    __uint128_t total_delay = 0;
    int* cache = new int[CACHE_SIZE];
    std::fill(cache, cache + CACHE_SIZE, UNINITIALIZED);
    std::pair<__uint128_t, int> intermediate[MAX_ITERATIONS];

    // for (unsigned int i = 1; i < K; ++i) {
    //     // total_delay += convergence_test_iterative(i, powers_of_3, cache, intermediate);
    //     total_delay += convergence_test_recursive(i, powers_of_3, cache);
    // }

    // const __uint128_t unskipped_iterations = (K >> BASE_BITS) * (1 >> BASE_BITS);
    const __uint128_t iterations = (K >> BASE_BITS) * S.size();
    for (unsigned int mH = 1; mH <= K >> BASE_BITS; ++mH) {
        for (unsigned int i = 0; i < S.size(); ++i) {
            const __uint128_t mL = S[i];
            const __uint128_t n = (static_cast<__uint128_t>(mH) << BASE_BITS) + mL;
            total_delay += convergence_test_recursive(n, n, powers_of_3, cache);
        }
    }

    delete[] cache;

    const auto end_total = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> total_duration = end_total - start_total;
    const double numbers_per_second = K / total_duration.count();

    // std::cout << "                      K = " << K << std::endl;
    std::cout << "Expected checksum for        K = 1000000: " << 87826377 << std::endl;
    std::cout << "Expected checksum for        K = 10000000: " << 1037278417 << std::endl;
    std::cout << "Checksum (sum of delays) for K = " << uint128_to_string(K) << ": " << static_cast<unsigned long long>(total_delay) << "\n";
    std::cout << "Iterations and fraction: " << uint128_to_string(iterations) << " " << std::fixed << std::setprecision(4) << (double) iterations / (double) K << "%" << std::endl;
    std::cout << std::defaultfloat;
    std::cout << "Processed " << uint128_to_string(K) << " numbers in " << total_duration.count() << " seconds.\n";
    std::cout << "Processing rate: " << numbers_per_second << " numbers/second.\n";
    std::cout << "Processing rate (log2): " << std::log2(numbers_per_second) << "\n";

    return 0;
}

int main() {
    //for (int i = 4; i < 32; i++) {
    for (int i = 31; i < 32; i++) {
        std::cout << "Testing for " << i << " base bits" << std::endl;
        main2(i);
        std::cout << std::endl;
    }
}