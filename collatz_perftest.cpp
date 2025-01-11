#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdint> // For uint128_t

constexpr unsigned int MAX_ITERATIONS = 1024;
constexpr unsigned int K = 1000000;
constexpr unsigned int CACHE_SIZE = K;
constexpr int UNINITIALIZED = -1;

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

    while (n > 1) {
        if (n < CACHE_SIZE && cache[n] != UNINITIALIZED) {
            delay += cache[n];
            break;
        }

        if (iteration_count >= MAX_ITERATIONS) {
            std::cerr << "Error: Exceeded maximum allowed iterations (" << MAX_ITERATIONS << ")\n";
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

        intermediate[iteration_count++] = {n, a + b};
    }

    for (int i = iteration_count - 1; i >= 0; --i) {
        if (intermediate[i].first < CACHE_SIZE) {
            cache[intermediate[i].first] = delay;
        }
        delay -= intermediate[i].second;
    }

    return delay;
}

// Recursive convergence test function
int convergence_test_recursive(const __uint128_t n_input, const std::vector<__uint128_t>& powers_of_3, int* cache, int depth = 0) {
    if (n_input < CACHE_SIZE && cache[n_input] != UNINITIALIZED) {
        return cache[n_input];
    }

    if (depth >= MAX_ITERATIONS) {
        std::cerr << "Error: Exceeded maximum allowed recursion depth (" << MAX_ITERATIONS << ")\n";
        std::exit(EXIT_FAILURE);
    }

    if (n_input == 1) {
        return 0;
    }

    const __uint128_t n_incremented = n_input + 1;
    const auto [truncated_n, a] = truncate_and_count_zeroes(n_incremented);
    const __uint128_t n_multiplied = truncated_n * powers_of_3[a] - 1;
    const auto [truncated_n2, b] = truncate_and_count_zeroes(n_multiplied);

    const int delay = a + b + convergence_test_recursive(truncated_n2, powers_of_3, cache, depth + 1);
    if (truncated_n2 < CACHE_SIZE) {
        cache[truncated_n2] = delay;
    }

    return delay;
}

int main() {
    const auto powers_of_3 = initialize_powers_of_3();
    const auto start_total = std::chrono::high_resolution_clock::now();

    __uint128_t total_delay = 0;
    int* cache = new int[CACHE_SIZE];
    std::fill(cache, cache + CACHE_SIZE, UNINITIALIZED);
    std::pair<__uint128_t, int> intermediate[MAX_ITERATIONS];

    for (unsigned int i = 1; i <= K; ++i) {
        total_delay += convergence_test_iterative(i, powers_of_3, cache, intermediate);
    }

    delete[] cache;

    const auto end_total = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> total_duration = end_total - start_total;
    const double numbers_per_second = K / total_duration.count();

    std::cout << "Checksum (sum of delays): " << static_cast<unsigned long long>(total_delay) << "\n";
    std::cout << "Processed " << K << " numbers in " << total_duration.count() << " seconds.\n";
    std::cout << "Processing rate: " << numbers_per_second << " numbers/second.\n";
    std::cout << "Processing rate (log2): " << std::log2(numbers_per_second) << "\n";

    return 0;
}
