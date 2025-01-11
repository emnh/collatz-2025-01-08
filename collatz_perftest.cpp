#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <immintrin.h> // For __builtin_ctz
#include <cstdint> // For uint128_t

int ctz128(__uint128_t value) {
    if (value == 0) {
        return 128; // Handle special case: all bits are zero
    }

    // Extract lower and upper 64 bits
    uint64_t lower = static_cast<uint64_t>(value);
    uint64_t upper = static_cast<uint64_t>(value >> 64);

    if (lower != 0) {
        return __builtin_ctzll(lower); // Count trailing zeros in the lower 64 bits
    } else {
        return 64 + __builtin_ctzll(upper); // Count trailing zeros in the upper 64 bits
    }
}

// Function to initialize powers of 3 up to 3^64
std::vector<__uint128_t> initialize_powers_of_3() {
    std::vector<__uint128_t> powers(65, 1);
    for (size_t i = 1; i < 65; ++i) {
        powers[i] = powers[i - 1] * 3;
    }
    return powers;
}

// Function to truncate trailing zeroes and count them in binary
std::pair<__uint128_t, int> truncate_and_count_zeroes(__uint128_t n) {
    int count = __builtin_ctzll(static_cast<unsigned long long>(n)); // Count trailing zeros in binary
    n >>= count; // Remove trailing zeros
    return {n, count};
}

// Convergence test function
int convergence_test(__uint128_t n, const std::vector<__uint128_t>& powers_of_3) {
    int delay = 0;

    __uint128_t n0 = n;

    while (n > n0) {
        n = n + 1;
        auto [truncated_n, a] = truncate_and_count_zeroes(n);
        n = truncated_n;
        if (a < 65) {
            n = n * powers_of_3[a];
        }
        n = n - 1;
        auto [truncated_n2, b] = truncate_and_count_zeroes(n);
        n = truncated_n2;
        delay += a + b;
    }

    return delay;
}

int main() {
    unsigned int K;
    std::cout << "Enter the value of K: ";
    std::cin >> K;

    auto powers_of_3 = initialize_powers_of_3();
    auto start_total = std::chrono::high_resolution_clock::now();

    __uint128_t total_delay = 0;

    for (unsigned int i = 1; i <= K; ++i) {
        total_delay += convergence_test(i, powers_of_3);
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_total - start_total;
    double numbers_per_second = K / total_duration.count();

    std::cout << "Checksum (sum of delays): " << static_cast<unsigned long long>(total_delay) << "\n";
    std::cout << "Processed " << K << " numbers in " << total_duration.count() << " seconds.\n";
    std::cout << "Processing rate: " << numbers_per_second << " numbers/second.\n";

    return 0;
}
