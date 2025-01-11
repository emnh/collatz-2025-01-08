#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <immintrin.h> // For __builtin_ctz
#include <cstdint> // For uint128_t

// Precomputed powers of 3 up to 3^64
const __uint128_t POWERS_OF_3[] = {
    1ULL, 3ULL, 9ULL, 27ULL, 81ULL, 243ULL, 729ULL, 2187ULL,
    6561ULL, 19683ULL, 59049ULL, 177147ULL, 531441ULL, 1594323ULL, 4782969ULL,
    14348907ULL, 43046721ULL, 129140163ULL, 387420489ULL, 1162261467ULL,
    3486784401ULL, 10460353203ULL, 31381059609ULL, 94143178827ULL,
    282429536481ULL, 847288609443ULL, 2541865828329ULL, 7625597484987ULL,
    22876792454961ULL, 68630377364883ULL, 205891132094649ULL,
    617673396283947ULL, 1853020188851841ULL, 5559060566555523ULL,
    16677181699666569ULL, 50031545098999707ULL, 150094635296999121ULL,
    450283905890997363ULL, 1350851717672992089ULL, 4052555153018976267ULL,
    12157665459056928801ULL, 36472996377170786403ULL, 109418989131512359209ULL,
    328256967394537077627ULL, 984770902183611232881ULL, 2954312706550833698643ULL,
    8862938119652501095929ULL, 26588814358957503287787ULL,
    79766443076872509863361ULL, 239299329230617529590083ULL,
    717897987691852588770249ULL, 2153693963075557766310747ULL,
    6461081889226673298932241ULL, 19383245667680019896796723ULL,
    58149737003040059690390169ULL, 174449211009120179071170507ULL,
    523347633027360537213511521ULL, 1570042899082081611640534563ULL,
    4710128697246244834921603689ULL, 14130386091738734504764811067ULL,
    42391158275216203514294433201ULL, 127173474825648610542883299603ULL
};

// Function to truncate trailing zeroes and count them in binary
std::pair<__uint128_t, int> truncate_and_count_zeroes(__uint128_t n) {
    int count = __builtin_ctzll(static_cast<unsigned long long>(n)); // Count trailing zeros in binary
    n >>= count; // Remove trailing zeros
    return {n, count};
}

// Convergence test function
std::vector<std::string> convergence_test(__uint128_t n) {
    std::vector<std::string> sequence;
    sequence.push_back(std::to_string(static_cast<unsigned long long>(n)));
    int delay = 0;

    while (n > 1) {
        n = n + 1;
        auto [truncated_n, a] = truncate_and_count_zeroes(n);
        n = truncated_n;
        if (a < 64) {
            n = n * POWERS_OF_3[a];
        }
        n = n - 1;
        auto [truncated_n2, b] = truncate_and_count_zeroes(n);
        n = truncated_n2;
        delay += a + b;
        sequence.push_back(std::to_string(static_cast<unsigned long long>(n)));

        if (n < static_cast<unsigned long long>(sequence.front()[0] - '0')) {
            break;
        }
    }

    sequence.push_back("delay: " + std::to_string(delay));
    return sequence;
}

int main() {
    unsigned int K;
    std::cout << "Enter the value of K: ";
    std::cin >> K;

    auto start_total = std::chrono::high_resolution_clock::now();

    for (unsigned int i = 1; i <= K; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::string> result = convergence_test(i);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;

        std::cout << "Input: " << i << ", Time: " << duration.count() << " ms\n";
        for (const auto& item : result) {
            std::cout << item << " ";
        }
        std::cout << "\n\n";
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_total - start_total;
    double numbers_per_second = K / total_duration.count();

    std::cout << "Processed " << K << " numbers in " << total_duration.count() << " seconds.\n";
    std::cout << "Processing rate: " << numbers_per_second << " numbers/second.\n";

    return 0;
}
