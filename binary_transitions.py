import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Function to compute the Collatz trajectory
def collatz_sequence(n):
    sequence = [n]
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = (3 * n + 1) // 2
        sequence.append(n)
    return sequence

# Function to analyze the binary characteristics
def analyze_binary(number):
    binary = bin(number)[2:]  # Get binary representation without the "0b"
    count_0 = binary.count('0')
    count_1 = binary.count('1')
    length = len(binary)
    transitions = sum(1 for i in range(len(binary) - 1) if binary[i] != binary[i + 1])
    return {
        "binary": binary,
        "count_0": count_0,
        "count_1": count_1,
        "0_to_1_ratio": count_0 / count_1 if count_1 > 0 else 0,
        "transitions": transitions
    }

# Simplified analysis function
def simplified_analysis(start, end):
    for num in range(start, end + 1):
        trajectory = collatz_sequence(num)
        print(f"Starting Number: {num}")
        for number in trajectory:
            binary_analysis = analyze_binary(number)
            print(
                f"Number: {number}, Binary: {binary_analysis['binary']}, "
                f"0s: {binary_analysis['count_0']}, 1s: {binary_analysis['count_1']}, "
                f"0-to-1 Ratio: {binary_analysis['0_to_1_ratio']:.2f}, Transitions: {binary_analysis['transitions']}"
            )
        print()  # Blank line for separation

# Driver code
if __name__ == "__main__":
    # Define range of numbers to analyze
    start, end = 1, 100

    # Perform simplified analysis
    simplified_analysis(start, end)
