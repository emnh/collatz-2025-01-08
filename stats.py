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
            n = 3 * n + 1
        sequence.append(n)
    return sequence

# Function to analyze the binary characteristics
def analyze_binary(number):
    binary = bin(number)[2:]  # Get binary representation without the "0b"
    count_0 = binary.count('0')
    count_1 = binary.count('1')
    length = len(binary)
    transitions = sum(1 for i in range(len(binary) - 1) if binary[i] == '1' and binary[i + 1] == '0')
    return {
        "0_to_1_ratio": count_0 / count_1 if count_1 > 0 else 0,
        "binary_length": length,
        "transitions": transitions
    }

# Main analysis function
def analyze_collatz(start, end):
    data = []

    for num in range(start, end + 1):
        trajectory = collatz_sequence(num)
        for number in trajectory:
            binary_analysis = analyze_binary(number)
            data.append(binary_analysis)

    return data

# Aggregate statistics
def aggregate_statistics(data):
    ratios = [entry["0_to_1_ratio"] for entry in data]
    lengths = [entry["binary_length"] for entry in data]
    transitions = [entry["transitions"] for entry in data]

    return {
        "average_ratio": np.mean(ratios),
        "average_length": np.mean(lengths),
        "average_transitions": np.mean(transitions),
        "ratio_histogram": np.histogram(ratios, bins=10),
        "length_histogram": np.histogram(lengths, bins=10),
        "transitions_histogram": np.histogram(transitions, bins=10),
    }

# Visualization
def plot_histograms(stats):
    plt.figure(figsize=(12, 8))

    # Plot histogram for ratios
    plt.subplot(3, 1, 1)
    plt.hist(stats["ratio_histogram"][1][:-1], bins=10, weights=stats["ratio_histogram"][0])
    plt.title("Histogram of 0-to-1 Ratios")
    plt.xlabel("0-to-1 Ratio")
    plt.ylabel("Frequency")

    # Plot histogram for binary lengths
    plt.subplot(3, 1, 2)
    plt.hist(stats["length_histogram"][1][:-1], bins=10, weights=stats["length_histogram"][0])
    plt.title("Histogram of Binary Lengths")
    plt.xlabel("Binary Length")
    plt.ylabel("Frequency")

    # Plot histogram for transitions
    plt.subplot(3, 1, 3)
    plt.hist(stats["transitions_histogram"][1][:-1], bins=10, weights=stats["transitions_histogram"][0])
    plt.title("Histogram of 1-to-0 Transitions")
    plt.xlabel("1-to-0 Transitions")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# Driver code
if __name__ == "__main__":
    # Define range of numbers to analyze
    start, end = 1, 100

    # Analyze the Collatz trajectories
    data = analyze_collatz(start, end)

    # Aggregate statistics
    stats = aggregate_statistics(data)

    # Print aggregated statistics
    print("Average 0-to-1 Ratio:", stats["average_ratio"])
    print("Average Binary Length:", stats["average_length"])
    print("Average 1-to-0 Transitions:", stats["average_transitions"])

    # Plot histograms
    plot_histograms(stats)
