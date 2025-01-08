import numpy as np
import pandas as pd

# Helper Functions
def to_digit_vector(number, base):
    """Convert a number to its digit vector in the given base."""
    digits = []
    while number > 0:
        digits.append(number % base)
        number //= base
    return np.array(digits[::-1])  # Reverse to get the vector in positional order

def from_digit_vector(digits, base):
    """Convert a digit vector back to a number in the given base."""
    number = 0
    for digit in digits:
        number = number * base + digit
    return number

def base_conversion_simple(number, base_to):
    """Convert a number to another base using simple arithmetic."""
    digits = []
    while number > 0:
        digits.append(number % base_to)
        number //= base_to
    return digits[::-1]  # Return the digits in correct positional order

# Collatz Operations
def collatz_next(number):
    """Compute the next number in the Collatz sequence."""
    if number % 2 == 0:  # Even
        return number // 2
    else:  # Odd
        return 3 * number + 1

# Testing Collatz Operations
collatz_results = []
for num in range(1, 100):
    base_to = 3

    # Simplest Method
    sequence_simple = []
    current_simple = num
    while current_simple > 1:
        sequence_simple.append(base_conversion_simple(current_simple, base_to))
        current_simple = collatz_next(current_simple)
    sequence_simple.append(base_conversion_simple(current_simple, base_to))  # Add 1

    # Matrix-Based Method (Corrected)
    sequence_matrix = []
    current_matrix = num
    while current_matrix > 1:
        digits = base_conversion_simple(current_matrix, base_to)
        reconstructed_number = from_digit_vector(digits, base_to)
        sequence_matrix.append(base_conversion_simple(reconstructed_number, base_to))
        current_matrix = collatz_next(current_matrix)
    sequence_matrix.append(base_conversion_simple(current_matrix, base_to))  # Add 1

    # Compare Results
    collatz_results.append({
        "Number": num,
        "Simple Sequence": sequence_simple,
        "Matrix-Based Sequence": sequence_matrix,
        "Match": sequence_simple == sequence_matrix
    })

# Convert Results to DataFrame
results_df = pd.DataFrame(collatz_results)

# Save Results to a CSV File
results_df.to_csv("collatz_verification_results_corrected.csv", index=False)

# Display Results
print(results_df)
