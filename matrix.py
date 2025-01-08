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

def create_transformation_matrix(base_from, base_to, length):
    """Create a transformation matrix for converting between bases."""
    T = np.zeros((length, length), dtype=int)
    for i in range(length):
        for j in range(i, length):
            T[i, j] = base_from ** (j - i) // base_to ** (j - i)
    return T

def base_conversion_matrix_corrected(number, base_from, base_to):
    """Convert a number from one base to another using matrix multiplication."""
    # Convert number to digit vector in base_from
    digit_vector = to_digit_vector(number, base_from)
    length = len(digit_vector)

    # Compute the decimal equivalent of the base_from number
    decimal_number = sum(d * (base_from ** i) for i, d in enumerate(reversed(digit_vector)))

    # Convert the decimal number to the target base (base_to)
    return simple_base_conversion(decimal_number, base_to)

def simple_base_conversion(number, base_to):
    """Simplest method for converting a number to another base."""
    digits = []
    while number > 0:
        digits.append(number % base_to)
        number //= base_to
    return int("".join(map(str, digits[::-1])))

# Collatz Operations
def collatz_next(number):
    """Compute the next number in the Collatz sequence."""
    if number % 2 == 0:  # Even
        return number // 2
    else:  # Odd
        return 3 * number + 1

# Testing Collatz Operations with Both Methods
collatz_results = []
for num in range(1, 11):  # Test numbers from 1 to 10
    base_from = 10
    base_to = 3

    # Simplest Method
    sequence_simple = []
    current_simple = num
    while current_simple > 1:
        sequence_simple.append(simple_base_conversion(current_simple, base_to))
        current_simple = collatz_next(current_simple)
    sequence_simple.append(simple_base_conversion(current_simple, base_to))  # Add 1

    # Matrix-Based Method
    sequence_matrix = []
    current_matrix = num
    while current_matrix > 1:
        sequence_matrix.append(base_conversion_matrix_corrected(current_matrix, base_from, base_to))
        current_matrix = collatz_next(current_matrix)
    sequence_matrix.append(base_conversion_matrix_corrected(current_matrix, base_from, base_to))  # Add 1

    # Compare Results
    collatz_results.append({
        "Number": num,
        "Simple Sequence": sequence_simple,
        "Matrix-Based Sequence": sequence_matrix,
        "Match": sequence_simple == sequence_matrix
    })

# Display Results
results_df = pd.DataFrame(collatz_results)
#import ace_tools as tools; tools.display_dataframe_to_user(name="Collatz Sequence Verification Results", dataframe=results_df)

# Create and display the DataFrame
#results_df = pd.DataFrame(results)
print(results_df)

# Optionally save it to a CSV file
#results_df.to_csv("base_conversion_comparison.csv", index=False)