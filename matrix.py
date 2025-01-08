import numpy as np
import pandas as pd

# Helper Functions
def to_digit_vector(number, base):
    """Convert a number to its digit vector in the given base."""
    digits = []
    while number > 0:
        digits.append(number % base)
        number //= base
    return np.array(digits[::-1])  # Reverse to maintain correct positional order

def from_digit_vector(digits, base):
    """Convert a digit vector back to a number in the given base."""
    number = 0
    for digit in digits:
        number = number * base + digit
    return number

def create_transformation_matrix(base_from, base_to, max_power):
    """Create a transformation matrix for converting between bases."""
    T = np.zeros((max_power, max_power), dtype=int)
    for i in range(max_power):
        for j in range(i, max_power):
            T[i, j] = (base_from ** (j - i)) // (base_to ** (j - i))
    return T

def base_conversion_matrix(number, base_from, base_to, max_power):
    """Convert a number from one base to another using matrix multiplication."""
    # Convert the number to its digit vector in base_from
    digit_vector = to_digit_vector(number, base_from)
    length = len(digit_vector)

    # Create the transformation matrix
    T = create_transformation_matrix(base_from, base_to, max_power)

    # Pad the digit vector to match the transformation matrix size
    padded_digit_vector = np.pad(digit_vector, (max_power - length, 0), constant_values=0)

    # Perform matrix multiplication to convert to the new base
    result_vector = np.dot(padded_digit_vector, T)

    # Convert result vector back to the target base
    result_number = from_digit_vector(result_vector, base_from)
    return to_digit_vector(result_number, base_to)

def simple_base_conversion(number, base_to):
    """Simplest method for converting a number to another base."""
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

# Testing Collatz Operations with Both Methods
collatz_results = []
for num in range(1, 11):  # Test numbers from 1 to 10
    base_from = 10
    base_to = 3
    max_power = 20  # Maximum power to handle matrix size

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
        converted = base_conversion_matrix(current_matrix, base_from, base_to, max_power)
        sequence_matrix.append(list(converted))  # Ensure consistent representation
        current_matrix = collatz_next(current_matrix)
    sequence_matrix.append(list(base_conversion_matrix(current_matrix, base_from, base_to, max_power)))  # Add 1

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
results_df.to_csv("collatz_verification_results_fixed.csv", index=False)

# Display Results
print(results_df)
