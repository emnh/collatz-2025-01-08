#!/usr/bin/env python3

from itertools import product

def generate_binary_strings(length):
    """
    Generate all binary strings of a given length.
    :param length: Length of the binary strings to generate.
    :return: List of binary strings.
    """
    return [''.join(bits) for bits in product('01', repeat=length)]

def compute_linear_transform(n, binary_string):
    """
    Apply the sequence of operations defined by the binary string to the input number n.
    - '0' represents dividing by 2.
    - '1' represents the operation 3n + 1.

    :param n: The starting number.
    :param binary_string: The string of binary instructions.
    :return: The transformed number after applying all operations.
    """
    result = n
    accumulated_sum = n  # Initialize with the starting number
    for char in binary_string:
        if char == '0':
            result //= 2  # Integer division by 2
        elif char == '1':
            result = 3 * result + 1  # Apply 3n + 1
        accumulated_sum += result  # Accumulate the intermediate result
    return result, accumulated_sum

def find_linear_transform(n, binary_string):
    """
    Find the linear transform T(x) = ax + b resulting from applying the binary string operations.

    :param n: The starting number.
    :param binary_string: The string of binary instructions.
    :return: Coefficients a and b such that T(x) = ax + b.
    """
    initial_value = n
    final_value, accumulated_sum = compute_linear_transform(n, binary_string)

    # Calculate coefficients a and b
    length = len(binary_string)
    if length == 0:
        return 1, 0  # Identity transform if no operations

    a = (final_value - initial_value) / initial_value if initial_value != 0 else 0
    b = accumulated_sum - (a * initial_value)
    return a, b

# Example usage:
length = 3  # Example length of binary strings
initial_number = 10  # Example starting number

# Generate and apply operations to each string
binary_strings = generate_binary_strings(length)

print("Binary strings and their transformations:")
for binary_string in binary_strings:
    transformed_value, accumulated_sum = compute_linear_transform(initial_number, binary_string)
    a, b = find_linear_transform(initial_number, binary_string)
    print(f"{binary_string}: Transformed value = {transformed_value}, Accumulated sum = {accumulated_sum}, T(x) = {a:.2f}x + {b:.2f}")
