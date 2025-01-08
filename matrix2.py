#!/usr/bin/env python3

import math
import numpy as np

def generate_binary_matrix(p):
    """
    Generates a matrix where each row corresponds to powers of 3 (3, 3^2, ..., 3^p)
    represented as binary vectors.

    Parameters:
        p (int): The maximum power of 3.

    Returns:
        numpy.ndarray: A matrix where each row is the binary representation of 3^k.
    """
    # Compute the powers of 3
    powers_of_3 = [3 ** i for i in range(0, p)]

    # Find the maximum binary length
    max_bin_length = len(bin(powers_of_3[-1])) - 2  # Exclude '0b' prefix

    # Convert each number to binary and pad to the same length
    binary_matrix = np.array([
        [int(bit) for bit in bin(num)[2:].zfill(max_bin_length)]
        for num in powers_of_3
    ])

    return binary_matrix

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

def propagate_carries(digits, base):
    """Propagate carries in a digit vector."""
    outdigits = []
    carry = 0
    print("digits", digits)
    for digit in reversed(digits):
        print("digit", digit, "carry", carry)
        current = digit + carry
        outdigits += [current % base]
        carry = current // base
    return outdigits

def matrix(N):
    ternary = to_digit_vector(N, 3)[::-1]
    p = len(ternary)
    #ternary = np.pad(ternary, p - len(ternary), constant_values=0)
    binary_matrix = generate_binary_matrix(p)
    # Display the matrix
    for row in binary_matrix:
        print(row)
    #print(ternary)
    binary_matrix = np.transpose(binary_matrix)
    binary = np.dot(binary_matrix, ternary)

    #N2 = from_digit_vector(binary, 2)
    binary = propagate_carries(binary, 2)
    N2 = from_digit_vector(binary, 2)
    sbin = ''.join(str(bit) for bit in binary).lstrip("0")
    should_be = bin(N2)[2:].lstrip("0")
    match = sbin == should_be
    print("ternary", ternary, "N", N, "N2", N2, "match", match, "bin", sbin, "should be", should_be)
    print("")

for i in range(1, 128+1):
    matrix(i)