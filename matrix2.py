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
    #return np.array(digits)
    return np.array(digits[::-1])  # Reverse to maintain correct positional order

def from_digit_vector(digits, base):
    """Convert a digit vector back to a number in the given base."""
    number = 0
    for digit in list(reversed(digits)):
        number = number * base + digit
    return number

def propagate_carries(digits, base):
    """Propagate carries in a digit vector."""
    outdigits = []
    carry = 0
    #print("digits", digits)
    for digit in reversed(digits):
        #print("digit", digit, "carry", carry)
        current = digit + carry
        outdigits += [current % base]
        carry = current // base
    if carry > 0:
        outdigits += [carry]
    return outdigits

def propagate_carries_div_2(digits, base):
    """Propagate carries in a digit vector."""
    outdigits = []
    carry = 0
    #print("digits", digits)
    for digit in digits: #reversed(digits):
        #print("digit", digit, "carry", carry)
        current = digit + carry
        while current > 0 and current % 2 == 0:
            current //= 2
        outdigits += [current % base]
        carry = current // base
    if carry > 0:
        outdigits += [carry]
    return outdigits


def matrix(N):
    ternary = to_digit_vector(N, 3)
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

def collatz_matrix(N):
    ternary = to_digit_vector(N, 3)
    #print(ternary)
    sequence = []
    print("prewhile", ternary, from_digit_vector(ternary, 3))
    trail = {}
    while from_digit_vector(ternary, 3) > 1:
        #print("While", from_digit_vector(ternary[::-1], 3))
        ternary_str = ''.join(str(bit) for bit in ternary).lstrip("0")
        if ternary_str in trail:
            print("Cycle", from_digit_vector(ternary, 3), "trail", trail)
            break
        trail[ternary_str] = True
        p = len(ternary)
        binary_matrix = generate_binary_matrix(p)
        # Display the matrix
        #for row in binary_matrix:
        #    print(row)
        #print(ternary)
        binary_matrix = np.transpose(binary_matrix)
        binary = np.dot(binary_matrix, ternary)
        print("binary_1", binary)
        binary = propagate_carries_div_2(binary, 2)
        while binary[-1] % 2 == 0:
            print("binary_A", binary)
            binary = binary[:-2]
        #print("binary_2", binary) 
        #binstring = bin(from_digit_vector(binary, 2))[2:]
        #binstring = "".join(list(reversed(binstring))).strip("0")
        #print("binstring", binstring, binary)
        #binary = to_digit_vector(int(binstring, 2), 2)
        #while binary[-1] % 2 == 0:
        #    binary = binary[:-2]
        #is_still_even = sum(binary) % 2 == 0
        N2 = from_digit_vector(binary, 2)
        ternary = to_digit_vector(N2, 3)
        #if is_still_even:
        #    pass
        #    #||binary = [bit // 2 for bit in binary]
        #else:
        #print("T1", ternary)
        ternary = np.append(ternary, 1) 
        #ternary = np.array(list(ternary).append(1))
        #print("T2", ternary)
        #print("Ternary", ternary)
        sbin = ''.join(str(bit) for bit in binary).lstrip("0")
        should_be = bin(N2)[2:].lstrip("0")
        match = sbin == should_be
        sequence.append((N2, match))
        if not should_be:
            print("ternary", ternary, "N", N, "N2", N2, "match", match, "bin", sbin, "should be", should_be)
            print("")
    return sequence

def classical_collatz_cycle_length(n, cache={}):
    """Compute the Collatz cycle length for n using the classical method."""
    cycle_length = 0
    newsteps = []
    yield n
    while n > 1:
        if n in cache:
            cycle_length += cache[n]
            break
        else:
            newsteps.append((n, cycle_length))
        if n % 2 == 0:
            n //= 2
            yield n
        else:
            n = 3 * n + 1
            yield n
        cycle_length += 1
    #for step in newsteps:
    #    cache[step[0]] = cycle_length - step[1]
    #return cycle_length

def main():
    for N in range(27, 28):
        print("N", N)
        sequence_mat = [x[0] for x in collatz_matrix(N)]
        sequence_cls = list(classical_collatz_cycle_length(N))
        print("Sequence", sequence_mat == sequence_cls, sequence_mat, sequence_cls)

if __name__ == "__main__":
    main()