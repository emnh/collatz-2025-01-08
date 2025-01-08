from collatz import classical_collatz_cycle_length, truncate_and_count_zeroes, truncate_and_count_ones, shortcut_collatz_cycle_length_gen

def powers_of_3_to_binary(max_power):
    """Precompute powers of 3 mapped to binary."""
    table = {}
    for k in range(max_power + 1):
        table[k] = bin(3**k)[2:]
    return table

def powers_of_2_to_ternary(max_power):
    """Precompute powers of 2 mapped to ternary."""
    table = {}
    for m in range(max_power + 1):
        # Convert 2^m to ternary
        table[m] = to_ternary(2**m)
    return table

def to_ternary(n):
    """Convert a decimal number to ternary."""
    if n == 0:
        return "0"
    digits = []
    while n > 0:
        digits.append(str(n % 3))
        n //= 3
    return ''.join(reversed(digits))

def shift_and_reconstruct(powers_3_to_bin, powers_2_to_tern, ternary_num):
    """
    Shift powers of 3 to the right, sum in binary, then map back to ternary.

    Args:
        powers_3_to_bin (dict): Precomputed powers of 3 to binary mapping.
        powers_2_to_tern (dict): Precomputed powers of 2 to ternary mapping.
        ternary_num (str): Ternary number as input.

    Returns:
        str: Resulting ternary number after operations.
    """
    # Convert ternary input to powers of 3
    powers = [int(digit) for digit in ternary_num[::-1]]  # Reverse for positional value
    max_power = len(powers) - 1

    # Sum shifted binary values
    binary_sum = 0
    for k, digit in enumerate(powers):
        if digit > 0:
            binary_value = int(powers_3_to_bin[k], 2)
            binary_shifted = binary_value >> 1  # Divide by 2 in binary
            binary_sum += digit * binary_shifted

    # Convert binary sum to ternary using the powers of 2 table
    ternary_result = ""
    while binary_sum > 0:
        lowest_power = binary_sum & -binary_sum  # Isolate the lowest power of 2
        power_index = (lowest_power).bit_length() - 1
        ternary_result = powers_2_to_tern.get(power_index, "0") + ternary_result
        binary_sum -= lowest_power

    return ternary_result or "0"

def isEven(digits_ternary):
    sd = sum(int(x) for x in digits_ternary)
    return sd % 2 == 0

# Precompute tables
powers_3_to_bin = powers_of_3_to_binary(1000)
powers_2_to_tern = powers_of_2_to_ternary(1000)

# Example input: ternary number
for i in range(1, 100):
    #ternary_num = "2101"  # Base-3 representation
    result = to_ternary(i)
    while int(result, 3) > 1:
        print("N3: ", result)
        while not isEven(result):
            result = result + "1"
            print("N3 mul: ", result)
        result = shift_and_reconstruct(powers_3_to_bin, powers_2_to_tern, result)
        print(f"N3 div: {result}")
    print("")