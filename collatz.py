#!/usr/bin/env python3

import graphviz
import math

def to_ternary(n):
    """Convert a number to its ternary representation."""
    if n == 0:
        return "0"
    
    digits = []
    is_negative = n < 0
    n = abs(n)

    while n > 0:
        digits.append(str(n % 3))
        n //= 3
    
    if is_negative:
        return "-" + "".join(reversed(digits))
    return "".join(reversed(digits))

def classical_collatz_cycle_length(n, cache={}):
    """Compute the Collatz cycle length for n using the classical method."""
    cycle_length = 0
    newsteps = []
    while n > 1:
        if n in cache:
            cycle_length += cache[n]
            break
        else:
            newsteps.append((n, cycle_length))
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        cycle_length += 1
    for step in newsteps:
        cache[step[0]] = cycle_length - step[1]
    return cycle_length

def classical_collatz_cycle_length_div2(n, cache={}):
    """Compute the Collatz cycle length for n using the classical method."""
    cycle_length = 0
    newsteps = []
    while n > 1:
        if n in cache:
            cycle_length += cache[n]
            break
        else:
            newsteps.append((n, cycle_length))
        if n % 2 == 0:
            n //= 2
        else:
            n = (3 * n + 1) // 2
        cycle_length += 1
    for step in newsteps:
        cache[step[0]] = cycle_length - step[1]
    return cycle_length

def truncate_and_count_zeroes(x):
    """Remove trailing zeroes and count them."""
    count = 0
    while x > 0 and x % 2 == 0:
        x //= 2
        count += 1
    return x, count

def truncate_and_count_ones(x):
    """Remove trailing ones and count them."""
    count = 0
    while x > 0 and x % 2 == 1:
        x = (x - 1) // 2
        count += 1
    return x, count

def shortcut_collatz_cycle_length_gen(n, cache={}):
    """Compute the Collatz cycle length for n using shortcuts."""
    cycle_length = 0

    loop_iterations = 0

    newsteps = []

    while n > 1:

        if n in cache:
            cycle_length += cache[n]
            yield n
            break
        else:
            yield n
            newsteps.append((n, cycle_length))

        loop_iterations += 1

        # Shortcut for trailing zeroes
        n, zero_count = truncate_and_count_zeroes(n)
        cycle_length += zero_count

        if n == 1:
            yield n
            break  # Early exit if n reduces to 1

        # Shortcut for trailing ones
        prefix, one_count = truncate_and_count_ones(n)

        if one_count > 1:
            # Shortcut calculation for ones
            n = 3 ** one_count * (prefix // 2) + (3 ** one_count - 1) // 2
            cycle_length += 2 * one_count + 1
        else:
            # Evaluate the zero count of the prefix
            if prefix == 0:  # Avoid infinite loop on zero prefix
                n = 1
                cycle_length += 1
                yield n
                break

            prefix, prefix_zero_count = truncate_and_count_zeroes(prefix)

            if prefix_zero_count > 1:
                half_zero_count = prefix_zero_count // 2
                n = (3 ** half_zero_count) * prefix
                if prefix_zero_count % 2 == 1:
                    n = n * 4 + 1
                else:
                    n = n * 2 + 1
                cycle_length += 3 * half_zero_count
            else:
                # Standard Collatz step
                if n % 2 == 1:
                    n = 3 * n + 1
                else:
                    n //= 2
                cycle_length += 1

    for step in newsteps:
        cache[step[0]] = cycle_length - step[1]

    yield n
    yield cycle_length, loop_iterations

def shortcut_collatz_cycle_length(n, cache={}):
    return list(shortcut_collatz_cycle_length_gen(n, cache))[-1]

def test_shortcut_vs_classical(max_n):
    """Compare the shortcut-based and classical methods for Collatz cycle length."""
    maxloop_classic, maxloop_shortcut = 0, 0
    classic_cache, shortcut_cache = {}, {}
    #joint_cache = {}
    records = []
    for n in range(1, max_n + 1):
        classical_length  = classical_collatz_cycle_length(n, classic_cache)
        shortcut_length, short_iters = shortcut_collatz_cycle_length(n, shortcut_cache)
        records = records + [(n, classical_length, short_iters)]
        #print(classic_cache, shortcut_cache)
        a, b = maxloop_classic, maxloop_shortcut
        maxloop_classic = max(maxloop_classic, classical_length)
        maxloop_shortcut = max(maxloop_shortcut, short_iters)
        n_bin = bin(n)[2:]
        if a != maxloop_classic:
            print(f"N: {n}, B: {n_bin}, Classic: {a} -> {maxloop_classic}")
        if b != maxloop_shortcut:
            print(f"N: {n}, B: {n_bin} Shortcut: {b} -> {maxloop_shortcut}")
        if shortcut_length != classical_length:
            print(f"Mismatch for n = {n}: Shortcut = {shortcut_length}, Classical = {classical_length}")
        else:
            diff = classical_length - short_iters
            #print(f"Match for n = {n}: Length = {shortcut_length}, Saved: {diff}, Loops {short_iters}")
        #print(f"Classic: {maxloop_classic} vs {maxloop_shortcut})

def max_collatz_by_length(max_bits):
    """Find numbers with the maximum Collatz cycle length for each binary length."""
    classic_cache = {}
    max_lengths_by_bits = {}
    numbers_by_max_length = {}

    for bit_length in range(1, max_bits + 1):
        start = 1 << (bit_length - 1)  # Smallest number with this bit length
        end = (1 << bit_length) - 1   # Largest number with this bit length

        max_length = 0
        max_numbers = []

        for n in range(start, end + 1):
            length = classical_collatz_cycle_length(n, classic_cache)
            if length > max_length:
                max_length = length
                max_numbers = [n]
            elif length == max_length:
                max_numbers.append(n)

        max_lengths_by_bits[bit_length] = max_length
        numbers_by_max_length[bit_length] = max_numbers

    for bit_length in sorted(numbers_by_max_length.keys()):
        print(f"Bit length {bit_length}: Max Collatz Length = {max_lengths_by_bits[bit_length]}")
        print("Numbers with max length in binary:")
        print(", ".join(bin(n)[2:] + " " + str(n) for n in numbers_by_max_length[bit_length]))
        for n in numbers_by_max_length[bit_length]:
            gens = list(shortcut_collatz_cycle_length_gen(n))[:-2]
            #print(gens)
            print("\n".join(bin(x)[2:] + " " + str(x) + (">" if x > n else "<") for x in gens))
        #print(", ".join(bin(n)[2:] + " " + str(n) + " b3:" + to_ternary(n) for n in numbers_by_max_length[bit_length]))
        print()

from graphviz import Digraph

def collatz_to_graphviz(sequences, output_file="collatz_graph"):
    """
    Generates a Graphviz graph for a list of Collatz sequences.
    
    Args:
        sequences (list of list of int): A list of Collatz sequences, where each sequence is a list of integers.
        output_file (str): The name of the output Graphviz file (without extension).
    
    Returns:
        None: Saves the graph to a file.
    """
    graph = Digraph(name=output_file, format="png")
    graph.attr(rankdir="LR")  # Arrange graph left-to-right for better visualization

    # Add nodes and edges for each sequence
    for sequence in sequences:
        for i in range(len(sequence) - 1):
            src = sequence[i]
            dst = sequence[i + 1]
            
            # Define node labels (decimal and binary)
            src_label = f"{src}\n{bin(src)[2:]}"
            dst_label = f"{dst}\n{bin(dst)[2:]}"
            
            # Add nodes with labels
            graph.node(name=str(src), label=src_label)
            graph.node(name=str(dst), label=dst_label)
            
            # Add edge
            graph.edge(str(src), str(dst))
    
    # Save and render the graph
    graph.render(filename=output_file, cleanup=True)
    print(f"Graph saved to {output_file}.png")

def large_number_test():
    for i in range(1, 11):
        N = 2**(10000*i) - 1
        #clen = classical_collatz_cycle_length(N)
        #print(clen)
        clen2 = shortcut_collatz_cycle_length(N)
        print(i, clen2)

if __name__ == "__main__":
    # Example Usage
    collatz_sequences = []
    for i in range(1, 2**8):
        collatz_sequences.append(list(shortcut_collatz_cycle_length_gen(i))[:-2])
    collatz_to_graphviz(collatz_sequences)

    # Specify the maximum bit length you want to analyze
    max_bit_length = 20  # Change this as needed
    max_collatz_by_length(max_bit_length)


    # The positive integer n with the largest currently known value of C, such
    # that it takes C log n iterations of the 3x + 1 function T(x) to reach 1, is
    # n = 7, 219, 136, 416, 377, 236, 271, 195 with C ≈ 36.7169 (Roosendaal [79,
    # 3x + 1 Completeness and Gamma records]).
    N = 7219136416377236271195
    clen = classical_collatz_cycle_length_div2(N)
    print(clen, clen / math.log(N))

    # Run the cleaned-up version
    #test_shortcut_vs_classical(2**20)
