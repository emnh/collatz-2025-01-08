#!/usr/bin/env python3


def simulate_collatz_on_right(R, rbits, buffer_bits, max_iterations=1000):
    """
    Simulate Collatz iterations on the integer R (the "right part")
    while ensuring that the result does not require more than
    (rbits + buffer_bits) bits. The simulation stops if the value
    would overflow into the buffer or if max_iterations is reached.

    Parameters:
      R             : initial right-part value (assumed < 2^rbits)
      rbits         : nominal bit-length of the right part
      buffer_bits   : extra bits reserved as a buffer
      max_iterations: maximum number of iterations to simulate

    Returns:
      iterations    : the number of safe Collatz iterations performed.
    """
    limit = 2 ** (rbits + buffer_bits)
    iterations = 0

    while iterations < max_iterations:
        # If R is even, safe division by 2.
        if R % 2 == 0:
            R //= 2
        else:
            # For odd numbers, compute 3n + 1 and check if it's within the safe limit.
            R_next = 3 * R + 1
            if R_next >= limit:
                break  # Operation would overflow the reserved space.
            R = R_next
        iterations += 1

    return iterations

def test_entanglement(rbits, buffer_bits, start=1, end=256, max_iterations=1000):
    """
    Tests a limited range of candidate starting values for the right part
    (by default, odd numbers in [1, 256)) and returns the one that can run
    the most safe iterations before causing an overflow.

    Parameters:
      rbits         : nominal bit-length for the right part.
      buffer_bits   : number of zeros reserved as the buffer.
      start         : starting candidate value.
      end           : ending candidate value (non-inclusive).
      max_iterations: maximum iterations to simulate for each candidate.

    Returns:
      best_R  : the candidate initial right part with the maximum safe iterations.
      max_iters: the corresponding number of safe iterations.
    """
    max_iters = 0
    best_R = None

    # Iterate only over odd values (even ones quickly reduce by division).
    for R in range(start, end, 2):
        iters = simulate_collatz_on_right(R, rbits, buffer_bits, max_iterations)
        if iters > max_iters:
            max_iters = iters
            best_R = R

    return best_R, max_iters

if __name__ == '__main__':
    # Example parameters:
    rbits = 8        # Use 8 bits for the right part (values 0 to 255)
    buffer_bits = 4  # Reserve 4 additional bits as a buffer.

    # Limit the candidate range to [1,256) and iterations to 1000.
    best_R, max_iters = test_entanglement(rbits, buffer_bits, start=1, end=256, max_iterations=1000)
    print(f"For rbits = {rbits} and buffer_bits = {buffer_bits}:")
    print(f"  Best initial right-part value: {best_R} (binary: {bin(best_R)})")
    print(f"  Achieves {max_iters} safe Collatz iterations before potential overflow.")

