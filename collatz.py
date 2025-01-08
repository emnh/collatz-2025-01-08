def classical_collatz_cycle_length(n):
    """Compute the Collatz cycle length for n using the classical method."""
    cycle_length = 0
    while n > 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        cycle_length += 1
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

def shortcut_collatz_cycle_length(n):
    """Compute the Collatz cycle length for n using shortcuts."""
    cycle_length = 0

    loop_iterations = 0

    while n > 1:
        
        loop_iterations += 1

        # Shortcut for trailing zeroes
        n, zero_count = truncate_and_count_zeroes(n)
        cycle_length += zero_count

        if n == 1:
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

    return cycle_length, loop_iterations

def test_shortcut_vs_classical(max_n):
    """Compare the shortcut-based and classical methods for Collatz cycle length."""
    for n in range(1, max_n + 1):
        shortcut_length, short_iters = shortcut_collatz_cycle_length(n)
        classical_length  = classical_collatz_cycle_length(n)
        if shortcut_length != classical_length:
            print(f"Mismatch for n = {n}: Shortcut = {shortcut_length}, Classical = {classical_length}")
        else:
            diff = classical_length - short_iters
            print(f"Match for n = {n}: Length = {shortcut_length}, Saved: {diff}, Loops {short_iters}")

# Run the cleaned-up version
test_shortcut_vs_classical(30)
