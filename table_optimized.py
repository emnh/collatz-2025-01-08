
from collatz import classical_collatz_cycle_length
from stats import collatz_sequence

class CollatzTable:
    def __init__(self, high_bits: int, low_bits: int):
        self.high_bits = high_bits
        self.low_bits = low_bits
        self.table_size = 1 << low_bits  # 2^low_bits
        self.B = [0] * self.table_size
        self.C = [0] * self.table_size
        self._initialize_tables()

    def _initialize_tables(self):
        """Precompute the B and C tables for all possible values of n_L."""
        for n_L in range(self.table_size):
            b = 1 << self.low_bits  # Initially, b = 2^d
            c = n_L  # Initially, c = n_L

            while b % 2 == 0 and c % 2 == 0:  # Apply even rule: divide both by 2
                b //= 2
                c //= 2

            while c % 2 == 1:  # Apply odd rule: triple b, triple c, and add 1
                b *= 3
                c = 3 * c + 1

            self.B[n_L] = b
            self.C[n_L] = c

    def compute_next(self, n: int) -> int:
        """Compute the next value in the Collatz sequence using the tables."""
        # Split n into high and low parts
        n_H = n >> self.low_bits  # High bits
        n_L = n & ((1 << self.low_bits) - 1)  # Low bits

        # Lookup tables and compute next n
        next_n = self.B[n_L] * n_H + self.C[n_L]
        return next_n

    def compute_sequence(self, n: int, steps: int) -> list:
        """Compute a sequence of Collatz steps."""
        sequence = [n]
        for _ in range(steps):
            n = self.compute_next(n)
            sequence.append(n)
        return sequence

# Example usage
if __name__ == "__main__":
    high_bits = 4  # Number of bits for the high part
    low_bits = 4   # Number of bits for the low part
    collatz = CollatzTable(high_bits, low_bits)

    # Compute a sequence starting from a number
    #start_number = 213  # Example starting number
    start_number = 27
    steps = 50          # Number of steps to compute

    seq = collatz_sequence(start_number)
    print(*seq)

    for num in seq:
        sequence = collatz.compute_sequence(num, steps)
        print("Collatz sequence:", sequence)
