import unittest
import numpy as np
from matrix2 import (
    generate_binary_matrix,
    to_digit_vector,
    from_digit_vector,
    propagate_carries,
    propagate_carries_div_2,
    matrix,
    collatz_matrix,
    classical_collatz_cycle_length
)

class TestMatrixFunctions(unittest.TestCase):

    def test_generate_binary_matrix(self):
        result = generate_binary_matrix(3)
        expected = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])
        self.assertTrue((result == expected).all())

    def test_to_digit_vector(self):
        result = to_digit_vector(45, 3)
        expected = np.array([0, 0, 1, 2, 1])  # Reversed order!
        self.assertTrue((result == expected).all())

    def test_from_digit_vector(self):
        digits = np.array([1, 2, 0, 0])  # 12 in base 3
        result = from_digit_vector(digits, 3)
        expected = 45
        self.assertEqual(result, expected)

    def test_propagate_carries(self):
        digits = np.array([10, 1, 5])  # Some large digit values
        result = propagate_carries(digits, 10)
        expected = [0, 2, 5]  # Carry propagation example
        self.assertEqual(result, expected)

    def test_propagate_carries_div_2(self):
        digits = np.array([4, 2, 6, 1])
        result = propagate_carries_div_2(digits, 2)
        expected = [1, 1]  # Expected after divide-2 propagation
        self.assertEqual(result, expected)

    def test_matrix(self):
        try:
            matrix(27)
        except Exception as e:
            self.fail(f"matrix(27) raised an exception {e}")

    def test_collatz_matrix(self):
        result = collatz_matrix(27)
        self.assertTrue(isinstance(result, list))

    def test_classical_collatz_cycle_length(self):
        n = 27
        result = list(classical_collatz_cycle_length(n))
        expected = [27, 82, 41, 124, 62, 31, 94, 47, 142, 71, 214, 107, 322, \
                    161, 484, 242, 121, 364, 182, 91, 274, 137, 412, 206, 103, 310, 155, \
                    466, 233, 700, 350, 175, 526, 263, 790, 395, 1186, 593, 1780, 890, 445, \
                    1336, 668, 334, 167, 502, 251, 754, 377, 1132, 566, 283, 850, 425, 1276, \
                    638, 319, 958, 479, 1438, 719, 2158, 1079, 3238, 1619, 4858, 2429, 7288, \
                    3644, 1822, 911, 2734, 1367, 4102, 2051, 6154, 3077, 9232, 4616, 2308, \
                    1154, 577, 1732, 866, 433, 1300, 650, 325, 976, 488, 244, 122, 61, 184, \
                    92, 46, 23, 70, 35, 106, 53, 160, 80, 40, 20, 10, 5, 16, 8, 4, 2, 1]
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
