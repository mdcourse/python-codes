import sys
import os
import unittest
import numpy as np

# Import library
sys.path.append(os.path.abspath("../molecular_simulation_code"))
from contacts_utilities import contact_matrix


class TestContactMatrix(unittest.TestCase):
    """Unit testing for contact_matrix()"""

    def test_contact_matrix_2d(self):
        """Test 2D contact matrix calculation with periodic boundary conditions."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [3.0, 3.0, 0.0]])
        cutoff = 2.0
        box = np.array([10.0, 10.0, 0.0, 90, 90, 90])

        expected_matrix = np.array([[False, True, False],
                                    [True, False, False],
                                    [False, False, False]])

        matrix = contact_matrix(positions, cutoff, box)

        np.testing.assert_array_equal(matrix, expected_matrix)

    def test_contact_matrix_3d(self):
        """Test 2D contact matrix calculation with periodic boundary conditions."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        cutoff = 2.0
        box = np.array([10.0, 10.0, 0.0, 90, 90, 90])

        expected_matrix = np.array([[False, True, False],
                                    [True, False, False],
                                    [False, False, False]])

        matrix = contact_matrix(positions, cutoff, box)

        np.testing.assert_array_equal(matrix, expected_matrix)

if __name__ == "__main__":
    unittest.main()
