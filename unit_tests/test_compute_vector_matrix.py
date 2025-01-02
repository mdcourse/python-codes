import sys
import os
import unittest
import numpy as np

# Import library
sys.path.append(os.path.abspath("../molecular_simulation_code"))
from distances_utilities import compute_vector_matrix


class TestComputeVectorMatrix(unittest.TestCase):
    """Unit testing for compute_vector_matrix()"""

    def test_2d_vector_matrix(self):
        """Test 2D vector matrix calculation with periodic boundary conditions."""
        atoms_positions = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
        box_size = np.array([10.0, 10.0, 0.0])

        expected_rij_matrix = np.array([[[ 0.0, 0.0, 0.0], [-1.0, -1.0, 0.0]],
                                        [[1.0, 1.0, 0.0], [ 0.0, 0.0, 0.0]]])

        rij_matrix = compute_vector_matrix(atoms_positions, box_size)

        np.testing.assert_array_almost_equal(rij_matrix, expected_rij_matrix, decimal=6)

    def test_3d_vector_matrix(self):
        """Test 3D vector matrix calculation with periodic boundary conditions."""
        atoms_positions = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        box_size = np.array([10.0, 10.0, 10.0])

        expected_rij_matrix = np.array([[[ 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]],
                                        [[1.0, 1.0, 1.0], [ 0.0, 0.0, 0.0]]])

        rij_matrix = compute_vector_matrix(atoms_positions, box_size)

        np.testing.assert_array_almost_equal(rij_matrix, expected_rij_matrix, decimal=6)


if __name__ == "__main__":
    unittest.main()
