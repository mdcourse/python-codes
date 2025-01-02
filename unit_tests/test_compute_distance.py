import sys
import os
import unittest
import numpy as np

# Import library
sys.path.append(os.path.abspath("../molecular_simulation_code"))
from distances_utilities import compute_distance


class TestComputeDistance(unittest.TestCase):
    """Unit testing for compute_distance()"""

    def test_2d_distance_close_points(self):
        """Test 2D distance calculation with periodic boundary conditions."""
        position_i = np.array([2.0, 2.0, 0.0])  # 2D position of atom i
        position_j = np.array([1.0, 1.0, 0.0])  # 2D position of atom j
        box = np.array([10.0, 10.0, 0.0, 90, 90, 90])  # MDA box

        expected_rij = np.array([1.0, 1.0, 0.0])
        expected_norm = np.sqrt(2)

        norm, rij_xyz = compute_distance(position_i, position_j, box)

        # Assert that the computed distance norm is close to the expected value
        self.assertAlmostEqual(norm, expected_norm, places=6)
        np.testing.assert_array_almost_equal(rij_xyz, expected_rij, decimal=6)

    def test_2d_distance_far_points(self):
        """Test 2D distance calculation with periodic boundary conditions."""
        position_i = np.array([9.0, 0.0, 0.0])  # 2D position of atom i
        position_j = np.array([1.0, 0.0, 0.0])  # 2D position of atom j
        box = np.array([10.0, 10.0, 0.0, 90, 90, 90])  # MDA box

        expected_rij = np.array([-2.0, 0.0, 0.0])  # Expected distance vector, considering periodic boundaries
        expected_norm = 2.0  # Expected distance norm

        norm, rij_xyz = compute_distance(position_i, position_j, box)

        # Assert that the computed distance norm is close to the expected value
        self.assertAlmostEqual(norm, expected_norm, places=6)
        np.testing.assert_array_almost_equal(rij_xyz, expected_rij, decimal=6)

    def test_3d_distance(self):
        """Test 3D distance calculation with periodic boundary conditions."""
        position_i = np.array([1.0, 1.0, 1.0])  # 3D position of atom i
        position_j = np.array([2.0, 2.0, 2.0])  # 3D position of atom j
        box = np.array([10.0, 10.0, 10.0])  # Box size (3D)

        expected_rij = np.array([-1.0, -1.0, -1.0])
        expected_norm = np.sqrt(3)

        norm, rij_xyz = compute_distance(position_i, position_j, box)

        # Assert that the computed distance norm is close to the expected value
        self.assertAlmostEqual(norm, expected_norm, places=6)
        np.testing.assert_array_almost_equal(rij_xyz, expected_rij, decimal=6)

    def test_zero_distance(self):
        """Test case where positions of atoms are identical (distance = 0)."""
        position_i = np.array([1.0, 1.0, 1.0]) 
        position_j = np.array([1.0, 1.0, 1.0])
        box = np.array([10.0, 10.0, 10.0])

        expected_rij = np.array([0.0, 0.0, 0.0])
        expected_norm = 0.0

        norm, rij_xyz = compute_distance(position_i, position_j, box)

        self.assertAlmostEqual(norm, expected_norm, places=6)
        np.testing.assert_array_almost_equal(rij_xyz, expected_rij, decimal=6)

if __name__ == "__main__":
    unittest.main()
