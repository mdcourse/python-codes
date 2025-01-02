import sys
import os
import unittest

# Import library
sys.path.append(os.path.abspath("../molecular_simulation_code"))
from potentials_utilities import compute_potentials


class TestComputePotentials(unittest.TestCase):
    """Unit testing for compute_potentials()"""
    def test_lennard_jones_potential(self):
        epsilon = 1.0
        sigma = 1.0
        r = 1.5
        potential_type = "LJ"
        
        expected = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
        result = compute_potentials(epsilon, sigma, r, potential_type)
        self.assertAlmostEqual(result, expected, places=6)

    def test_hard_sphere_potential(self):
        epsilon = 1.0
        sigma = 1.0
        for r, expected in zip([0.5, 1.5], [epsilon, 0.0]):
            potential_type = "HS"
            
            result = compute_potentials(epsilon, sigma, r, potential_type)
            self.assertAlmostEqual(result, expected, places=6)

    def test_invalid_potential_type(self):
        epsilon = 1.0
        sigma = 1.0
        r = 1.5
        potential_type = "INVALID"
        
        with self.assertRaises(Exception):  # Expect an error for unsupported potential_type
            compute_potentials(epsilon, sigma, r, potential_type)

if __name__ == "__main__":
    unittest.main()
