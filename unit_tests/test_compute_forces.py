import sys
import os
import unittest

# Import library
sys.path.append(os.path.abspath("../molecular_simulation_code"))
from potentials_utilities import compute_forces


class TestComputeForces(unittest.TestCase):
    """Unit testing for compute_forces()"""
    def test_lennard_jones_force(self):
        """Test Lennard-Jones force calculation."""
        epsilon = 1.0
        sigma = 1.0
        r = 2.0
        potential_type = "LJ"
        
        # Calculate expected force using the formula for LJ force
        expected = 48 * epsilon * ((sigma / r) ** 12 - 0.5 * (sigma / r) ** 6) / r
        result = compute_forces(epsilon, sigma, r, potential_type)
        
        # Assert that the computed force is close to the expected value
        self.assertAlmostEqual(result, expected, places=6)

    def test_hard_sphere_force(self):
        """Test Hard-Sphere force calculation."""
        epsilon = 1.0
        sigma = 1.0
        r = 2.0  # Any r > sigma will return 0
        potential_type = "HS"
        
        expected = 0.0  # Hard-Sphere force is zero for r > sigma
        result = compute_forces(epsilon, sigma, r, potential_type)
        
        self.assertEqual(result, expected)

    def test_invalid_potential_type(self):
        """Test invalid potential type input."""
        epsilon = 1.0
        sigma = 1.0
        r = 1.5
        potential_type = "INVALID"
        
        # Expect ValueError for unsupported potential_type
        with self.assertRaises(ValueError):  
            compute_forces(epsilon, sigma, r, potential_type)

    def test_zero_distance_lj(self):
        """Test force calculation for zero distance."""
        epsilon = 1.0
        sigma = 1.0
        r = 0.0
        potential_type = "LJ"
        
        # Expect division by zero error if r == 0
        with self.assertRaises(ZeroDivisionError):  
            compute_forces(epsilon, sigma, r, potential_type)

if __name__ == "__main__":
    unittest.main()
