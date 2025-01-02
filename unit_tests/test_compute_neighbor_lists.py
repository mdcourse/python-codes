import sys
import os
import unittest
import numpy as np

# Import library
sys.path.append(os.path.abspath("../molecular_simulation_code"))
from contacts_utilities import compute_neighbor_lists


class TestComputeNeighborLists(unittest.TestCase):
    """Unit testing for compute_neighbor_lists()"""

    def test_generate_neighbor_lists(self):
        """Test neighbor list generation from a contact matrix."""
        matrix = np.array([[False, True, False],
                           [True, False, False],
                           [False, False, False]], dtype=bool)

        expected_neighbor_lists = [np.array([1]), np.array([])]
        
        # Compute the neighbor lists
        neighbor_lists = compute_neighbor_lists(matrix)
        
        # Verify that the computed neighbor lists match the expected result
        for i, neighbors in enumerate(neighbor_lists):
            np.testing.assert_array_equal(neighbors, expected_neighbor_lists[i])

if __name__ == "__main__":
    unittest.main()
