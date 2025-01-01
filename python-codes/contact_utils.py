import numpy as np
from numba import njit

@njit
def contact_matrix(positions, cutoff, box):
    """Compute a boolean contact matrix from the particle positions."""
    n_atoms = positions.shape[0]  # number of particles
    matrix = np.zeros((n_atoms, n_atoms), dtype=np.bool_)
    for i in range(n_atoms - 1):
        for j in range(i + 1, n_atoms):
            diff = positions[i] - positions[j]  # raw distance vector between particles i and j
            diff -= np.round(diff / box[:3]) * box[:3]  # minimum image convention
            dist_sq = np.dot(diff, diff)  # squared distance between the two particles
            if dist_sq < cutoff ** 2:
                matrix[i, j] = True  # the particle i is in contact with particle j
                matrix[j, i] = True  # symmetric matrix
    return matrix

@njit
def compute_neighbor_lists(matrix):
    """Generate neighbor lists from a boolean contact matrix."""
    n_atoms = matrix.shape[0]
    neighbor_lists = []
    for cpt in range(n_atoms - 1):
        neighbor_list = np.where(matrix[cpt])[0]  # particles that are in contact with particle cpt
        neighbor_list = neighbor_list[neighbor_list > cpt]  # only keep neighbors with indices greater than cpt to avoid redundancy
        neighbor_lists.append(neighbor_list)
    return neighbor_lists
