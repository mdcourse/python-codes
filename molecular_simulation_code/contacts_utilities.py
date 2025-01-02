"""Utilities for computing particle contact matrices and neighbor lists."""

import numpy as np
from numba import njit
from typing import List

@njit
def contact_matrix(positions: np.ndarray, cutoff: float, box: np.ndarray) -> np.ndarray:
    """Compute a boolean contact matrix from the particle positions."""
    total_atoms = positions.shape[0]  # number of particles
    matrix = np.zeros((total_atoms, total_atoms), dtype=np.bool_)
    for i in range(total_atoms - 1):
        for j in range(i + 1, total_atoms):
            diff = positions[i] - positions[j]  # raw distance vector between particles i and j
            if box[2] == 0: # 2D
                diff[:2] -= np.round(diff[:2] / box[:2]) * box[:2]  # minimum image convention in 2D
            else: # 3D
                diff -= np.round(diff / box[:3]) * box[:3]  # minimum image convention in 3D
            dist_sq = np.dot(diff, diff)  # squared distance between the two particles
            if dist_sq < cutoff ** 2:
                matrix[i, j] = True  # the particle i is in contact with particle j
                matrix[j, i] = True  # symmetric matrix
    return matrix

@njit
def compute_neighbor_lists(matrix: np.ndarray) -> List[np.ndarray]:
    """Generate neighbor lists from the contact matrix."""
    total_atoms = matrix.shape[0]
    neighbor_lists = []
    for cpt in range(total_atoms - 1):
        neighbor_list = np.where(matrix[cpt])[0]  # particles that are in contact with particle cpt
        neighbor_list = neighbor_list[neighbor_list > cpt]  # only keep neighbors with indices greater than cpt to avoid redundancy
        neighbor_lists.append(neighbor_list)
    return neighbor_lists
