"""Utilities for calculating interatomic distances and vector matrices
with periodic boundary conditions."""

import numpy as np
from numba import njit
from typing import Tuple

@njit
def compute_distance(position_i: np.ndarray, positions_j: np.ndarray, box: np.ndarray) -> Tuple[float, np.ndarray]:
    """Measure the distances between a single atom and its neighbors, taking into account periodic boundary conditions."""
    if box[2] == 0:  # 2D
        rij_xyz = np.zeros(3)   # Ensure 3 components for each vector
        rij_xyz[:2] = np.remainder(position_i[:2] - positions_j[:2] + box[:2] / 2.0, box[:2]) - box[:2] / 2.0
    else:  # 3D
        rij_xyz = np.remainder(position_i - positions_j + box[:3] / 2.0, box[:3]) - box[:3] / 2.0
    norms = np.sqrt(np.sum(rij_xyz**2))
    return norms, rij_xyz

@njit
def compute_vector_matrix(atoms_positions: np.ndarray, box_size: np.ndarray) -> np.ndarray:
    """Matrix of vectors between all particles."""
    total_atoms = atoms_positions.shape[0]
    rij_matrix = np.zeros((total_atoms, total_atoms, 3))

    if box_size[2] == 0:  # 2D
        for Ni in range(total_atoms - 1):
            rij_xyz = np.zeros((total_atoms, 3))  # Ensure 3 components for each vector
            rij_xyz[:, :2] = (np.remainder(atoms_positions[Ni, :2] - atoms_positions[:, :2]
                                           + box_size[:2] / 2.0, box_size[:2]) - box_size[:2] / 2.0)
            rij_matrix[Ni] = rij_xyz
            rij_matrix[:, Ni] = -rij_xyz
    else:  # 3D
        for Ni in range(total_atoms - 1):
            rij_xyz = (np.remainder(atoms_positions[Ni] - atoms_positions
                                    + box_size / 2.0, box_size) - box_size / 2.0)
            rij_matrix[Ni] = rij_xyz
            rij_matrix[:, Ni] = -rij_matrix[Ni]
    return rij_matrix