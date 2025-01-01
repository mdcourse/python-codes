import numpy as np
from numba import njit

@njit
def compute_distance(position_i, positions_j, box_size):
    """Measure the distances between a single atom and its neighbors, taking into account periodic boundary conditions."""
    rij_xyz = np.remainder(position_i - positions_j + box_size[:3] / 2.0, box_size[:3]) - box_size[:3] / 2.0
    norms = np.sqrt(np.sum(rij_xyz**2))
    return norms, rij_xyz

@njit
def compute_vector_matrix(atoms_positions, box_size):
    """Matrix of vectors between all particles."""
    Nat = atoms_positions.shape[0]
    rij_matrix = np.zeros((Nat, Nat, 3))
    for Ni in range(Nat-1):
        rij_xyz = (np.remainder(atoms_positions[Ni] - atoms_positions
                                + box_size/2.0, box_size) - box_size/2.0)
        rij_matrix[Ni] = rij_xyz
    return rij_matrix
