"""Utilities for computing Lennard-Jones potentials and total potential energy."""

from numba import njit
import numpy as np
from typing import List

from distance_utils import compute_distance

@njit
def LJ_potentials(epsilon: float, sigma: float, r: float, derivative: bool = False) -> float:
    """Compute the Lennard-Jones (LJ) potential or its derivative for a pair of atoms."""
    if derivative:  # Compute the derivative of the Lennard-Jones potential
        return 48 * epsilon * ((sigma / r) ** 12 - 0.5 * (sigma / r) ** 6) / r
    else:  # Compute the Lennard-Jones potential itself
        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

@njit
def compute_potential(neighbor_lists: List[np.ndarray], atoms_positions: np.ndarray, 
                      box_size: np.ndarray, cross_coefficients: np.ndarray) -> float:
    """Compute the potential energy by summing up all pair contributions."""
    total_atoms = atoms_positions.shape[0]
    energy_potential = 0.0
    for atom_i in range(total_atoms - 1):
        neighbor_of_i = neighbor_lists[atom_i]
        for atom_j in neighbor_of_i:
            sigma_ij = cross_coefficients[0][atom_i]
            epsilon_ij = cross_coefficients[1][atom_i]
            rij, _ = compute_distance(atoms_positions[atom_i], atoms_positions[atom_j], box_size)
            energy_potential += LJ_potentials(epsilon_ij, sigma_ij, rij)
    return energy_potential

