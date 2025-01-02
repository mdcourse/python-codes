"""Utilities for computing interparticle force vectors and matrices 
using Lennard-Jones potentials and periodic boundary conditions."""

import warnings
import numpy as np
from numba import njit
from potentials_utilities import compute_forces
from distances_utilities import compute_distance
from typing import List


@njit
def compute_force_vector(neighbor_lists: List[np.ndarray], atoms_positions: np.ndarray, 
                         box_size: np.ndarray, cross_coefficients: np.ndarray,
                         potential_type: str) -> np.ndarray:
    """Compute the force vector between the particles."""
    # Initialize the force vector for all atoms
    total_atoms = atoms_positions.shape[0]
    force_vector = np.zeros((total_atoms, 3))

    # Loop through each atom and its neighbors
    for atom_i in range(total_atoms - 1):
        neighbor_of_i = neighbor_lists[atom_i]
        for atom_j in neighbor_of_i:
            # Extract interaction coefficients for atom pairs
            sigma_ij = cross_coefficients[0][atom_i]
            epsilon_ij = cross_coefficients[1][atom_i]

            # Compute distance and direction vector between atoms i and j
            rij, rij_xyz = compute_distance(atoms_positions[atom_i], atoms_positions[atom_j], box_size)

            # Calculate forces
            fij_xyz = compute_forces(epsilon_ij, sigma_ij, rij, potential_type)

            # Update force matrix
            force_vector[atom_i] += np.sum((fij_xyz * rij_xyz.T / rij).T, axis=0)
            force_vector[atom_j] -= np.sum(fij_xyz * rij_xyz.T / rij, axis=0)
    return force_vector

@njit
def compute_force_matrix(neighbor_lists: List[np.ndarray], atoms_positions: np.ndarray, 
                         box_size: np.ndarray, cross_coefficients: np.ndarray,
                         potential_type: str) -> np.ndarray:
    """Compute the force matrix between the particles."""
    # Initialize the force matrix for all atoms
    total_atoms = atoms_positions.shape[0]
    force_matrix = np.zeros((total_atoms, total_atoms, 3))

    # Loop through each atom and its neighbors
    for atom_i in range(total_atoms - 1):
        neighbor_of_i = neighbor_lists[atom_i]
        for atom_j in neighbor_of_i:
            # Extract interaction coefficients for atom pairs
            sigma_ij = cross_coefficients[0][atom_i]
            epsilon_ij = cross_coefficients[1][atom_i]

            # Compute distance and direction vector between atoms i and j
            rij, rij_xyz = compute_distance(atoms_positions[atom_i], atoms_positions[atom_j], box_size)
            
            # Calculate forces
            fij_xyz = compute_forces(epsilon_ij, sigma_ij, rij, potential_type)
            
            # Update force matrix
            force_matrix[atom_i, atom_j] += fij_xyz * rij_xyz / rij
            force_matrix[atom_j, atom_i] += fij_xyz * rij_xyz / rij
    return force_matrix

