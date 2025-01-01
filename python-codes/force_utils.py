"""Utilities for computing interparticle force vectors and matrices 
using Lennard-Jones potentials and periodic boundary conditions."""

import numpy as np
from numba import njit
from pot_utils import LJ_potentials
from distance_utils import compute_distance

@njit
def compute_force_vector(neighbor_lists, atoms_positions, box_size, cross_coefficients):
    """Compute the force vector between the particles."""
    total_atoms = atoms_positions.shape[0]
    force_vector = np.zeros((total_atoms, 3))
    for atom_i in range(total_atoms - 1):
        neighbor_of_i = neighbor_lists[atom_i]
        for atom_j in neighbor_of_i:
            sigma_ij = cross_coefficients[0][atom_i]
            epsilon_ij = cross_coefficients[1][atom_i]
            rij, rij_xyz = compute_distance(atoms_positions[atom_i], atoms_positions[atom_j], box_size)
            fij_xyz = LJ_potentials(epsilon_ij, sigma_ij, rij, derivative=True)
            force_vector[atom_i] += np.sum((fij_xyz * rij_xyz.T / rij).T, axis=0)
            force_vector[atom_j] -= np.sum(fij_xyz * rij_xyz.T / rij, axis=0)
    return force_vector

@njit
def compute_force_matrix(neighbor_lists, atoms_positions, box_size, cross_coefficients):
    """Compute the force matrix between the particles."""
    total_atoms = atoms_positions.shape[0]
    force_matrix = np.zeros((total_atoms, total_atoms, 3))
    for atom_i in range(total_atoms - 1):
        neighbor_of_i = neighbor_lists[atom_i]
        for atom_j in neighbor_of_i:
            sigma_ij = cross_coefficients[0][atom_i]
            epsilon_ij = cross_coefficients[1][atom_i]
            rij, rij_xyz = compute_distance(atoms_positions[atom_i], atoms_positions[atom_j], box_size)
            fij_xyz = LJ_potentials(epsilon_ij, sigma_ij, rij, derivative=True)
            force_matrix[atom_i, atom_j] += fij_xyz * rij_xyz / rij
            force_matrix[atom_j, atom_i] += fij_xyz * rij_xyz / rij
    return force_matrix

