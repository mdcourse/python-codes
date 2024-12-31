import numpy as np
import pint
from numba import njit
from potentials import potentials


def validate_units(value: pint.Quantity, expected_unit: pint.Unit, name: str):
    """Validate the units of a given quantity."""
    if not isinstance(value, pint.Quantity):
        raise TypeError(f"Invalid type for {name}: expected a 'pint.Quantity', got '{type(value).__name__}'.")
    if value.units != expected_unit:
        raise ValueError(f"Invalid units for {name}: expected {expected_unit}, got {value.units}")

def nondimensionalize_single(value: pint.Quantity, ref_value: pint.Quantity) -> float:
    """Nondimensionalize a single quantity."""
    return (value / ref_value).magnitude

def nondimensionalize_array(array: np.ndarray, ref_value: pint.Quantity) -> np.ndarray:
    """Nondimensionalize a NumPy array."""
    return (array / ref_value).magnitude

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
    # Generate neighbor lists from a boolean contact matrix.
    n_atoms = matrix.shape[0]  # empty list to store the neighbor lists for each particle
    neighbor_lists = []
    for cpt in range(n_atoms - 1):
        neighbor_list = np.where(matrix[cpt])[0]  # particles that are in contact with particle cpt
        neighbor_list = neighbor_list[neighbor_list > cpt]  # only keep neighbors with indices greater than cpt to avoid redundancy
        neighbor_lists.append(neighbor_list)
    return neighbor_lists

@njit
def compute_distance(position_i, positions_j, box_size, only_norm = True):
    """Measure the distances between two particles."""
    rij_xyz = np.remainder(position_i - positions_j
                           + box_size[:3] / 2.0, box_size[:3]) - box_size[:3] / 2.0
    norms = np.sqrt(np.sum(rij_xyz**2, axis=1))
    if only_norm:
        return norms, None
    else:
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

@njit
def compute_force_vector(neighbor_lists, atoms_positions, box_size,
                         sigma_ij_list, epsilon_ij_list):
    """Compute the force vector between the particles."""

    total_atoms = atoms_positions.shape[0]  # Number of particles in the system
    force_vector = np.zeros((total_atoms, 3))  # Initialize the force vector with zeros

    for Ni in range(total_atoms - 1):
        neighbor_of_i = neighbor_lists[Ni]
        # Compute distances and distance vectors between particle Ni and its neighbors
        rij, rij_xyz = compute_distance(atoms_positions[Ni],
                                        atoms_positions[neighbor_of_i],
                                        box_size, only_norm=False)

        # Retrieve precomputed Lennard-Jones parameters
        sigma_ij = sigma_ij_list[Ni]
        epsilon_ij = epsilon_ij_list[Ni]

        # Compute the force magnitude using the Lennard-Jones potential derivative
        fij_xyz = potentials(epsilon_ij, sigma_ij, rij, derivative=True)

        # Add the contributions to the force vector
        force_vector[Ni] += np.sum((fij_xyz * rij_xyz.T / rij).T, axis=0)
        for idx, neighbor in enumerate(neighbor_of_i):
            force_vector[neighbor] -= fij_xyz[idx] * rij_xyz[idx] / rij[idx]

    return force_vector

@njit
def compute_force_matrix(neighbor_lists, atoms_positions, box_size,
    sigma_ij_list, epsilon_ij_list):
    """Compute the force matrix between the particles."""

    total_atoms = atoms_positions.shape[0]  # Number of particles in the system
    force_matrix = np.zeros((total_atoms, total_atoms, 3))  #  # Initialize the force matrix with zeros

    for Ni in range(total_atoms - 1):
        neighbor_of_i = neighbor_lists[Ni]
        # Compute distances and distance vectors between particle Ni and its neighbors
        rij, rij_xyz = compute_distance(atoms_positions[Ni],
                                        atoms_positions[neighbor_of_i],
                                        box_size, only_norm=False)

        # Retrieve precomputed Lennard-Jones parameters
        sigma_ij = sigma_ij_list[Ni]
        epsilon_ij = epsilon_ij_list[Ni]

        # Compute the force magnitude using the Lennard-Jones potential derivative
        fij_xyz = potentials(epsilon_ij, sigma_ij, rij, derivative=True)

        # Distribute the force contributions into the force matrix
        for idx, neighbor in enumerate(neighbor_of_i):
            force_matrix[Ni, neighbor] += fij_xyz[idx] * rij_xyz[idx] / rij[idx]

    return force_matrix

@njit
def compute_potential(neighbor_lists, atoms_positions, box_size,
                      sigma_ij_list, epsilon_ij_list):
    """Compute the potential energy by summing up all pair contributions."""
    total_atoms = atoms_positions.shape[0]  # Number of particles in the system
    energy_potential = 0
    for Ni in np.arange(total_atoms-1):
        # Read neighbor list
        neighbor_of_i = neighbor_lists[Ni]
        # Measure distance
        rij, _ = compute_distance(atoms_positions[Ni],
                                  atoms_positions[neighbor_of_i],
                                  box_size, only_norm=True)
        # Measure potential using pre-calculated cross coefficients
        sigma_ij = sigma_ij_list[Ni]
        epsilon_ij = epsilon_ij_list[Ni]
        energy_potential += np.sum(potentials(epsilon_ij, sigma_ij, rij))
    return energy_potential

"""
@njit
def wrap_in_box(atoms_positions, box_boundaries):
    #Wrap particle positions into the simulation box.
    # Iterate over each spatial dimension
    for dim in range(3):
        # Particles outside the upper boundary
        out_ids = atoms_positions[:, dim] > box_boundaries[dim, 1]
        atoms_positions[out_ids, dim] -= box_boundaries[dim, 1] - box_boundaries[dim, 0]

        # Particles outside the lower boundary
        out_ids = atoms_positions[:, dim] < box_boundaries[dim, 0]
        atoms_positions[out_ids, dim] += box_boundaries[dim, 1] - box_boundaries[dim, 0]
    return atoms_positions
"""