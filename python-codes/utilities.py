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
def compute_distance_old(position_i, positions_j, box_size):
    """Measure the distances between two particles."""
    rij_xyz = np.remainder(position_i - positions_j
                           + box_size[:3] / 2.0, box_size[:3]) - box_size[:3] / 2.0
    norms = np.sqrt(np.sum(rij_xyz**2, axis=1))
    return norms, rij_xyz
    
@njit
def compute_distance(position_i, positions_j, box_size):
    """Measure the distances between a single atom and its neighbors, taking into account periodic boundary conditions."""
    # Compute the displacement between atom i and its neighbors (position_j is a 2D array for multiple neighbors)
    rij_xyz = np.remainder(position_i - positions_j + box_size[:3] / 2.0, box_size[:3]) - box_size[:3] / 2.0
    
    # Compute the norms (distances) between atom i and all neighbors
    norms = np.sqrt(np.sum(rij_xyz**2))  # Sum the squared displacements for each neighbor along x, y, z
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
def compute_force_vector_old(neighbor_lists, atoms_positions, box_size,
                         sigma_ij_list, epsilon_ij_list):
    """Compute the force vector between the particles."""

    total_atoms = atoms_positions.shape[0]  # Number of particles in the system
    force_vector = np.zeros((total_atoms, 3))  # Initialize the force vector with zeros

    for Ni in range(total_atoms - 1):
        neighbor_of_i = neighbor_lists[Ni]
        # Compute distances and distance vectors between particle Ni and its neighbors
        rij, rij_xyz = compute_distance(atoms_positions[Ni],
                                        atoms_positions[neighbor_of_i], box_size)

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
def compute_force_vector(neighbor_lists, atoms_positions, box_size, cross_coefficients):
    """Compute the force vector between the particles."""

    total_atoms = atoms_positions.shape[0]  # Number of particles in the system
    force_vector = np.zeros((total_atoms, 3))  # Initialize the force vector with zeros

    # Iterate over all atoms (excluding the last atom)
    for atom_i in range(total_atoms - 1):
        # Get the list of neighbors for atom_i
        neighbor_of_i = neighbor_lists[atom_i]
        
        # Loop through the neighbors of atom_i
        for atom_j in neighbor_of_i:
            # Retrieve the precomputed Lennard-Jones parameters (sigma_ij, epsilon_ij)
            sigma_ij = cross_coefficients[0][atom_i]  # Corresponding sigma_ij
            epsilon_ij = cross_coefficients[1][atom_i]  # Corresponding epsilon_ij

            # Compute the distance between atoms i and j
            rij, rij_xyz = compute_distance(atoms_positions[atom_i], atoms_positions[atom_j], box_size)
            
            # Compute the force magnitude using the Lennard-Jones potential derivative
            fij_xyz = potentials(epsilon_ij, sigma_ij, rij, derivative=True)

            # Add the force contribution for atom_i
            force_vector[atom_i] += np.sum((fij_xyz * rij_xyz.T / rij).T, axis=0)

            # Add the opposite force for atom_j (Newton's third law)
            force_vector[atom_j] -= np.sum(fij_xyz * rij_xyz.T / rij, axis=0)

    return force_vector

@njit
def compute_force_matrix_old(neighbor_lists, atoms_positions, box_size,
    sigma_ij_list, epsilon_ij_list):
    """Compute the force matrix between the particles."""

    total_atoms = atoms_positions.shape[0]  # Number of particles in the system
    force_matrix = np.zeros((total_atoms, total_atoms, 3))  #  # Initialize the force matrix with zeros

    for Ni in range(total_atoms - 1):
        neighbor_of_i = neighbor_lists[Ni]
        # Compute distances and distance vectors between particle Ni and its neighbors
        rij, rij_xyz = compute_distance(atoms_positions[Ni],
                                        atoms_positions[neighbor_of_i], box_size)

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
def compute_force_matrix(neighbor_lists, atoms_positions, box_size, cross_coefficients):
    """Compute the force matrix between the particles."""
    
    total_atoms = atoms_positions.shape[0]  # Number of particles in the system
    force_matrix = np.zeros((total_atoms, total_atoms, 3))  # Initialize the force matrix with zeros

    # Iterate over all atoms (excluding the last atom)
    for atom_i in range(total_atoms - 1):
        # Get the list of neighbors for atom_i
        neighbor_of_i = neighbor_lists[atom_i]
        
        # Loop through the neighbors of atom_i
        for atom_j in neighbor_of_i:
            # Retrieve the precomputed Lennard-Jones parameters (sigma_ij, epsilon_ij)
            sigma_ij = cross_coefficients[0][atom_i]  # Corresponding sigma_ij
            epsilon_ij = cross_coefficients[1][atom_i]  # Corresponding epsilon_ij

            # Compute the distance between atoms i and j
            rij, rij_xyz = compute_distance(atoms_positions[atom_i], atoms_positions[atom_j], box_size)
            
            # Compute the force magnitude using the Lennard-Jones potential derivative
            fij_xyz = potentials(epsilon_ij, sigma_ij, rij, derivative=True)

            # Distribute the force contributions into the force matrix
            force_matrix[atom_i, atom_j] += fij_xyz * rij_xyz / rij
            force_matrix[atom_j, atom_i] += fij_xyz * rij_xyz / rij  # Since the force is reciprocal

    return force_matrix

@njit
def compute_potential(neighbor_lists, atoms_positions, box_size, cross_coefficients):
    """Compute the potential energy by summing up all pair contributions."""
    # Iterate over all atoms
    total_atoms = atoms_positions.shape[0]  # Number of particles in the system
    energy_potential = 0.0
    
    # Iterate over each atom (excluding the last atom)
    for atom_i in range(total_atoms - 1):
        # Get the list of neighbors for atom_i
        neighbor_of_i = neighbor_lists[atom_i]
        
        # Loop through the neighbors of atom_i
        for atom_j in neighbor_of_i:
            sigma_ij = cross_coefficients[0][atom_i]  # Corresponding sigma_ij
            epsilon_ij = cross_coefficients[1][atom_i]  # Corresponding epsilon_ij
            
            # Compute the distance between atoms i and j
            rij, _ = compute_distance(atoms_positions[atom_i], atoms_positions[atom_j], box_size)
            
            # Compute and accumulate the LJ potential for this pair
            energy_potential += potentials(epsilon_ij, sigma_ij, rij)
    
    return energy_potential