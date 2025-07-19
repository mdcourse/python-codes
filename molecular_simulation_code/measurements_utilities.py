from numba import njit
import numpy as np
from numba.typed import List

from distances_utilities import compute_vector_matrix
from forces_utilities import compute_force_matrix
from potentials_utilities import compute_potentials
from distances_utilities import compute_distance


@property
def potential_type_int(self):
    if self.potential_type == "LJ":
        return 0
    elif self.potential_type == "HS":
        return 1
    else:
        raise ValueError(f"Unknown potential_type: {self.potential_type}")

@njit
def _compute_epot(
    neighbor_lists,  # Numba typed list of np.ndarray
    atoms_positions: np.ndarray, 
    box_size: np.ndarray, 
    cross_sigmas: np.ndarray, 
    cross_epsilons: np.ndarray, 
    potential_type: int = 0
) -> float:
    """
    Compute the potential energy by summing up all pair contributions.

    potential_type:
        0 = LJ
        1 = HS
    """

    total_atoms = atoms_positions.shape[0]
    energy_potential = 0.0
    for atom_i in range(total_atoms - 1):
        neighbor_of_i = neighbor_lists[atom_i]
        for atom_j in neighbor_of_i:
            sigma_ij = cross_sigmas[atom_i]
            epsilon_ij = cross_epsilons[atom_i]
            rij, _ = compute_distance(atoms_positions[atom_i], atoms_positions[atom_j], box_size)
            energy_potential += compute_potentials(epsilon_ij, sigma_ij, rij, 0)

    return energy_potential

@njit
def compute_epot(
    neighbor_lists,  # tuple: (neighbors_array, neighbor_counts_array)
    atoms_positions: np.ndarray, 
    box_size: np.ndarray, 
    cross_sigmas: np.ndarray, 
    cross_epsilons: np.ndarray, 
    potential_type: int = 0
) -> float:
    total_atoms = atoms_positions.shape[0]
    neighbors, neighbor_counts = neighbor_lists
    energy_potential = 0.0

    for atom_i in range(total_atoms):
        count_i = neighbor_counts[atom_i]
        for idx in range(count_i):
            atom_j = neighbors[atom_i, idx]
            sigma_ij = cross_sigmas[atom_i, atom_j]
            epsilon_ij = cross_epsilons[atom_i, atom_j]
            rij, _ = compute_distance(atoms_positions[atom_i], atoms_positions[atom_j], box_size)            
            energy_potential += compute_potentials(epsilon_ij, sigma_ij, rij, 0)

    return energy_potential

# @njit
def compute_pressure(atoms_positions, box_mda, neighbor_lists,
                     cross_coefficients, potential_type, temperature = None):
    """Evaluate p based on the Virial equation (Eq. 4.4.2 in Frenkel-Smit,
    Understanding molecular simulation: from algorithms to applications, 2002)"""

    # Compute the ideal contribution
    total_atoms = atoms_positions.shape[0] # total number of atoms
    if box_mda[2] == 0: # 2D
        dimension = 2 # 2D
        volume = np.prod(box_mda[:2]) # box volume
    else:
        dimension = 3 # 3D
        volume = np.prod(box_mda[:3]) # box volume
    Ndof = dimension*total_atoms-dimension    
    if temperature is None: # for MC, simply use the desired temperature as input
        print("TO BE IMPLEMENTED")
        # temperature = calculate_temperature() #TODO this is for later on, when velocities are computed
    p_ideal = Ndof*temperature/(volume*dimension)

    # Compute the non-ideal contribution
    vector_matrix = compute_vector_matrix(atoms_positions, box_mda[:3])
    force_matrix = compute_force_matrix(neighbor_lists, atoms_positions, box_mda,
                                        cross_coefficients, potential_type)

    distances_forces = np.sum(force_matrix*vector_matrix)
    p_nonideal = distances_forces/(volume*dimension)

    # Final pressure
    pressure = p_ideal+p_nonideal
    return pressure

@njit
def compute_density(self):
    """Calculate the mass density."""
    # TOFIX: not used yet
    volume = np.prod(self.box_mda[:3])  # Unitless
    total_mass = np.sum(self.atoms_mass)  # Unitless
    return total_mass/volume  # Unitless
