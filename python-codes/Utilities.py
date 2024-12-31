import numpy as np
from MDAnalysis.analysis import distances


from potentials import potentials
from utilities import contact_matrix, compute_neighbor_lists, compute_distance

class Utilities:
    def __init__(self,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)

    def update_neighbor_lists(self, force_update=False):
        """Update the neighbor lists."""
        if (self.step % self.neighbor == 0) or force_update:  # Check if an update is needed
            # Compute the contact matrix based on current particle positions
            matrix = contact_matrix(self.atoms_positions,
                                    cutoff=self.cut_off,
                                    box=self.box_size)
            # Compute the neighbor lists from the contact matrix
            self.neighbor_lists = compute_neighbor_lists(matrix)

    def update_cross_coefficients(self, force_update=False):
        if (self.step % self.neighbor == 0) | force_update:
            # Precalculte LJ cross-coefficients
            sigma_ij_list = []
            epsilon_ij_list = []
            for Ni in np.arange(np.sum(self.number_atoms)-1): # tofix error for GCMC
                # Read information about atom i
                sigma_i = self.atoms_sigma[Ni]
                epsilon_i = self.atoms_epsilon[Ni]
                neighbor_of_i = self.neighbor_lists[Ni]
                # Read information about neighbors j
                sigma_j = self.atoms_sigma[neighbor_of_i]
                epsilon_j = self.atoms_epsilon[neighbor_of_i]
                # Calculare cross parameters
                sigma_ij_list.append((sigma_i+sigma_j)/2)
                epsilon_ij_list.append((epsilon_i+epsilon_j)/2)
            self.sigma_ij_list = sigma_ij_list
            self.epsilon_ij_list = epsilon_ij_list

    def compute_potential(self):
        """Compute the potential energy by summing up all pair contributions."""
        energy_potential = 0
        for Ni in np.arange(np.sum(self.number_atoms)-1):
            # Read neighbor list
            neighbor_of_i = self.neighbor_lists[Ni]
            # Measure distance
            rij, _ = compute_distance(self.atoms_positions[Ni],
                                      self.atoms_positions[neighbor_of_i],
                                      self.box_size)
            # Measure potential using pre-calculated cross coefficients
            sigma_ij = self.sigma_ij_list[Ni]
            epsilon_ij = self.epsilon_ij_list[Ni]
            energy_potential += np.sum(potentials(epsilon_ij, sigma_ij, rij))
        return energy_potential

    def wrap_in_box(self):
        for dim in np.arange(3):
            out_ids = self.atoms_positions[:, dim] \
                > self.box_boundaries[dim][1]
            self.atoms_positions[:, dim][out_ids] \
                -= np.diff(self.box_boundaries[dim])[0]
            out_ids = self.atoms_positions[:, dim] \
                < self.box_boundaries[dim][0]
            self.atoms_positions[:, dim][out_ids] \
                += np.diff(self.box_boundaries[dim])[0]
