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

    def wrap_in_box(self):
        """Wrap particle positions into the simulation box."""
        # Note: act on atom position, called often, does not 
        # return anything --> make sense as a method
        for dim in np.arange(3):
            out_ids = self.atoms_positions[:, dim] \
                > self.box_boundaries[dim][1]
            self.atoms_positions[:, dim][out_ids] \
                -= np.diff(self.box_boundaries[dim])[0]
            out_ids = self.atoms_positions[:, dim] \
                < self.box_boundaries[dim][0]
            self.atoms_positions[:, dim][out_ids] \
                += np.diff(self.box_boundaries[dim])[0]

    def wrap_in_box(self):
        """Wrap particle positions into the simulation box."""
        # Iterate over each spatial dimension (x, y, z)
        for dim in range(3):
            # Length of the box in the current dimension
            box_length = self.box_boundaries[dim][1] - self.box_boundaries[dim][0]
            
            # Particles outside the upper boundary
            out_ids_upper = self.atoms_positions[:, dim] > self.box_boundaries[dim][1]
            self.atoms_positions[out_ids_upper, dim] -= box_length

            # Particles outside the lower boundary
            out_ids_lower = self.atoms_positions[:, dim] < self.box_boundaries[dim][0]
            self.atoms_positions[out_ids_lower, dim] += box_length