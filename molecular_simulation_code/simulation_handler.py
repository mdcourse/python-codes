import numpy as np
from numba.typed import List

# from contacts_utilities import contact_matrix, compute_neighbor_lists
from contacts_utilities import compute_neighbor_lists


class Utilities:
    def __init__(self,
                *args,  # SG: remove ? unnecessary unless you're planning to inherit from another class in the future
                **kwargs):  # SG: remove ? unnecessary unless you're planning to inherit from another class in the future
        super().__init__(*args, **kwargs)  # SG: remove ? unnecessary unless you're planning to inherit from another class in the future

    def update_neighbor_lists(self, force_update=False):
        """Update the neighbor lists based on contact analysis."""
        if (self.step % self.neighbor == 0) or force_update:  # Check if an update is needed

            # Compute the neighbor lists from the contact matrix
            self.neighbor_lists = compute_neighbor_lists(self.positions,
                                                           self.cut_off,
                                                           self.box_mda,
                                                           1000)

    def update_cross_coefficients(self, force_update=False):
        """Update the Lennard-Jones cross-coefficients for all atom pairs."""

        if (self.step % self.neighbor == 0) or force_update:

            N = np.sum(self.number_atoms)
            atom_types = self.atom_types
            sigma_matrix = self.sigmas
            epsilon_matrix = self.epsilons
            neighbors, neighbor_counts = self.neighbor_lists

            # Lists of lists to hold σ_ij and ε_ij for each atom
            self.cross_sigmas = np.zeros((N, N), dtype=np.float64)
            self.cross_epsilons = np.zeros((N, N), dtype=np.float64)

            for i in range(N):
                type_i = atom_types[i]
                count_i = neighbor_counts[i]  # number of valid neighbors of atom i

                for idx in range(count_i):
                    j = neighbors[i, idx]  # get neighbor index

                    type_j = atom_types[j]
                    sigma_ij = sigma_matrix[type_i-1, type_j-1]
                    epsilon_ij = epsilon_matrix[type_i-1, type_j-1]

                    self.cross_sigmas[i, j] = sigma_ij
                    self.cross_epsilons[i, j] = epsilon_ij

    def wrap_in_box(self):
        """Wrap particle positions into the simulation box."""
        # Iterate over each spatial dimension (x, y, z)
        for dim in range(3):
            # Length of the box in the current dimension
            box_length = self.box_bounds[dim][1] - self.box_bounds[dim][0]
            
            # Particles outside the upper boundary
            out_ids_upper = self.positions[:, dim] > self.box_bounds[dim][1]
            self.positions[out_ids_upper, dim] -= box_length

            # Particles outside the lower boundary
            out_ids_lower = self.positions[:, dim] < self.box_bounds[dim][0]
            self.positions[out_ids_lower, dim] += box_length
