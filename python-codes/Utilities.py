import numpy as np

from utils_contact import contact_matrix, compute_neighbor_lists


class Utilities:
    def __init__(self,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)

    def update_neighbor_lists(self, force_update=False):
        """Update the neighbor lists based on contact analysis."""
        if (self.step % self.neighbor == 0) or force_update:  # Check if an update is needed
            # Compute the contact matrix based on current particle positions
            matrix = contact_matrix(self.atoms_positions,
                                    cutoff=self.cut_off,
                                    box=self.box_mda)

            # Compute the neighbor lists from the contact matrix
            self.neighbor_lists = compute_neighbor_lists(matrix)

    def update_cross_coefficients(self, force_update=False):
        """Update the Lennard-Jones cross-coefficients for all atom pairs."""
        # Check if an update is necessary
        if (self.step % self.neighbor == 0) or force_update:
                # Initialize lists to store cross-coefficients
                sigma_ijs = []  # To store sigma_ij values
                epsilon_ijs = []  # To store epsilon_ij values
                
                # Iterate over all atoms (excluding the last atom)
                for atom_i in range(np.sum(self.number_atoms) - 1):
                    # Get the properties of the current atom (atom_i)
                    sigma_i = self.atoms_sigma[atom_i]
                    epsilon_i = self.atoms_epsilon[atom_i]

                    # Get the list of neighbors for atom_i
                    neighbor_of_i = self.neighbor_lists[atom_i]

                    # Calculate cross-parameters with each neighbor
                    sigma_j = self.atoms_sigma[neighbor_of_i]  # Neighbor sigma values
                    epsilon_j = self.atoms_epsilon[neighbor_of_i]  # Neighbor epsilon values

                    # Calculate cross-coefficients for each neighbor pair
                    for sigma_j_val, epsilon_j_val in zip(sigma_j, epsilon_j):
                        sigma_ij = (sigma_i + sigma_j_val) / 2  # Average sigma
                        epsilon_ij = (epsilon_i + epsilon_j_val) / 2  # Average epsilon
                        
                        # Append the calculated cross-coefficients
                        sigma_ijs.append(sigma_ij)
                        epsilon_ijs.append(epsilon_ij)
                
                # Store the cross-coefficients as a 2D NumPy array
                self.cross_coefficients = np.array([sigma_ijs, epsilon_ijs])

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
