import numpy as np
from MDAnalysis.analysis import distances


from potentials import potentials


class Utilities:
    def __init__(self,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)


    def update_neighbor_lists(self, force_update=False):
        if (self.step % self.neighbor == 0) | force_update:
            matrix = distances.contact_matrix(self.atoms_positions,
                cutoff=self.cut_off, #+2,
                returntype="numpy",
                box=self.box_size)
            neighbor_lists = []
            for cpt, array in enumerate(matrix[:-1]):
                list = np.where(array)[0].tolist()
                list = [ele for ele in list if ele > cpt]
                neighbor_lists.append(list)
            self.neighbor_lists = neighbor_lists

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
            rij = self.compute_distance(self.atoms_positions[Ni],
                                        self.atoms_positions[neighbor_of_i],
                                        self.box_size)
            # Measure potential using pre-calculated cross coefficients
            sigma_ij = self.sigma_ij_list[Ni]
            epsilon_ij = self.epsilon_ij_list[Ni]
            energy_potential += np.sum(potentials(epsilon_ij, sigma_ij, rij))
        return energy_potential

    def compute_distance(self,position_i, positions_j, box_size, only_norm = True):
        """
        Measure the distances between two particles.
        # TOFIX: Move as a function instead of a method?
        """
        rij_xyz = np.nan_to_num(np.remainder(position_i - positions_j
                  + box_size[:3]/2.0, box_size[:3]) - box_size[:3]/2.0)
        if only_norm:
            return np.linalg.norm(rij_xyz, axis=1)
        else:
            return np.linalg.norm(rij_xyz, axis=1), rij_xyz

    def compute_force(self, return_vector = True):
        if return_vector: # return a N-size vector
            force_vector = np.zeros((np.sum(self.number_atoms),3))
        else: # return a N x N matrix
            force_matrix = np.zeros((np.sum(self.number_atoms),
                                    np.sum(self.number_atoms),3))
        for Ni in np.arange(np.sum(self.number_atoms)-1):
            # Read neighbor list
            neighbor_of_i = self.neighbor_lists[Ni]
            # Measure distance
            rij, rij_xyz = self.compute_distance(self.atoms_positions[Ni],
                                        self.atoms_positions[neighbor_of_i],
                                        self.box_size, only_norm = False)
            # Measure force using information about cross coefficients
            sigma_ij = self.sigma_ij_list[Ni]
            epsilon_ij = self.epsilon_ij_list[Ni]       
            fij_xyz = potentials(epsilon_ij, sigma_ij, rij, derivative = True)
            if return_vector:
                # Add the contribution to both Ni and its neighbors
                force_vector[Ni] += np.sum((fij_xyz*rij_xyz.T/rij).T, axis=0)
                force_vector[neighbor_of_i] -= (fij_xyz*rij_xyz.T/rij).T 
            else:
                # Add the contribution to the matrix
                force_matrix[Ni][neighbor_of_i] += (fij_xyz*rij_xyz.T/rij).T
        if return_vector:
            return force_vector
        else:
            return force_matrix

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

    def evaluate_rij_matrix(self):
        """Matrix of vectors between all particles."""
        Nat = np.sum(self.number_atoms)
        Box = self.box_size[:3]
        rij_matrix = np.zeros((Nat, Nat,3))
        pos_j = self.atoms_positions
        for Ni in range(Nat-1):
            pos_i = self.atoms_positions[Ni]
            rij_xyz = (np.remainder(pos_i - pos_j + Box/2.0, Box) - Box/2.0)
            rij_matrix[Ni] = rij_xyz
        return rij_matrix
