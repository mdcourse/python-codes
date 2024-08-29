import numpy as np
from MDAnalysis.analysis import distances


from potentials import LJ_potential


class Utilities:
    def __init__(self,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)


    def update_neighbor_lists(self):
        if (self.step % self.neighbor == 0):
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

    def update_cross_coefficients(self):
        if (self.step % self.neighbor == 0):
            # Precalculte LJ cross-coefficients
            sigma_ij_list = []
            epsilon_ij_list = []
            for Ni in np.arange(self.total_number_atoms-1): # tofix error for GCMC
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

    def compute_potential(self, output):
        if output == "force-vector":
            forces = np.zeros((self.total_number_atoms,3))
        elif output == "force-matrix":
            forces = np.zeros((self.total_number_atoms,self.total_number_atoms,3))
        energy_potential = 0
        box_size = self.box_size[:3]
        half_box_size = self.box_size[:3]/2.0
        for Ni in np.arange(self.total_number_atoms-1):
            # Read information about atom i
            position_i = self.atoms_positions[Ni]
            neighbor_of_i = self.neighbor_lists[Ni]
            # Read information about neighbors j and cross coefficient
            positions_j = self.atoms_positions[neighbor_of_i]
            sigma_ij = self.sigma_ij_list[Ni]
            epsilon_ij = self.epsilon_ij_list[Ni]
            # Measure distances
            rij_xyz = (np.remainder(position_i - positions_j + half_box_size, box_size) - half_box_size)
            rij = np.linalg.norm(rij_xyz, axis=1)
            # Measure potential
            if output == "potential":
                energy_potential += np.sum(LJ_potential(epsilon_ij, sigma_ij, rij))
            else:
                derivative_potential = LJ_potential(epsilon_ij, sigma_ij, rij, derivative = True)
                if output == "force-vector":
                    forces[Ni] += np.sum((derivative_potential*rij_xyz.T/rij).T, axis=0)
                    forces[neighbor_of_i] -= (derivative_potential*rij_xyz.T/rij).T 
                elif output == "force-matrix":
                    forces[Ni][neighbor_of_i] += (derivative_potential*rij_xyz.T/rij).T
        if output=="potential":
            return energy_potential
        elif (output == "force-vector") | (output == "force-matrix"):
            return forces

    def wrap_in_box(self):
        for dim in np.arange(self.dimensions):
            out_ids = self.atoms_positions[:, dim] \
                > self.box_boundaries[dim][1]
            self.atoms_positions[:, dim][out_ids] \
                -= np.diff(self.box_boundaries[dim])[0]
            out_ids = self.atoms_positions[:, dim] \
                < self.box_boundaries[dim][0]
            self.atoms_positions[:, dim][out_ids] \
                += np.diff(self.box_boundaries[dim])[0]

    def calculate_density(self):
        """Calculate the mass density."""
        volume = np.prod(self.box_size[:3])  # Unitless
        total_mass = np.sum(self.atoms_mass)  # Unitless
        return total_mass/volume  # Unitless

    def calculate_pressure(self):
        """Evaluate p based on the Virial equation (Eq. 4.4.2 in Frenkel-Smit,
        Understanding molecular simulation: from algorithms to applications, 2002)"""
        # Ideal contribution
        Ndof = self.dimensions*self.total_number_atoms-self.dimensions    
        volume = np.prod(self.box_size[:3])
        try:
            self.calculate_temperature() # this is for later on, when velocities are computed
            temperature = self.temperature
        except:
            temperature = self.desired_temperature # for MC, simply use the desired temperature
        p_ideal = Ndof*temperature/(volume*self.dimensions)
        # Non-ideal contribution
        distances_forces = np.sum(self.compute_potential(output="force-matrix")*self.evaluate_rij_matrix())
        p_nonideal = distances_forces/(volume*self.dimensions)
        # Final pressure
        self.pressure = p_ideal+p_nonideal

    def evaluate_rij_matrix(self):
        """Matrix of vectors between all particles."""
        box_size = self.box_size[:3]
        half_box_size = self.box_size[:3]/2.0
        rij_matrix = np.zeros((self.total_number_atoms,self.total_number_atoms,3))
        positions_j = self.atoms_positions
        for Ni in range(self.total_number_atoms-1):
            position_i = self.atoms_positions[Ni]
            rij_xyz = (np.remainder(position_i - positions_j + half_box_size, box_size) - half_box_size)
            rij_matrix[Ni] = rij_xyz
        return rij_matrix
