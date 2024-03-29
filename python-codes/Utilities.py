from scipy import constants as cst
import numpy as np

import MDAnalysis as mda
from MDAnalysis.analysis import distances
from Potentials import LJ_potential

import warnings
warnings.filterwarnings('ignore')

class Utilities:
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_kinetic_energy(self):
        """Calculate the kinetic energy based on the velocities of the atoms.
        $Ekin = \sum_{i=1}^Natom 1/2 m_i v_i^2$
        """
        self.Ekin = np.sum(self.atoms_mass*(self.atoms_velocities.T)**2)/2 # tofix atoms of different masses

    def calculate_temperature(self):
        """ Follow the expression given in the LAMMPS documentation
        $Ndof = Ndim * Natom - Ndim$
        $T(t) = \sum_{i=1}^Natom \dfrac{m_i v_i^2 (t)}{k_\text{B} Ndof}$
        """
        self.calculate_kinetic_energy()
        Ndof = self.dimensions*self.total_number_atoms-self.dimensions
        self.temperature = 2*self.Ekin/Ndof

    def calculate_density(self):
        """Calculate the mass density."""
        volume = np.prod(self.box_size)
        total_mass = np.sum(self.atoms_mass)
        return total_mass/volume

    def calculate_pressure(self):
        """Evaluate p based on the Virial equation (Eq. 4.4.2 in Frenkel-Smith 2002)"""
        Ndof = self.dimensions*self.total_number_atoms-self.dimensions    
        volume = np.prod(self.box_size)
        self.calculate_temperature()
        p_ideal = (Ndof/self.dimensions)*self.temperature/volume
        p_non_ideal = 1/(volume*self.dimensions)*np.sum(self.calculate_LJ_potential_force(output="force-matrix")*self.evaluate_rij_matrix())
        self.pressure = (p_ideal+p_non_ideal)

    def calculate_rdf(self):
        r = np.linalg.norm(self.evaluate_rij_matrix(), axis=2)


    def evaluate_rij_matrix(self):
        """Matrix of vectors between all particles."""
        rij_matrix = np.zeros((self.total_number_atoms,self.total_number_atoms,3))
        for Ni in range(self.total_number_atoms-1):
            position_i = self.atoms_positions[Ni]
            positions_j = self.atoms_positions
            rij_xyz = (np.remainder(position_i - positions_j
                                    + self.box_size[:3]/2., self.box_size[:3])
                                    - self.box_size[:3]/2.)
            rij_matrix[Ni] = rij_xyz
        return rij_matrix

    def calculate_r(self, position_i, positions_j):
        """Calculate the shortest distance between position_i and positions_j.
        # to fix : use the MDAnalysis option
        """
        rij = (np.remainder(position_i - positions_j
                            + self.box_size[:3]/2., self.box_size[:3])
                            - self.box_size[:3]/2.)
        return np.linalg.norm(rij, axis=1)

    def calculate_LJ_potential_force(self, output):
        """Calculate potential and force from Lennard-Jones potential."""
        # to fix : wont work for insert/delete ... 
        if output == "force-vector":
            forces = np.zeros((self.total_number_atoms,3))
        elif output == "force-matrix":
            forces = np.zeros((self.total_number_atoms,self.total_number_atoms,3))
        energy_potential = 0
        for Ni in np.arange(self.total_number_atoms-1):
            # Read information about atom i
            position_i = self.atoms_positions[Ni]
            sigma_i = self.atoms_sigma[Ni]
            epsilon_i = self.atoms_epsilon[Ni]
            neighbor_of_i = self.neighbor_lists[Ni]
            # Read information about neighbors j
            positions_j = self.atoms_positions[neighbor_of_i]
            sigma_j = self.atoms_sigma[neighbor_of_i]
            epsilon_j = self.atoms_epsilon[neighbor_of_i]
            # Measure distances and other cross parameters
            rij_xyz = (np.remainder(position_i - positions_j
                                    + self.box_size[:3]/2., self.box_size[:3])
                                    - self.box_size[:3]/2.)
            rij = np.linalg.norm(rij_xyz, axis=1)
            sigma_ij = (sigma_i+sigma_j)/2
            epsilon_ij = (epsilon_i+epsilon_j)/2
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
        """Re-wrap the atoms that are outside the box."""
        for dim in np.arange(self.dimensions):
            out_ids = self.atoms_positions[:, dim] > self.box_boundaries[dim][1]
            self.atoms_positions[:, dim][out_ids] -= np.diff(self.box_boundaries[dim])[0]
            out_ids = self.atoms_positions[:, dim] < self.box_boundaries[dim][0]
            self.atoms_positions[:, dim][out_ids] += np.diff(self.box_boundaries[dim])[0]

    def update_neighbor_lists(self):
        """Update the neighbor lists."""
        if (self.step % self.neighbor == 0):

            matrix = distances.contact_matrix(self.atoms_positions,
                cutoff=self.cut_off, #+2,
                returntype="numpy",
                box=self.box_size)

            cpt = 0
            neighbor_lists = []
            for array in matrix[:-1]:
                list = np.where(array)[0].tolist()
                list = [ele for ele in list if ele > cpt]
                cpt += 1
                neighbor_lists.append(list)

            self.neighbor_lists = neighbor_lists

    def set_initial_velocity(self):
        """Give velocity to atoms so that the initial temperature is the desired one."""
        self.atoms_velocities = np.random.normal(size=(self.total_number_atoms, self.dimensions))
        self.calculate_temperature()
        scale = np.sqrt(1+((self.desired_temperature/self.temperature)-1))
        self.atoms_velocities *= scale
