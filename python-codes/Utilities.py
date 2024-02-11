from scipy import constants as cst
from decimal import Decimal
import numpy as np
import copy

import warnings
warnings.filterwarnings('ignore')

class Utilities:
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def perform_energy_minimization(self, displacement = 0.01):
        """Perform energy minimmization using the steepest descent method."""
        if self.minimization_steps is not None:
            for self.step in range(0, self.minimization_steps+1):
                self.update_neighbor_lists()
                Epot = self.calculate_potential_energy(self.atoms_positions)
                trial_atoms_positions = copy.deepcopy(self.atoms_positions)
                forces = self.evaluate_LJ_force()
                self.max_forces = np.max(np.abs(forces))
                trial_atoms_positions = self.atoms_positions + forces/self.max_forces*displacement
                trial_Epot = self.calculate_potential_energy(trial_atoms_positions)
                if trial_Epot<Epot: # accept new position
                    self.atoms_positions = trial_atoms_positions
                    self.wrap_in_box()
                    displacement *= 1.2
                else: # reject new position
                    displacement *= 0.2
                self.update_log(minimization = True)
                self.update_dump(filename="dump.min.lammpstrj", velocity=False, minimization = True)

    def calculate_kinetic_energy(self):
        """Calculate the kinetic energy based on the velocities of the atoms.
        $Ekin = \sum_{i=1}^Natom 1/2 m_i v_i^2$
        """
        self.Ekin = np.sum(self.atoms_mass * (self.atoms_velocities.T)**2) / 2

    def calculate_temperature(self):
        """ Follow the expression given in the LAMMPS documentation
        $Ndof = Ndim * Natom - Ndim$
        $T(t) = \sum_{i=1}^Natom \dfrac{m_i v_i^2 (t)}{k_\text{B} Ndof}$
        """
        self.calculate_kinetic_energy()
        Ndof = self.dimensions*self.total_number_atoms-self.dimensions
        self.temperature = 2*self.Ekin/Ndof

    def calculate_pressure(self):
        """Evaluate p based on the Virial equation (Eq. 4.4.2 in Frenkel-Smith 2002)"""
        Ndof = self.dimensions*self.total_number_atoms-self.dimensions    
        volume = np.prod(np.diff(self.box_boundaries))
        self.calculate_temperature()
        p_ideal = (Ndof/self.dimensions)*self.temperature/volume
        #p_non_ideal = 1/(volume*self.dimensions)*np.sum(self.evaluate_LJ_matrix()*self.evaluate_rij_matrix())
        p_non_ideal = 1/(volume*self.dimensions)*np.sum(self.evaluate_LJ_force(return_matrix=True)*self.evaluate_rij_matrix())
        self.pressure = (p_ideal+p_non_ideal)

    def evaluate_rij_matrix(self):
        """Evaluate vector rij between particles."""
        box_size = np.diff(self.box_boundaries).reshape(3) # tofix : make an internal variable ? - to be recalcuted only if berendsen
        rij_matrix = np.zeros((self.total_number_atoms,self.total_number_atoms,3))
        for Ni in range(self.total_number_atoms-1):
            position_i = self.atoms_positions[Ni]
            positions_j = self.atoms_positions
            rij_xyz = (np.remainder(position_i - positions_j + box_size/2., box_size) - box_size/2.)
            #rij = np.linalg.norm(rij_xyz, axis=0)
            rij_matrix[Ni] = rij_xyz
        return rij_matrix

    def calculate_r(self, position_i, positions_j):
        """Calculate the shortest distance between position_i and positions_j."""
        box_size = np.diff(self.box_boundaries).reshape(3)
        rij = (np.remainder(position_i - positions_j + box_size/2., box_size) - box_size/2.)
        return np.linalg.norm(rij, axis=1)

    def calculate_potential_energy(self, atoms_positions):
        """Calculate potential energy from Lennard-Jones potential."""
        energy_potential = 0
        for position_i, sigma_i, epsilon_i in zip(atoms_positions, self.atoms_sigma, self.atoms_epsilon):
            r = self.calculate_r(position_i, atoms_positions)
            sigma_j = self.atoms_sigma
            epsilon_j = self.atoms_epsilon
            sigma_ij = np.array((sigma_i+sigma_j)/2)
            epsilon_ij = np.array((epsilon_i+epsilon_j)/2)
            energy_potential_i = np.sum(4*epsilon_ij[r>0]*(np.power(sigma_ij[r>0]/r[r>0], 12)-np.power(sigma_ij[r>0]/r[r>0], 6)))
            energy_potential += energy_potential_i
        energy_potential /= 2 # To avoid counting potential energy twice
        return energy_potential
    
    def evaluate_LJ_force(self, return_matrix = False):
        if return_matrix:
            forces = np.zeros((self.total_number_atoms,self.total_number_atoms,3))
        else:
            forces = np.zeros((self.total_number_atoms,3))
        box_size = np.diff(self.box_boundaries).reshape(3)
        for Ni, position_i, sigma_i, epsilon_i, neighbor_i in zip(np.arange(self.total_number_atoms-1),
                                                    self.atoms_positions,
                                                    self.atoms_sigma,
                                                    self.atoms_epsilon,
                                                    self.neighbor_lists):
            
            N_j = np.arange(self.total_number_atoms)[neighbor_i]
            positions_j = self.atoms_positions[neighbor_i]
            sigma_j = self.atoms_sigma[neighbor_i]
            epsilon_j = self.atoms_epsilon[neighbor_i]
            rij_xyz = (np.remainder(position_i - positions_j + box_size/2., box_size) - box_size/2.).T
            rij = np.linalg.norm(rij_xyz, axis=0)
            for Nj, sigma_j0, epsilon_j0, rij0, rij_xyz0 in zip(N_j[rij < self.cut_off],
                                            sigma_j[rij < self.cut_off], epsilon_j[rij < self.cut_off],
                                            rij[rij < self.cut_off], rij_xyz.T[rij < self.cut_off]):
                sigma_ij = (sigma_i+sigma_j0)/2
                epsilon_ij = (epsilon_i+epsilon_j0)/2
                dU_dr = 48*epsilon_ij/rij0*((sigma_ij/rij0)**12-0.5*(sigma_ij/rij0)**6)
                if return_matrix:
                    forces[Ni][Nj] += dU_dr*rij_xyz0/rij0
                else:
                    forces[Ni] += dU_dr*rij_xyz0/rij0
                    forces[Nj] -= dU_dr*rij_xyz0/rij0
        return forces
    
    def wrap_in_box(self):
        for dim in np.arange(self.dimensions):
            out_ids = self.atoms_positions[:, dim] > self.box_boundaries[dim][1]
            if np.sum(out_ids) > 0:
                self.atoms_positions[:, dim][out_ids] -= np.diff(self.box_boundaries[dim])[0]
            out_ids = self.atoms_positions[:, dim] < self.box_boundaries[dim][0]
            if np.sum(out_ids) > 0:
                self.atoms_positions[:, dim][out_ids] += np.diff(self.box_boundaries[dim])[0]

    def update_neighbor_lists(self):
        if (self.step % self.neighbor == 0):
            box_size = np.diff(self.box_boundaries).reshape(3)
            neighbor_lists = []
            for Ni in range(self.total_number_atoms-1):
                a_list = []
                for Nj in np.arange(Ni+1,self.total_number_atoms):
                    if Ni != Nj:
                        position_i = self.atoms_positions[Ni]
                        position_j = self.atoms_positions[Nj]
                        rij_xyz = (np.remainder(position_i - position_j \
                                                + box_size/2., box_size) \
                                                - box_size/2.).T
                        rij = np.sqrt(np.sum(rij_xyz**2))
                        if rij < (self.cut_off+2):
                            a_list.append(Nj) 
                neighbor_lists.append(a_list.copy())
            self.neighbor_lists = neighbor_lists
