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

    def calculate_kinetic_energy(self):
        self.Ekin = np.sum(self.atom_mass*np.sum(self.atoms_velocities**2, axis=1)/2)

    def calculate_temperature(self):
        """Follow the expression given in the LAMMPS documentation"""
        self.calculate_kinetic_energy()
        Ndof = self.dimensions*self.total_number_atoms-self.dimensions
        self.temperature = 2*self.Ekin/Ndof

    def calculate_pressure(self):
        """Evaluate p based on the Virial equation (Eq. 4.4.2 in Frenkel-Smith 2002)"""
        Ndof = self.dimensions*self.total_number_atoms-self.dimensions    
        volume = np.prod(np.diff(self.box_boundaries))
        self.calculate_temperature()
        p_ideal = (Ndof/self.dimensions)*self.temperature/volume
        p_non_ideal = 1/(volume*self.dimensions)*np.sum(self.evaluate_LJ_matrix()*self.evaluate_rij_matrix())
        self.pressure = (p_ideal+p_non_ideal)

    def evaluate_rij_matrix(self):
        """Evaluate vector rij between particles."""
        rij = np.zeros((self.total_number_atoms,self.total_number_atoms,3))
        for Ni in range(self.total_number_atoms-1):
            position_i = self.atoms_positions[Ni]
            for Nj in np.arange(Ni+1,self.total_number_atoms):
                position_j = self.atoms_positions[Nj]
                box_size = np.diff(self.box_boundaries).reshape(3)
                rij_xyz = (np.remainder(position_i - position_j + box_size/2., box_size) - box_size/2.).T
                r = np.sqrt(np.sum(rij_xyz**2))
                if r < self.cut_off:
                    rij[Ni][Nj] = rij_xyz
        return rij

    def calculate_r(self, position_i, positions_j, number_atoms = None):
        """Calculate the shortest distance between position_i and positions_j."""
        if number_atoms is None:
            rij2 = np.zeros(self.total_number_atoms)
        else:
            rij2 = np.zeros(number_atoms)
        box_size = np.diff(self.box_boundaries).reshape(3)
        rij = (np.remainder(position_i - positions_j + box_size/2., box_size) - box_size/2.).T
        for dim in np.arange(self.dimensions):
            rij2 += np.power(rij[dim, :], 2)
        return np.sqrt(rij2)

    def calculate_potential_energy(self, atoms_positions, number_atoms = None):
        """Calculate potential energy assuming Lennard-Jones potential interaction."""
        energy_potential = 0
        for position_i in atoms_positions:
            r = self.calculate_r(position_i, atoms_positions, number_atoms)
            energy_potential_i = np.sum(4*(1/np.power(r[r>0], 12)-1/np.power(r[r>0], 6)))
            energy_potential += energy_potential_i
        energy_potential /= 2 # Avoid counting potential energy twice
        return energy_potential

    def evaluate_LJ_force(self):
        """Evaluate force based on LJ potential derivative."""
        forces = np.zeros((self.total_number_atoms,3))
        for Ni in range(self.total_number_atoms-1):
            position_i = self.atoms_positions[Ni]
            for Nj in np.arange(Ni+1,self.total_number_atoms):
                position_j = self.atoms_positions[Nj]
                box_size = np.diff(self.box_boundaries).reshape(3)
                rij_xyz = (np.remainder(position_i - position_j + box_size/2., box_size) - box_size/2.).T
                rij = np.sqrt(np.sum(rij_xyz**2))
                if rij < self.cut_off:
                    dU_dr = 48/rij*(1/rij**12-0.5/rij**6)
                    forces[Ni] += dU_dr*rij_xyz/rij
                    forces[Nj] -= dU_dr*rij_xyz/rij
        return forces
    
    def evaluate_LJ_matrix(self):
        """Evaluate force based on LJ potential derivative."""
        forces = np.zeros((self.total_number_atoms,self.total_number_atoms,3))
        for Ni in range(self.total_number_atoms-1):
            position_i = self.atoms_positions[Ni]
            for Nj in np.arange(Ni+1,self.total_number_atoms):
                position_j = self.atoms_positions[Nj]
                box_size = np.diff(self.box_boundaries).reshape(3)
                rij_xyz = (np.remainder(position_i - position_j + box_size/2., box_size) - box_size/2.).T
                rij = np.sqrt(np.sum(rij_xyz**2))
                if rij < self.cut_off:
                    dU_dr = 48/rij*(1/rij**12-0.5/rij**6)
                    forces[Ni][Nj] += dU_dr*rij_xyz/rij
        return forces
    
    def wrap_in_box(self):
        for dim in np.arange(self.dimensions):
            out_ids = self.atoms_positions[:, dim] > self.box_boundaries[dim][1]
            if np.sum(out_ids) > 0:
                self.atoms_positions[:, dim][out_ids] -= np.diff(self.box_boundaries[dim])[0]
            out_ids = self.atoms_positions[:, dim] < self.box_boundaries[dim][0]
            if np.sum(out_ids) > 0:
                self.atoms_positions[:, dim][out_ids] += np.diff(self.box_boundaries[dim])[0]
