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

    def calculate_r(self, position_i, positions_j):
        """Calculate the shortest distance between position_i and positions_j."""
        rij = (np.remainder(position_i - positions_j
                            + self.box_size/2., self.box_size) - self.box_size/2.)
        return np.linalg.norm(rij, axis=1)

    def calculate_potential_energy(self, atoms_positions):
        """Calculate potential energy from Lennard-Jones potential."""
        energy_potential = 0
        for position_i, sigma_i, epsilon_i in zip(atoms_positions,
                                                  self.atoms_sigma,
                                                  self.atoms_epsilon):
            r = self.calculate_r(position_i, atoms_positions)
            sigma_j = self.atoms_sigma
            epsilon_j = self.atoms_epsilon
            sigma_ij = np.array((sigma_i+sigma_j)/2)
            epsilon_ij = np.array((epsilon_i+epsilon_j)/2)
            energy_potential_i = np.sum(4*epsilon_ij[r>0]*(np.power(sigma_ij[r>0]/r[r>0], 12)-np.power(sigma_ij[r>0]/r[r>0], 6)))
            energy_potential += energy_potential_i
        return energy_potential/2

    def wrap_in_box(self):
        """Re-wrap the atoms that are outside the box."""
        for dim in np.arange(self.dimensions):
            out_ids = self.atoms_positions[:, dim] > self.box_boundaries[dim][1]
            #if np.sum(out_ids) > 0: # tofix : necesary ?
            self.atoms_positions[:, dim][out_ids] -= np.diff(self.box_boundaries[dim])[0]
            out_ids = self.atoms_positions[:, dim] < self.box_boundaries[dim][0]
            #if np.sum(out_ids) > 0:
            self.atoms_positions[:, dim][out_ids] += np.diff(self.box_boundaries[dim])[0]