import numpy as np
import copy
from Outputs import Outputs

import warnings
warnings.filterwarnings('ignore')


class MinimizeEnergy(Outputs):
    def __init__(self,
                 maximum_steps,
                 cut_off=9,
                 neighbor=1,
                 displacement=0.01,
                 *args,
                 **kwargs):
        self.neighbor = neighbor
        self.cut_off = cut_off
        self.displacement = displacement
        self.maximum_steps = maximum_steps
        super().__init__(*args, **kwargs)
        self.nondimensionalize_units_2()

    def nondimensionalize_units_2(self):
        """Use LJ prefactors to convert units into non-dimensional."""
        self.cut_off = self.cut_off/self.reference_distance
        self.displacement = self.displacement/self.reference_distance

    def run(self):
        """Perform energy minimmization using the steepest descent method."""
        for self.step in range(0, self.maximum_steps+1):
            # Measure the initial energy and max force
            self.update_neighbor_lists()
            init_Epot = self.compute_potential(output="potential")
            initial_positions = copy.deepcopy(self.atoms_positions)
            forces = self.compute_potential(output="force-vector")
            max_forces = np.max(np.abs(forces))
            # Test a new sets of positions
            self.atoms_positions = self.atoms_positions \
                + forces/max_forces*self.displacement
            trial_Epot = self.compute_potential(output="potential")
            # Keep the more favorable energy
            if trial_Epot < init_Epot:  # accept new position
                Epot = trial_Epot
                self.wrap_in_box()
                self.displacement *= 1.2
            else:  # reject new position
                Epot = init_Epot
                self.atoms_positions = initial_positions
                self.displacement *= 0.2
            self.update_log_minimize(Epot, max_forces)
            self.update_dump_file(filename="dump.min.lammpstrj")
