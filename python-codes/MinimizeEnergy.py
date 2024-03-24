import numpy as np
import copy

import warnings
warnings.filterwarnings('ignore')

from InitializeSimulation import InitializeSimulation
from Utilities import Utilities
from Outputs import Outputs

class MinimizeEnergy(InitializeSimulation, Utilities, Outputs):
    def __init__(self,
        minimization_steps=500,
        *args,
        **kwargs,
        ):

        self.neighbor = 1
        self.cut_off = 12

        self.minimization_steps = minimization_steps
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
                self.update_dump(filename="dump.min.lammpstrj",
                                 velocity=False, minimization = True)