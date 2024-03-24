import numpy as np
import copy

import warnings
warnings.filterwarnings('ignore')

from InitializeSimulation import InitializeSimulation
from Utilities import Utilities
from Outputs import Outputs

class MinimizeEnergy(InitializeSimulation):
    def __init__(self,
        minimization_steps=500,
        cut_off = 9,
        neighbor = 1,
        displacement = 0.01,
        *args,
        **kwargs,
        ):

        self.neighbor = neighbor
        self.cut_off = cut_off
        self.displacement = displacement

        self.minimization_steps = minimization_steps
        super().__init__(*args, **kwargs)

    def perform_energy_minimization(self):
        """Perform energy minimmization using the steepest descent method."""
        if self.minimization_steps is not None:
            for self.step in range(0, self.minimization_steps+1):
                # Measure the initial energy and max force
                self.update_neighbor_lists()
                initial_Epot = self.calculate_LJ_potential_force(output="potential")
                initial_positions = copy.deepcopy(self.atoms_positions)
                forces = self.calculate_LJ_potential_force(output="force-vector")
                max_forces = np.max(np.abs(forces))
                # Test a new sets of positions
                self.atoms_positions = self.atoms_positions + forces/max_forces*self.displacement
                trial_Epot = self.calculate_LJ_potential_force(output="potential")
                # Keep the more favorable energy
                if trial_Epot<initial_Epot: # accept new position
                    Epot = trial_Epot
                    self.wrap_in_box()
                    self.displacement *= 1.2
                else: # reject new position
                    Epot = initial_Epot
                    self.atoms_positions = initial_positions
                    self.displacement *= 0.2
                self.log_minimize(Epot, max_forces)
                self.update_dump(filename="dump.min.lammpstrj",velocity=False)