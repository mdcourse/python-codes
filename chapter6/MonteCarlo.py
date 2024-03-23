from scipy import constants as cst
from decimal import Decimal
import numpy as np
import copy

import warnings
warnings.filterwarnings('ignore')

from InitializeSimulation import InitializeSimulation
from Utilities import Utilities
from Outputs import Outputs

class MonteCarlo(InitializeSimulation, Utilities, Outputs):
    def __init__(self,
        maximum_steps,
        displace_mc=None,
        *args,
        **kwargs,
        ):
        super().__init__(*args, **kwargs)

        self.maximum_steps = maximum_steps
        self.displace_mc = displace_mc

        if self.displace_mc is not None:
            self.displace_mc /= self.reference_distance

    def run(self):
        for self.step in range(0, self.maximum_steps+1):
            self.monte_carlo_displacement()
            self.wrap_in_box()
            self.update_dump(filename="dump.mc.lammpstrj", velocity=False)

    def monte_carlo_displacement(self):
        if self.displace_mc is not None:
            beta =  1/self.desired_temperature
            Epot = self.calculate_potential_energy(self.atoms_positions)
            trial_atoms_positions = copy.deepcopy(self.atoms_positions)
            atom_id = np.random.randint(self.total_number_atoms)
            trial_atoms_positions[atom_id] += (np.random.random(3)-0.5)*self.displace_mc
            trial_Epot = self.calculate_potential_energy(trial_atoms_positions)
            acceptation_probability = np.min([1, np.exp(-beta*(trial_Epot-Epot))])
            if np.random.random() <= acceptation_probability:
                self.atoms_positions = trial_atoms_positions
                self.Epot = trial_Epot
            else:
                self.Epot = Epot 