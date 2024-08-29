from dumper import update_dump_file
from logger import log_simulation_data


from scipy import constants as cst
import numpy as np
import copy
import os
from Measurements import Measurements

import warnings
warnings.filterwarnings('ignore')


class MonteCarlo(Measurements):
    def __init__(self,
                maximum_steps,
                cut_off = 9,
                displace_mc = None,
                neighbor = 1,
                desired_temperature = 300,
                thermo_outputs = "press",
                data_folder = None,
                *args,
                **kwargs):
        self.maximum_steps = maximum_steps
        self.cut_off = cut_off
        self.displace_mc = displace_mc
        self.neighbor = neighbor
        self.desired_temperature = desired_temperature
        self.thermo_outputs = thermo_outputs
        self.data_folder = data_folder
        if self.data_folder is not None:
            if os.path.exists(self.data_folder) is False:
                os.mkdir(self.data_folder)
        super().__init__(*args, **kwargs)
        self.nondimensionalize_units_3()


    def monte_carlo_move(self):
        """Monte Carlo move trial."""
        if self.displace_mc is not None: # only trigger if displace_mc was provided by the user
            try: # try using the last saved Epot, if it exists
                initial_Epot = self.Epot
            except: # If self.Epot does not exists yet, calculate it
                initial_Epot = self.compute_potential(output="potential")
            # Make a copy of the initial atoms positions
            initial_positions = copy.deepcopy(self.atoms_positions)
            # Pick an atom id randomly
            atom_id = np.random.randint(self.total_number_atoms)
            # Move the chosen atom in a random direction
            # The maximum displacement is set by self.displace_mc
            move = (np.random.random(self.dimensions)-0.5)*self.displace_mc 
            self.atoms_positions[atom_id] += move
            # Measure the optential energy of the new configuration
            trial_Epot = self.compute_potential(output="potential")
            # Evaluate whether the new configuration should be kept or not
            beta =  1/self.desired_temperature
            delta_E = trial_Epot-initial_Epot
            random_number = np.random.random() # random number between 0 and 1
            acceptation_probability = np.min([1, np.exp(-beta*delta_E)])
            if random_number <= acceptation_probability: # Accept new position
                self.Epot = trial_Epot
            else: # Reject new position
                self.Epot = initial_Epot
                self.atoms_positions = initial_positions # Revert to initial positions

    def nondimensionalize_units_3(self):
        """Use LJ prefactors to convert units into non-dimensional."""
        self.cut_off = self.cut_off/self.reference_distance
        self.desired_temperature = self.desired_temperature \
            /self.reference_temperature
        if self.displace_mc is not None:
            self.displace_mc /= self.reference_distance

    def run(self):
        """Perform the loop over time."""
        for self.step in range(0, self.maximum_steps+1):
            self.update_neighbor_lists()
            self.update_cross_coefficients()
            self.monte_carlo_move()
            self.wrap_in_box()
            log_simulation_data(self)
            update_dump_file(self, "dump.mc.lammpstrj")
