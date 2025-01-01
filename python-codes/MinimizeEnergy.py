from dumper import update_dump_file
from logger import log_simulation_data
from force_utils import compute_force_vector
from pot_utils import compute_potential

from Measurements import Measurements
import numpy as np
import copy


class MinimizeEnergy(Measurements):
    def __init__(self,
                maximum_steps,
                *args,
                **kwargs):
        self.maximum_steps = maximum_steps
        super().__init__(*args, **kwargs)


    def run(self):
        self.displacement = 0.01 # pick a random initial displacement (dimentionless)
        # *step* loops for 0 to *maximum_steps*+1
        for self.step in range(0, self.maximum_steps+1):
            # First, evaluate the initial energy and max force
            self.update_neighbor_lists() # Rebuild neighbor list, if necessary
            self.update_cross_coefficients() # Recalculate the cross coefficients, if necessary
            # Compute Epot/MaxF/force
            if hasattr(self, 'Epot') is False: # If self.Epot does not exist yet, calculate it
                self.Epot = compute_potential(self.neighbor_lists, self.atoms_positions, self.box_mda, self.cross_coefficients)
            if hasattr(self, 'MaxF') is False: # If self.MaxF does not exist yet, calculate it
                forces = compute_force_vector(self.neighbor_lists, self.atoms_positions, self.box_mda, self.cross_coefficients)
                self.MaxF = np.max(np.abs(forces))
            init_Epot = self.Epot
            init_MaxF = self.MaxF
            # Save the current atom positions
            init_positions = copy.deepcopy(self.atoms_positions)
            # Move the atoms in the opposite direction of the maximum force
            self.atoms_positions = self.atoms_positions \
                + forces/init_MaxF*self.displacement
            # Recalculate the energy
            trial_Epot = compute_potential(self.neighbor_lists, self.atoms_positions, self.box_mda, self.cross_coefficients)
            # Keep the more favorable energy
            if trial_Epot < init_Epot: # accept new position
                self.Epot = trial_Epot
                forces = compute_force_vector(self.neighbor_lists, self.atoms_positions, self.box_mda, self.cross_coefficients) # calculate the new max force and save it
                self.MaxF = np.max(np.abs(forces))
                self.wrap_in_box()  # Wrap atoms in the box
                #self.atoms_positions = wrap_in_box(self.atoms_positions,
                #                                   self.box_boundaries)
                self.displacement *= 1.2 # Multiply the displacement by a factor 1.2
            else: # reject new position
                self.Epot = init_Epot # Revert to old energy
                self.atoms_positions = init_positions # Revert to old positions
                self.displacement *= 0.2 # Multiply the displacement by a factor 0.2
            log_simulation_data(self)
            update_dump_file(self, "dump.min.lammpstrj")
