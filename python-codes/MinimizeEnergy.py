from dumper import update_dump_file
from logger import log_simulation_data
import numpy as np
import copy


from Measurements import Measurements
import os


class MinimizeEnergy(Measurements):
    def __init__(self,
                maximum_steps,
                cut_off=9,
                neighbor=1,
                displacement=0.01,
                thermo_outputs="MaxF",
                data_folder=None,
                *args,
                **kwargs):
        self.neighbor = neighbor
        self.cut_off = cut_off
        self.displacement = displacement
        self.maximum_steps = maximum_steps
        self.thermo_outputs = thermo_outputs
        self.data_folder = data_folder
        if self.data_folder is not None:
            if os.path.exists(self.data_folder) is False:
                os.mkdir(self.data_folder)
        super().__init__(*args, **kwargs)
        self.nondimensionalize_units_2()


    def nondimensionalize_units_2(self):
        """Use LJ prefactors to convert units into non-dimensional."""
        self.cut_off = self.cut_off/self.reference_distance
        self.displacement = self.displacement/self.reference_distance

    def run(self):
        # *step* loops for 0 to *maximum_steps*+1
        for self.step in range(0, self.maximum_steps+1):
            # First, meevaluate the initial energy and max force
            self.update_neighbor_lists() # Rebuild neighbor list, if necessary
            self.update_cross_coefficients() # Recalculate the cross coefficients, if necessary
            try: # try using the last saved Epot and MaxF, if it exists
                init_Epot = self.Epot
                init_MaxF = self.MaxF
            except: # If Epot/MaxF do not exists yet, calculate them both
                init_Epot = self.compute_potential(output="potential")
                forces = self.compute_potential(output="force-vector")
                init_MaxF = np.max(np.abs(forces))
            # Save the current atom positions
            init_positions = copy.deepcopy(self.atoms_positions)
            # Move the atoms in the opposite direction of the maximum force
            self.atoms_positions = self.atoms_positions \
                + forces/init_MaxF*self.displacement
            # Recalculate the energy
            trial_Epot = self.compute_potential(output="potential")
            # Keep the more favorable energy
            if trial_Epot < init_Epot: # accept new position
                self.Epot = trial_Epot
                # calculate the new max force and save it
                forces = self.compute_potential(output="force-vector")
                self.MaxF = np.max(np.abs(forces))
                self.wrap_in_box()  # Wrap atoms in the box, if necessary
                self.displacement *= 1.2 # Multiply the displacement by a factor 1.2
            else: # reject new position
                self.Epot = init_Epot # Revert to old energy
                self.atoms_positions = init_positions # Revert to old positions
                self.displacement *= 0.2 # Multiply the displacement by a factor 0.2
            log_simulation_data(self)
            update_dump_file(self, "dump.min.lammpstrj")
