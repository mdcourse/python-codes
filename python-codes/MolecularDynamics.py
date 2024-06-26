import numpy as np

import warnings
warnings.filterwarnings('ignore')

from InitializeSimulation import InitializeSimulation
from Measurements import Measurements

class MolecularDynamics(InitializeSimulation, Measurements): #to fix should be Outputs
    def __init__(self,
                maximum_steps,
                tau_temp = None,
                tau_press = None,
                minimization_steps = None,
                cut_off = 10,
                time_step=1,
                neighbor=10,
                desired_temperature = 300,
                desired_pressure=1,
                *args,
                **kwargs,
                ):
        self.maximum_steps = maximum_steps
        self.tau_temp = tau_temp
        self.tau_press = tau_press
        self.minimization_steps = minimization_steps
        self.cut_off = cut_off
        self.time_step = time_step
        self.neighbor = neighbor
        self.desired_temperature=desired_temperature
        self.desired_pressure=desired_pressure
        super().__init__(*args, **kwargs)

        self.nondimensionalize_units_4()
        
    def nondimensionalize_units_4(self):
        """Use LJ prefactors to convert units into non-dimensional."""
        self.cut_off /= self.reference_distance
        self.time_step /= self.reference_time
        self.desired_temperature /= self.reference_temperature
        self.desired_pressure /= self.reference_pressure
        if self.tau_temp is not None:
            self.tau_temp /= self.reference_time
        if self.tau_press is not None:
            self.tau_press /= self.reference_time
    
    def run(self):
        self.write_topology_file(filename="initial.data",
                                 velocity=False)
        self.set_initial_velocity()
        for self.step in range(0, self.maximum_steps+1):
            self.update_neighbor_lists()
            self.integrate_equation_of_motion()
            self.wrap_in_box()
            self.apply_berendsen_thermostat()
            self.apply_berendsen_barostat()
            self.update_log()
            self.update_dump(filename = "dump.md.lammpstrj")
        #self.write_lammps_data(filename = "final.data")
        #self.write_lammps_parameters()
        #self.write_lammps_variables()

    def integrate_equation_of_motion(self):
        """Integrate equation of motion using half-step velocity"""
        if self.step == 0:
            self.atoms_accelerations = (self.calculate_LJ_potential_force(output = "force-vector").T/self.atoms_mass).T
        atoms_velocity_Dt2 = self.atoms_velocities + self.atoms_accelerations*self.time_step/2
        self.atoms_positions = self.atoms_positions + atoms_velocity_Dt2*self.time_step
        self.atoms_accelerations = (self.calculate_LJ_potential_force(output = "force-vector").T/self.atoms_mass).T
        self.atoms_velocities = atoms_velocity_Dt2 + self.atoms_accelerations*self.time_step/2

    def apply_berendsen_thermostat(self):
        """Rescale velocities based on Berendsen thermostat."""
        if self.tau_temp is not None:
            self.calculate_temperature()
            scale = np.sqrt(1+self.time_step*((self.desired_temperature/self.temperature)-1)/self.tau_temp)
            self.atoms_velocities *= scale

    def apply_berendsen_barostat(self):
        """Rescale box size based on Berendsten barostat."""
        if self.tau_press is not None:
            self.calculate_pressure()
            scale = np.sqrt(1+self.time_step*((self.pressure/self.desired_pressure)-1)/self.tau_press)
            self.box_boundaries *= scale
            self.box_size = np.diff(self.box_boundaries).reshape(3)
            self.atoms_positions *= scale
            