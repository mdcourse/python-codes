

import numpy as np
from InitializeSimulation import InitializeSimulation
from distance_utils import compute_vector_matrix
from force_utils import compute_force_matrix


class Measurements(InitializeSimulation):
    def __init__(self,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)
      

    def calculate_pressure(self):
        """Evaluate p based on the Virial equation (Eq. 4.4.2 in Frenkel-Smit,
        Understanding molecular simulation: from algorithms to applications, 2002)"""

        
        # Compute the ideal contribution
        Nat = np.sum(self.number_atoms) # total number of atoms
        dimension = 3 # 3D is the only possibility here
        Ndof = dimension*Nat-dimension    
        volume = np.prod(self.box_size[:3]) # box volume
        try:
            self.calculate_temperature() # this is for later on, when velocities are computed
            temperature = self.temperature
        except:
            temperature = self.desired_temperature # for MC, simply use the desired temperature
        p_ideal = Ndof*temperature/(volume*dimension)

        # Compute the non-ideal contribution
        vector_matrix = compute_vector_matrix(self.atoms_positions, self.box_size[:3])
        force_matrix = compute_force_matrix(self.neighbor_lists, self.atoms_positions, self.box_size, self.cross_coefficients)

        distances_forces = np.sum(force_matrix*vector_matrix)
        p_nonideal = distances_forces/(volume*dimension)
        # Final pressure
        self.pressure = p_ideal+p_nonideal

    def calculate_density(self):
        """Calculate the mass density."""
        # TOFIX: not used yet
        volume = np.prod(self.box_size[:3])  # Unitless
        total_mass = np.sum(self.atoms_mass)  # Unitless
        return total_mass/volume  # Unitless
