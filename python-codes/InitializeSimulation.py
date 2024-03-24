import numpy as np

import warnings
warnings.filterwarnings('ignore')

from Utilities import Utilities
from Outputs import Outputs

class InitializeSimulation(Utilities, Outputs):
    def __init__(self,
                 number_atoms=[10], # List
                 box_dimensions=[10, 10, 10], # List
                 epsilon=[0.1], # List
                 sigma=[1], # List
                 atom_mass=[1], # List
                 seed=None, # Int
                 initial_positions=None,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.number_atoms = number_atoms
        self.box_dimensions = box_dimensions
        self.dimensions = len(box_dimensions)
        self.epsilon = epsilon
        self.sigma = sigma
        self.atom_mass = atom_mass
        self.seed = seed
        self.initial_positions = initial_positions

        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.calculate_LJunits_prefactors()
        self.nondimensionalize_units_1()
        self.define_box()
        self.calculate_cross_coefficients()
        self.populate_box()
        self.write_data_file(filename="initial.data",
                               velocity=False)

    def nondimensionalize_units_1(self):
        """Use LJ prefactors to convert units into non-dimensional."""
        # Normalize box dimensions
        box_dimensions = []
        for L in self.box_dimensions:
            box_dimensions.append(L/self.reference_distance)
        self.box_dimensions = box_dimensions
        # Normalize LJ properties
        epsilon, sigma, atom_mass = [], [], []
        for e0, s0, m0 in zip(self.epsilon, self.sigma, self.atom_mass):
            epsilon.append(e0/self.reference_energy)
            sigma.append(s0/self.reference_distance)
            atom_mass.append(m0/self.reference_mass)
        self.epsilon = epsilon
        self.sigma = sigma
        self.atom_mass = atom_mass

    def define_box(self):
        """Define box boundaries based on the box dimensions."""
        box_boundaries = np.zeros((self.dimensions, 2))
        for dim, L in zip(range(self.dimensions), self.box_dimensions):
            box_boundaries[dim] = -L/2, L/2
        self.box_boundaries = box_boundaries
        # Also define the box size following MDAnalysis conventions
        box_size = np.diff(self.box_boundaries).reshape(3)
        box_geometry = np.array([90, 90, 90])
        self.box_size = np.array(box_size.tolist()+box_geometry.tolist())
        
    def populate_box(self):
        """Place atoms at random positions within the box."""
        if self.initial_positions is None:
            atoms_positions = np.zeros((self.total_number_atoms, self.dimensions))
            for dim in np.arange(self.dimensions):
                atoms_positions[:, dim] = np.random.random(self.total_number_atoms)*np.diff(self.box_boundaries[dim]) - np.diff(self.box_boundaries[dim])/2    
            self.atoms_positions = atoms_positions
        else:
            self.atoms_positions = self.initial_positions

