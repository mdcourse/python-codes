import numpy as np

import warnings
warnings.filterwarnings('ignore')

from Prepare import Prepare

class InitializeSimulation(Prepare):
    def __init__(self,
                 box_dimensions=[10, 10, 10], # List - Angstroms
                 seed=None, # Int
                 initial_positions=None, # Array - Angstroms
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.box_dimensions = box_dimensions
        self.dimensions = len(box_dimensions)
        self.seed = seed
        self.initial_positions = initial_positions

        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.nondimensionalize_units_1()
        self.define_box()
        self.populate_box()

    def nondimensionalize_units_1(self):
        """Use LJ prefactors to convert units into non-dimensional."""
        # Normalize box dimensions
        box_dimensions = []
        for L in self.box_dimensions:
            box_dimensions.append(L/self.reference_distance)
        self.box_dimensions = box_dimensions

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

