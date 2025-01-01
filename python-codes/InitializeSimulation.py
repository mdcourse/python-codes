import os
import numpy as np
from Prepare import Prepare
from Utilities import Utilities


class InitializeSimulation(Prepare, Utilities):
    def __init__(self,
                box_dimensions,  # List - Angstroms
                cut_off, # Angstroms
                initial_positions=None,  # Array - Angstroms
                neighbor=1, # Integer
                thermo_period = None,
                dumping_period = None,
                thermo_outputs = None,
                data_folder="Outputs/",
                *args,
                **kwargs,
                ):
        super().__init__(*args, **kwargs)
        self.box_dimensions = box_dimensions
        self.cut_off = cut_off
        self.neighbor = neighbor
        self.step = 0 # initialize simulation step
        self.initial_positions = initial_positions
        self.thermo_period = thermo_period
        self.dumping_period = dumping_period
        self.thermo_outputs = thermo_outputs
        self.data_folder = data_folder
        if os.path.exists(self.data_folder) is False:
            os.mkdir(self.data_folder)

        self.nondimensionalize_units(["box_dimensions",
                                      "cut_off",
                                      "initial_positions"])
        self.define_box()
        self.populate_box()
        self.update_neighbor_lists()
        self.update_cross_coefficients()

    def define_box(self):
        """Define the simulation box. Only 3D boxes are supported."""

        # Optional: Check if the number of elements in box_dimensions is exactly 3
        if len(self.box_dimensions) != 3:
            raise ValueError("box_dimensions must have exactly three elements (for 3D).")

        # Initialize a 3x2 array to hold the boundaries
        box_boundaries = np.zeros((3, 2))

        # Loop through each dimension (x, y, z)
        for dim, length in enumerate(self.box_dimensions):
            box_boundaries[dim] = -length / 2, length / 2  # Set the min and max boundary for each dimension

        self.box_boundaries = box_boundaries
        box_size = self.box_boundaries[:, 1] - self.box_boundaries[:, 0]  # SG TOFIX Redundant !
        box_geometry = np.array([90, 90, 90])
        self.box_size = np.array(box_size.tolist()+box_geometry.tolist())

    def populate_box(self):
        """Populate the simulation box with atom positions."""
        total_atoms = np.sum(self.number_atoms) # total number of atoms

        # Check if initial positions for atoms have been provided
        if self.initial_positions is None:
            # Create random positions within the box
            atoms_positions = np.zeros((total_atoms, 3))
            for dim in np.arange(3):
                # Get the size of the box in the current dimension
                diff_box = np.diff(self.box_boundaries[dim])

                # Generate random positions between 0 and 1 for each atom
                random_pos = np.random.random(total_atoms)

                # Scale and shift the random positions so that they lie within the box 
                atoms_positions[:, dim] = random_pos*diff_box-diff_box/2
            self.atoms_positions = atoms_positions
        else:
            # If initial positions are provided, use them directly
            self.atoms_positions = self.initial_positions
