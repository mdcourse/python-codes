import os
import numpy as np
from prepare_simulation import Prepare
from simulation_handler import Utilities


class InitializeSimulation(Prepare, Utilities):
    def __init__(self,
                box_dimensions,  # List - Angstroms
                cut_off, # Angstroms
                initial_positions=None,  # Array - Angstroms
                neighbor=1, # Integer - frequency for neighbor list rebuild
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

        # If needed, create folder for output
        if os.path.exists(self.data_folder) is False:
            os.mkdir(self.data_folder)

        self.nondimensionalize_units(["box_dimensions", "cut_off", "initial_positions"])
        self.define_box()
        self.populate_box()
        self.update_neighbor_lists()
        self.update_cross_coefficients()

    def define_box(self):
        """Define the simulation box. Supports both 2D and 3D boxes."""

        # Optional: Check if the number of elements in box_dimensions is 2 or 3
        if len(self.box_dimensions) not in [2, 3]:
            raise ValueError("box_dimensions must have exactly two or three elements.")

        # Ensure the box is 3D. If its it's 2D, then append a 0 along z
        box_dimensions_3d = self.box_dimensions + [0] * (3 - len(self.box_dimensions))

        # Loop through each dimension
        box_boundaries = np.zeros((3, 2))
        for dim, length in enumerate(box_dimensions_3d):
            box_boundaries[dim, 0] = -length / 2  # Set the min boundary
            box_boundaries[dim, 1] = length / 2  # Set the max boundary
        self.box_boundaries = box_boundaries

        # Create MDAnalysis box (Lx, Ly, Lz, 90, 90, 90)
        box_geometry = np.array([90, 90, 90])
        self.box_mda = np.array(box_dimensions_3d + box_geometry.tolist())

        # Optional: Validate box_mda length (should be 6)
        if len(self.box_mda) != 6:
            raise ValueError("box_mda has wrong length")

    def populate_box(self):
        """Populate the simulation box with atom positions."""
        total_atoms = np.sum(self.number_atoms) # total number of atoms

        # Check if initial positions for atoms have been provided
        if self.initial_positions is None:
            # Create random positions within the box
            atoms_positions = np.zeros((total_atoms, 3))
            for dim in range(len(self.box_dimensions)):

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
