from scipy import constants as cst
from decimal import Decimal
import numpy as np
import copy

import warnings
warnings.filterwarnings('ignore')

class InitializeSimulation:
    def __init__(self,
                 number_atoms,
                 Lx,
                 dimensions=3,
                 Ly=None,
                 Lz=None,
                 epsilon=0.1,
                 sigma=1,
                 atom_mass=1,
                 seed=None,
                 desired_temperature=300,
                 desired_pressure=1,
                 provided_positions=None,
                 provided_velocities=None,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.number_atoms = number_atoms
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.dimensions = dimensions
        self.epsilon = epsilon
        self.sigma = sigma
        self.atom_mass = atom_mass
        self.seed = seed
        self.desired_temperature = desired_temperature
        self.desired_pressure = desired_pressure
        self.provided_positions = provided_positions
        self.provided_velocities = provided_velocities

        if self.seed is not None:
            np.random.seed(self.seed)

        self.reference_distance = self.sigma
        self.reference_energy = self.epsilon
        self.reference_mass = self.atom_mass
        self.reference_time = np.sqrt((self.reference_mass/cst.kilo/cst.Avogadro)*(self.reference_distance*cst.angstrom)**2/(self.reference_energy*cst.calorie*cst.kilo/cst.Avogadro))/cst.femto

        self.Lx = self.nondimensionalise_units(self.Lx, "distance")
        self.Ly = self.nondimensionalise_units(self.Ly, "distance")
        self.Lz = self.nondimensionalise_units(self.Lz, "distance")
        self.epsilon = self.nondimensionalise_units(self.epsilon, "energy")
        self.sigma = self.nondimensionalise_units(self.sigma, "distance")
        self.atom_mass = self.nondimensionalise_units(self.atom_mass, "mass")
        self.desired_temperature = self.nondimensionalise_units(self.desired_temperature, "temperature")
        self.desired_pressure = self.nondimensionalise_units(self.desired_pressure, "pressure")

        self.initialize_box()
        self.initialize_atoms()
        self.populate_box()
        self.give_velocity()
        self.write_lammps_data(filename="initial.data")

    def nondimensionalise_units(self, variable, type):
        kB = cst.Boltzmann*cst.Avogadro/cst.calorie/cst.kilo # kCal/mol/K
        if variable is not None:
            if type == "distance":
                variable /= self.reference_distance
            elif type == "energy":
                variable /= self.reference_energy
            elif type == "mass":
                variable /= self.reference_mass
            elif type == "temperature":
                variable /= self.reference_energy/kB
            elif type == "time":
                variable /= self.reference_time
            elif type == "pressure":
                variable *= cst.atm*cst.angstrom**3*cst.Avogadro/cst.calorie/cst.kilo/self.reference_energy*self.reference_distance**3
            else:
                print("Unknown variable type", type)
        return variable

    def initialize_box(self):
        """Define box boundaries based on Lx, Ly, and Lz.

        If Ly or Lz or both are None, then Lx is used instead"""
        box_boundaries = np.zeros((self.dimensions, 2))
        for dim, L in zip(range(self.dimensions), [self.Lx, self.Ly, self.Lz]):
            if L is not None:
                box_boundaries[dim] = -L/2, L/2
            else:
                box_boundaries[dim] = -self.Lx/2, self.Lx/2
        box_boundaries = self.nondimensionalise_units(box_boundaries, "distance")
        self.box_boundaries = box_boundaries

    def initialize_atoms(self):
        """"""    
        if isinstance(self.number_atoms, list):
            self.total_number_atoms = np.sum(self.number_atoms)
        else:
            self.total_number_atoms = self.number_atoms
        
    def populate_box(self):
        """Place atoms at random positions within the box."""
        atoms_positions = np.zeros((self.total_number_atoms, self.dimensions))
        if self.provided_positions is not None: # tofix : do we really want provided positions ?
            atoms_positions = self.provided_positions/self.reference_distance
        else:
            for dim in np.arange(self.dimensions):
                atoms_positions[:, dim] = np.random.random(self.total_number_atoms)*np.diff(self.box_boundaries[dim]) - np.diff(self.box_boundaries[dim])/2    
        self.atoms_positions = atoms_positions

    def give_velocity(self):
        """Give velocity to atoms so that the initial temperature is the desired one."""
        if self.provided_velocities is not None:
            atoms_velocities = self.provided_velocities/self.reference_distance*self.reference_time
        else:
            atoms_velocities = np.zeros((self.total_number_atoms, self.dimensions))
            for dim in np.arange(self.dimensions):
                atoms_velocities = np.zeros((self.total_number_atoms, self.dimensions))
                atoms_velocities[:, dim] = np.random.normal(size=self.total_number_atoms)
        self.atoms_velocities = atoms_velocities
        self.calculate_temperature()
        scale = np.sqrt(1+((self.desired_temperature/self.temperature)-1))
        self.atoms_velocities *= scale
