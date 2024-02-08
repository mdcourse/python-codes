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

        self.assert_correctness_parameters()

        self.reference_distance = self.sigma[0]
        self.reference_energy = self.epsilon[0]
        self.reference_mass = self.atom_mass[0]

        self.reference_time = np.sqrt((self.reference_mass/cst.kilo/cst.Avogadro)
                                      *(self.reference_distance*cst.angstrom)**2
                                      /(self.reference_energy*cst.calorie*cst.kilo/cst.Avogadro))/cst.femto
        self.Lx = self.nondimensionalise_units(self.Lx, "distance")
        self.Ly = self.nondimensionalise_units(self.Ly, "distance")
        self.Lz = self.nondimensionalise_units(self.Lz, "distance")

        epsilon = []
        for epsilon0 in self.epsilon:
            epsilon.append(self.nondimensionalise_units(epsilon0, "energy"))
        self.epsilon = epsilon
        sigma = []
        for sigma0 in self.sigma:
            sigma.append(self.nondimensionalise_units(sigma0, "distance"))
        self.sigma = np.array(sigma)
        atom_mass = []
        for atom_mass0 in self.atom_mass:
            atom_mass.append(self.nondimensionalise_units(atom_mass0, "mass"))
        self.atom_mass = np.array(atom_mass)

        self.desired_temperature = self.nondimensionalise_units(self.desired_temperature, "temperature")
        self.desired_pressure = self.nondimensionalise_units(self.desired_pressure, "pressure")
        
        self.initialize_box()
        self.initialize_atoms()
        self.populate_box()
        self.set_initial_velocity()
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

    def assert_correctness_parameters(self):
        """Assert that the parameters entered are correct"""
        if isinstance(self.number_atoms, list):
            assert isinstance(self.sigma, list)
            assert isinstance(self.epsilon, list)
            assert isinstance(self.atom_mass, list)
            assert len(self.number_atoms) == len(self.sigma)
            assert len(self.number_atoms) == len(self.epsilon)
            assert len(self.number_atoms) == len(self.atom_mass)
        else:
            assert isinstance(self.sigma, int)
            assert isinstance(self.epsilon, int)
            assert isinstance(self.atom_mass, int)
            # if entries are integer, convert to list with 1 element
            self.number_atoms = [self.number_atoms]
            self.epsilon = [self.epsilon]
            self.atom_mass = [self.atom_mass]
            self.sigma = [self.sigma]

    def initialize_box(self):
        """Define box boundaries based on Lx, Ly, and Lz.
        If Ly or Lz or both are None, then Lx is used
        along the y and z instead"""
        box_boundaries = np.zeros((self.dimensions, 2))
        for dim, L in zip(range(self.dimensions), [self.Lx, self.Ly, self.Lz]):
            if L is not None:
                box_boundaries[dim] = -L/2, L/2
            else:
                box_boundaries[dim] = -self.Lx/2, self.Lx/2
        self.box_boundaries = self.nondimensionalise_units(box_boundaries, "distance")

    def initialize_atoms(self):
        """Create initial atom array from input parameters"""
        self.total_number_atoms = np.sum(self.number_atoms)
        atoms_sigma = []
        atoms_epsilon = []
        atoms_mass = []
        atoms_type = []
        for sigma, epsilon, mass, number_atoms, type in zip(self.sigma, self.epsilon,
                                                            self.atom_mass, self.number_atoms,
                                                            np.arange(len(self.number_atoms))+1):
            atoms_sigma += [sigma] * number_atoms
            atoms_epsilon += [epsilon] * number_atoms
            atoms_mass += [mass] * number_atoms
            atoms_type += [type] * number_atoms
        self.atoms_sigma = np.array(atoms_sigma)
        self.atoms_epsilon = np.array(atoms_epsilon)
        self.atoms_mass = np.array(atoms_mass)
        self.atoms_type = np.array(atoms_type)
        
    def populate_box(self):
        """Place atoms at random positions within the box."""
        atoms_positions = np.zeros((self.total_number_atoms, self.dimensions))
        if self.provided_positions is not None: # tofix : do we really want provided positions ?
            atoms_positions = self.provided_positions/self.reference_distance
        else:
            for dim in np.arange(self.dimensions):
                atoms_positions[:, dim] = np.random.random(self.total_number_atoms)*np.diff(self.box_boundaries[dim]) - np.diff(self.box_boundaries[dim])/2    
        self.atoms_positions = atoms_positions

    def set_initial_velocity(self):
        """Give velocity to atoms so that the initial temperature is the desired one."""
        if self.provided_velocities is not None:
            atoms_velocities = self.provided_velocities/self.reference_distance*self.reference_time
        else:
            atoms_velocities = np.zeros((self.total_number_atoms, self.dimensions))
            for dim in np.arange(self.dimensions):
                atoms_velocities[:, dim] = np.random.normal(size=self.total_number_atoms)
        self.atoms_velocities = atoms_velocities
        self.calculate_temperature()
        scale = np.sqrt(1+((self.desired_temperature/self.temperature)-1))
        self.atoms_velocities *= scale
