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
                 Ly=None,
                 Lz=None,
                 epsilon=[0.1],
                 sigma=[1],
                 atom_mass=[1],
                 seed=None,
                 desired_temperature=300,
                 desired_pressure=1,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.number_atoms = number_atoms
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.dimensions = 3
        self.epsilon = epsilon
        self.sigma = sigma
        self.atom_mass = atom_mass
        self.seed = seed
        self.desired_temperature = desired_temperature
        self.desired_pressure = desired_pressure
        if self.seed is not None:
            np.random.seed(self.seed)
        self.assert_correctness_parameters()
        
        self.calculate_LJunits_prefactors()
        self.nondimensionalize_units()
        self.define_box()
        self.identify_atom_properties()
        self.calculate_cross_coefficients()
        self.populate_box()
        self.set_initial_velocity()
        self.write_lammps_data(filename="initial.data")

    def nondimensionalize_units(self):
        """Use LJ prefactors to convert units into non-dimensional."""
        self.Lx /= self.reference_distance
        if self.Ly is not None:
            self.Ly /= self.reference_distance
        if self.Lz is not None:
            self.Lz /= self.reference_distance
        epsilon, sigma, atom_mass = [], [], []
        for e0, s0, m0 in zip(self.epsilon, self.sigma, self.atom_mass):
            epsilon.append(e0/self.reference_energy)
            sigma.append(s0/self.reference_distance)
            atom_mass.append(m0/self.reference_mass)
        self.epsilon = epsilon
        self.sigma = sigma
        self.atom_mass = atom_mass
        self.desired_temperature /= self.reference_temperature
        self.desired_pressure /= self.reference_pressure

    def calculate_LJunits_prefactors(self):
        """Calculate LJ non-dimensional units.
        
        Distances, energies, and masses are normalized by
        the $\sigma$, $\epsilon$, and $m$ parameters from the
        first type of atom.
        In addition:
        - Times are normalized by $\sqrt{m \sigma^2 / \epsilon}$.
        - Temperature are normalized by $\epsilon/k_\text{B}$, 
          where $k_\text{B}$ is the Boltzmann constant.
        - Pressures are normalized by $\epsilon/\sigma^3$.
        """
        self.reference_distance = self.sigma[0] # Angstrom
        self.reference_energy = self.epsilon[0] # Kcal/mol
        self.reference_mass = self.atom_mass[0] # g/mol
        mass_kg = self.atom_mass[0]/cst.kilo/cst.Avogadro # kg
        epsilon_J = self.epsilon[0]*cst.calorie*cst.kilo/cst.Avogadro # J
        sigma_m = self.sigma[0]*cst.angstrom # m
        time_s = np.sqrt(mass_kg*sigma_m**2/epsilon_J) # s
        self.reference_time = time_s / cst.femto # fs
        kB = cst.Boltzmann*cst.Avogadro/cst.calorie/cst.kilo # kCal/mol/K
        self.reference_temperature = self.epsilon[0]/kB # K
        pressure_pa = epsilon_J/sigma_m**3 # Pa
        self.reference_pressure = pressure_pa/cst.atm # atm

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
            self.charge = [self.atom_charge]

    def define_box(self):
        """Define box boundaries based on Lx, Ly, and Lz.
        If Ly or Lz or both are None, then Lx is used
        along the y and z instead"""
        box_boundaries = np.zeros((self.dimensions, 2))
        for dim, L in zip(range(self.dimensions),
                          [self.Lx, self.Ly, self.Lz]):
            if L is not None:
                box_boundaries[dim] = -L/2, L/2
            else:
                box_boundaries[dim] = -self.Lx/2, self.Lx/2
        self.box_boundaries = box_boundaries
        box_size = np.diff(self.box_boundaries).reshape(3)
        box_geometry = np.array([90, 90, 90])
        self.box_size = np.array(box_size.tolist()+box_geometry.tolist())

    def identify_atom_properties(self):
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
        for dim in np.arange(self.dimensions):
            atoms_positions[:, dim] = np.random.random(self.total_number_atoms)*np.diff(self.box_boundaries[dim]) - np.diff(self.box_boundaries[dim])/2    
        self.atoms_positions = atoms_positions

    def set_initial_velocity(self):
        """Give velocity to atoms so that the initial temperature is the desired one."""
        self.atoms_velocities = np.random.normal(size=(self.total_number_atoms, self.dimensions))
        self.calculate_temperature()
        scale = np.sqrt(1+((self.desired_temperature/self.temperature)-1))
        self.atoms_velocities *= scale
