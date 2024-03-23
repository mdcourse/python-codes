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
                 epsilon=[0.1],
                 sigma=[1],
                 atom_mass=[1],
                 Ly=None,
                 Lz=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs) 

        self.number_atoms = number_atoms
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.dimensions = 3
        self.epsilon = epsilon
        self.sigma = sigma
        self.atom_mass = atom_mass

        self.calculate_LJunits_prefactors()
        self.nondimensionalize_units()
        self.define_box()
        self.identify_atom_properties()
        self.populate_box()

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

    def identify_atom_properties(self):
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
        self.box_size = np.diff(box_boundaries).reshape(3)

    def populate_box(self):
        """Place atoms at random positions within the box."""
        atoms_positions = np.zeros((self.total_number_atoms, self.dimensions))
        for dim in np.arange(self.dimensions):
            atoms_positions[:, dim] = np.random.random(self.total_number_atoms)*np.diff(self.box_boundaries[dim]) - np.diff(self.box_boundaries[dim])/2    
        self.atoms_positions = atoms_positions