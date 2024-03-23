from scipy import constants as cst
from decimal import Decimal
import numpy as np
import copy

import warnings
warnings.filterwarnings('ignore')

class InitializeSimulation:
    def __init__(self,
                 epsilon=[0.1],
                 sigma=[1],
                 atom_mass=[1],
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs) 

        self.epsilon = epsilon
        self.sigma = sigma
        self.atom_mass = atom_mass

        self.calculate_LJunits_prefactors()
        self.nondimensionalize_units()

    def nondimensionalize_units(self):
        """Use LJ prefactors to convert units into non-dimensional."""
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