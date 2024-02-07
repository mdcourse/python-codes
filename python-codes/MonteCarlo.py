from scipy import constants as cst
from decimal import Decimal
import numpy as np
import copy

import warnings
warnings.filterwarnings('ignore')

from InitializeSimulation import InitializeSimulation
from Utilities import Utilities
from Outputs import Outputs

class MonteCarlo(InitializeSimulation, Utilities, Outputs):
    def __init__(self,
        maximum_steps,
        cut_off = 10,
        displace_mc=None,
        mu = None,
        *args,
        **kwargs,
        ):

        self.maximum_steps = maximum_steps
        self.cut_off = cut_off
        self.displace_mc = displace_mc
        self.mu = mu
        super().__init__(*args, **kwargs)

        self.cut_off = self.nondimensionalise_units(self.cut_off, "distance")
        self.displace_mc = self.nondimensionalise_units(self.displace_mc, "distance")
        self.mu = self.nondimensionalise_units(self.mu, "energy")

    def run(self):
        """Perform the loop over time."""
        
        for self.step in range(0, self.maximum_steps+1):
            if self.displace_mc is not None:
                self.monte_carlo_displacement()
            if self.mu is not None:
                self.monte_carlo_insert_delete()
            self.wrap_in_box()
            self.update_log()
            self.update_dump(velocity=False)
        self.write_lammps_data(filename="final.data")

    def monte_carlo_displacement(self):
        beta =  1/self.desired_temperature
        Epot = self.calculate_potential_energy(self.atoms_positions)
        trial_atoms_positions = copy.deepcopy(self.atoms_positions)
        atom_id = np.random.randint(self.number_atoms)
        trial_atoms_positions[atom_id] += (np.random.random(3)-0.5)*self.displace_mc
        trial_Epot = self.calculate_potential_energy(trial_atoms_positions)
        acceptation_probability = np.min([1, np.exp(-beta*(trial_Epot-Epot))])
        if np.random.random() <= acceptation_probability:
            self.atoms_positions = trial_atoms_positions
            self.Epot = trial_Epot
        else:
            self.Epot = Epot 

    def calculate_Lambda(self, mass):
        """Estimate de Broglie wavelength in LJ units."""
        m_kg = mass/cst.Avogadro*cst.milli*self.reference_mass
        kB_kCal_mol_K = cst.Boltzmann*cst.Avogadro/cst.calorie/cst.kilo
        T_K = self.desired_temperature*self.reference_energy/kB_kCal_mol_K
        Lambda = cst.h/np.sqrt(2*np.pi*cst.Boltzmann*m_kg*T_K)/cst.angstrom
        return self.nondimensionalise_units(Lambda, "distance")

    def monte_carlo_insert_delete(self):
        Epot = self.calculate_potential_energy(self.atoms_positions)
        trial_atoms_positions = copy.deepcopy(self.atoms_positions)
        if np.random.random() < 0.5:
            number_atoms = self.number_atoms + 1
            atom_position = np.zeros((1, self.dimensions))
            for dim in np.arange(self.dimensions):
                atom_position[:, dim] = np.random.random(1)*np.diff(self.box_boundaries[dim]) - np.diff(self.box_boundaries[dim])/2
            trial_atoms_positions = np.vstack([trial_atoms_positions, atom_position])
            trial_Epot = self.calculate_potential_energy(trial_atoms_positions, number_atoms = number_atoms)
            Lambda = self.calculate_Lambda(self.atom_mass)
            volume = np.prod(np.diff(self.box_boundaries))
            beta = 1/self.desired_temperature
            acceptation_probability = np.min([1, volume/(Lambda**self.dimensions*(self.number_atoms + 1))*np.exp(beta*(self.mu-trial_Epot+Epot))])
        else:
            number_atoms = self.number_atoms - 1
            if number_atoms > 0:
                atom_id = np.random.randint(self.number_atoms)
                trial_atoms_positions = np.delete(trial_atoms_positions, atom_id, axis=0)
                trial_Epot = self.calculate_potential_energy(trial_atoms_positions, number_atoms = number_atoms)
                Lambda = self.calculate_Lambda(self.atom_mass)
                volume = np.prod(np.diff(self.box_boundaries))
                beta = 1/self.desired_temperature
                acceptation_probability = np.min([1, (Lambda**self.dimensions*(self.number_atoms)/volume)*np.exp(-beta*(self.mu+trial_Epot-Epot))])
            else:
                acceptation_probability = 0
        if np.random.random() < acceptation_probability:
            self.atoms_positions = trial_atoms_positions
            self.Epot = trial_Epot
            self.number_atoms = number_atoms
        else:
            self.Epot = Epot
