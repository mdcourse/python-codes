import numpy as np
from scipy import constants as cst


class Prepare:
    def __init__(self,
                number_atoms=[10],  # List - no unit
                epsilon=[0.1],  # List - Kcal/mol
                sigma=[3],  # List - Angstrom
                atom_mass=[10],  # List - g/mol
                *args,
                **kwargs):
        self.number_atoms = number_atoms
        self.epsilon = epsilon
        self.sigma = sigma
        self.atom_mass = atom_mass
        super().__init__(*args, **kwargs)
        self.calculate_LJunits_prefactors()
        self.nondimensionalize_units_0()
        self.identify_atom_properties()


    def calculate_LJunits_prefactors(self):
        """Calculate the Lennard-Jones units prefacors."""
        # Define the reference distance, energy, and mass
        self.reference_distance = self.sigma[0]  # Angstrom
        self.reference_energy = self.epsilon[0]  # kcal/mol
        self.reference_mass = self.atom_mass[0]  # g/mol
        # Calculate the prefactor for the time
        mass_kg = self.atom_mass[0]/cst.kilo/cst.Avogadro  # kg
        epsilon_J = self.epsilon[0]*cst.calorie*cst.kilo/cst.Avogadro  # J
        sigma_m = self.sigma[0]*cst.angstrom  # m
        time_s = np.sqrt(mass_kg*sigma_m**2/epsilon_J)  # s
        self.reference_time = time_s / cst.femto  # fs
        # Calculate the prefactor for the temperature
        kB = cst.Boltzmann*cst.Avogadro/cst.calorie/cst.kilo  # kcal/mol/K
        self.reference_temperature = self.epsilon[0]/kB  # K
        # Calculate the prefactor for the pressure
        pressure_pa = epsilon_J/sigma_m**3  # Pa
        self.reference_pressure = pressure_pa/cst.atm  # atm

    def nondimensionalize_units_0(self):
        # Normalize LJ properties
        epsilon, sigma, atom_mass = [], [], []
        for e0, s0, m0 in zip(self.epsilon, self.sigma, self.atom_mass):
            epsilon.append(e0/self.reference_energy)
            sigma.append(s0/self.reference_distance)
            atom_mass.append(m0/self.reference_mass)
        self.epsilon = epsilon
        self.sigma = sigma
        self.atom_mass = atom_mass

    def identify_atom_properties(self):
        """Identify the atom properties for each atom."""
        self.total_number_atoms = np.sum(self.number_atoms)
        atoms_sigma = []
        atoms_epsilon = []
        atoms_mass = []
        atoms_type = []
        for parts in zip(self.sigma,
                        self.epsilon,
                        self.atom_mass,
                        self.number_atoms,
                        np.arange(len(self.number_atoms))+1):
            sigma, epsilon, mass, number_atoms, type = parts
            atoms_sigma += [sigma] * number_atoms
            atoms_epsilon += [epsilon] * number_atoms
            atoms_mass += [mass] * number_atoms
            atoms_type += [type] * number_atoms
        self.atoms_sigma = np.array(atoms_sigma)
        self.atoms_epsilon = np.array(atoms_epsilon)
        self.atoms_mass = np.array(atoms_mass)
        self.atoms_type = np.array(atoms_type)
