import numpy as np
from scipy import constants as cst


class Prepare:
    def __init__(self,
                ureg, # Pint unit registry
                number_atoms, # List - no unit
                epsilon, # List - Kcal/mol
                sigma, # List - Angstrom
                atom_mass,  # List - g/mol
                *args,
                **kwargs):
        self.ureg = ureg
        self.number_atoms = number_atoms
        self.epsilon = epsilon
        self.sigma = sigma
        self.atom_mass = atom_mass
        super().__init__(*args, **kwargs)
        self.calculate_LJunits_prefactors()
        self.nondimensionalize_units(["epsilon", "sigma", "atom_mass"])
        self.identify_atom_properties()


    def calculate_LJunits_prefactors(self):
        """Calculate the Lennard-Jones units prefactors."""
        # First define Boltzmann and Avogadro constants
        kB = cst.Boltzmann*cst.Avogadro/cst.calorie/cst.kilo  # kcal/mol/K
        kB *= self.ureg.kcal/self.ureg.mol/self.ureg.kelvin
        Na = cst.Avogadro/self.ureg.mol
        # Define the reference distance, energy, and mass
        self.ref_length = self.sigma[0]  # Angstrom
        self.ref_energy = self.epsilon[0]  # kcal/mol
        self.ref_mass = self.atom_mass[0]  # g/mol
        # Optional: assert that units were correctly provided by users
        assert self.ref_length.units == self.ureg.angstrom, \
            f"Error: Provided sigma has wrong units, should be angstrom"
        assert self.ref_energy.units == self.ureg.kcal/self.ureg.mol, \
            f"Error: Provided epsilon has wrong units, should be kcal/mol"
        assert self.ref_mass.units == self.ureg.g/self.ureg.mol, \
            f"Error: Provided mass has wrong units, should be g/mol"
        # Calculate the prefactor for the time (in femtosecond)
        self.ref_time = np.sqrt(self.ref_mass \
            *self.ref_length**2/self.ref_energy).to(self.ureg.femtosecond)
        # Calculate the prefactor for the temperature (in Kelvin)
        self.ref_temperature = self.ref_energy/kB  # Kelvin
        # Calculate the prefactor for the pressure (in Atmosphere)
        self.ref_pressure = (self.ref_energy \
            /self.ref_length**3/Na).to(self.ureg.atmosphere)
        # Group all the reference quantities into a list for practicality
        self.ref_quantities = [self.ref_length, self.ref_energy,
            self.ref_mass, self.ref_time, self.ref_pressure, self.ref_temperature]
        self.ref_units = [ref.units for ref in self.ref_quantities]

    def nondimensionalize_units(self, quantities_to_normalise):
        for name in quantities_to_normalise:
            quantity = getattr(self, name)  # Get the attribute by name
            if isinstance(quantity, list):
                for i, element in enumerate(quantity):
                    assert element.units in self.ref_units, \
                        f"Error: Units not part of the reference units"
                    ref_value = self.ref_quantities[self.ref_units.index(element.units)]
                    quantity[i] = element/ref_value
                    assert quantity[i].units == self.ureg.dimensionless, \
                        f"Error: Quantities are not properly nondimensionalized"
                    quantity[i] = quantity[i].magnitude # get rid of ureg
                setattr(self, name, quantity)
            elif len(np.shape(quantity)) > 0: # for position array
                assert element.units in self.ref_units, \
                    f"Error: Units not part of the reference units"
                ref_value = self.ref_quantities[self.ref_units.index(element.units)]
                quantity = quantity/ref_value
                assert quantity.units == self.ureg.dimensionless, \
                    f"Error: Quantities are not properly nondimensionalized"
                quantity = quantity.magnitude # get rid of ureg
                setattr(self, name, quantity)
            else:
                if quantity is not None:
                    assert np.shape(quantity) == (), \
                        f"Error: The quantity is a list or an array"
                    assert quantity.units in self.ref_units, \
                        f"Error: Units not part of the reference units"
                    ref_value = self.ref_quantities[self.ref_units.index(quantity.units)]
                    quantity = quantity/ref_value
                    assert quantity.units == self.ureg.dimensionless, \
                        f"Error: Quantities are not properly nondimensionalized"
                    quantity = quantity.magnitude # get rid of ureg
                    setattr(self, name, quantity)

    def identify_atom_properties(self):
        """Identify the properties for each atom."""
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
