import numpy as np
from scipy import constants as cst
from typing import List
import pint

from unit_conversion_utilities import validate_units, nondimensionalize_single, nondimensionalize_array
from file_parsers import read_data_file, read_inc_file

class Prepare:
    def __init__(self,
                ureg: pint.UnitRegistry,  # Pint unit registry
                data_file: str,
                parameter_file: str,
                potential_type: str = "LJ",  # Default value, explicitly typed as str
                *args,
                **kwargs):
        self.ureg = ureg

        masses, pair_coeffs = read_inc_file(parameter_file, ureg)
        number_atoms, atom_ids, atom_types, positions_array, box_bounds = read_data_file(data_file)

        self.number_atoms = number_atoms
        self.epsilon = epsilon_list
        self.sigma = sigma_list
        self.atom_mass = atom_mass_list

        self.potential_type = potential_type

        super().__init__(*args, **kwargs)

        self.calculate_LJunits_prefactors()
        self.nondimensionalize_units(["epsilon", "sigma", "atom_mass"])
        self.assign_atom_properties()

    def calculate_LJunits_prefactors(self):
        """Calculate the Lennard-Jones units prefactors."""
        kB = cst.Boltzmann * cst.Avogadro / cst.calorie / cst.kilo  # Boltzmann constant in kcal/mol/K
        kB *= self.ureg.kcal / self.ureg.mol / self.ureg.kelvin  # give pint units to kB
        Na = cst.Avogadro / self.ureg.mol  # Avogadro number in mol-1

        # Reference values
        self.ref_length = self.sigma[0]  # Angstrom
        self.ref_energy = self.epsilon[0]  # kcal/mol
        self.ref_mass = self.atom_mass[0]  # g/mol

        validate_units(self.ref_length, self.ureg.angstrom, "sigma")
        validate_units(self.ref_energy, self.ureg.kcal / self.ureg.mol, "epsilon")
        validate_units(self.ref_mass, self.ureg.g / self.ureg.mol, "mass")

        self.ref_time = np.sqrt(self.ref_mass * self.ref_length**2 / self.ref_energy).to(self.ureg.femtosecond)
        self.ref_temperature = self.ref_energy / kB  # Kelvin
        self.ref_pressure = (self.ref_energy / self.ref_length**3 / Na).to(self.ureg.atmosphere)

        self.ref_quantities = [
            self.ref_length, self.ref_energy, self.ref_mass,
            self.ref_time, self.ref_pressure, self.ref_temperature]
        self.ref_units = [ref.units for ref in self.ref_quantities]

    def nondimensionalize_units(self, quantities_to_normalize: List[str]):
        """Converts specified quantities to nondimensionalized form."""
        for name in quantities_to_normalize:
            quantity = getattr(self, name)
            if isinstance(quantity, list):
                ref_value = self.get_ref_value(quantity[0])
                nondimensionalized = [nondimensionalize_single(q, ref_value) for q in quantity]
                setattr(self, name, nondimensionalized)
            elif isinstance(quantity, np.ndarray):
                ref_value = self.get_ref_value(quantity[0])
                nondimensionalized = nondimensionalize_array(quantity, ref_value)
                setattr(self, name, nondimensionalized)
            elif isinstance(quantity, pint.Quantity):
                ref_value = self.get_ref_value(quantity)
                nondimensionalized = nondimensionalize_single(quantity, ref_value)
                setattr(self, name, nondimensionalized)
            else:
                return None

    def get_ref_value(self, value: pint.Quantity) -> pint.Quantity:
        """Get the reference value for a given unit."""
        assert value.units in self.ref_units, f"Error: Units not part of the reference units"
        return self.ref_quantities[self.ref_units.index(value.units)]

    def assign_atom_properties(self):
        """Assign properties to each atom."""
        atom_data = {
            "sigma": self.sigma,
            "epsilon": self.epsilon,
            "mass": self.atom_mass,
            "type": np.arange(len(self.number_atoms)) + 1,
        }

        total_atoms = np.sum(self.number_atoms)
        atom_properties = {key: np.empty(total_atoms) for key in atom_data.keys()}

        index = 0
        for i, count in enumerate(self.number_atoms):
            for key, values in atom_data.items():
                atom_properties[key][index:index + count] = values[i]
            index += count

        self.atoms_sigma = atom_properties["sigma"]
        self.atoms_epsilon = atom_properties["epsilon"]
        self.atoms_mass = atom_properties["mass"]
        self.atoms_type = atom_properties["type"]
