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

        masses, epsilons, sigmas = read_inc_file(parameter_file, ureg)
        number_atoms, atom_ids, atom_types, positions, box_bounds = read_data_file(data_file, ureg)

        self.masses = masses
        self.epsilons = epsilons
        self.sigmas = sigmas
        self.atom_ids = atom_ids
        self.atom_types = atom_types
        self.positions = positions
        self.box_bounds = box_bounds

        self.number_atoms = number_atoms

        self.potential_type = potential_type

        self.calculate_LJunits_prefactors()
        self.nondimensionalize_units(["epsilons",
                                      "sigmas",
                                      "masses",
                                      "box_bounds",
                                      "positions"])
        
        box_geometry = np.array([90, 90, 90])
        box_lengths = np.diff(self.box_bounds, axis=1).flatten()
        self.box_mda = np.concatenate([box_lengths, box_geometry])

    @property
    def kB(self):
        # Boltzmann constant in kcal/mol/K
        return (cst.Boltzmann * cst.Avogadro / cst.calorie / cst.kilo) * self.ureg.kcal / self.ureg.mol / self.ureg.kelvin

    @property
    def Na(self):
        # Avogadro number in mol-1
        return cst.Avogadro / self.ureg.mol

    def calculate_LJunits_prefactors(self):
        """Calculate the Lennard-Jones units prefactors."""

        # Select reference values (first atom type)
        self.ref_length = self.sigmas[0, 0]  # Angstrom
        self.ref_energy = self.epsilons[0, 0]  # kcal/mol
        self.ref_mass = self.masses[0]  # g/mol

        validate_units(self.ref_length, self.ureg.angstrom, "sigma")
        validate_units(self.ref_energy, self.ureg.kcal / self.ureg.mol, "epsilon")
        validate_units(self.ref_mass, self.ureg.g / self.ureg.mol, "mass")

        self.ref_time = np.sqrt(self.ref_mass * self.ref_length**2 / self.ref_energy).to(self.ureg.femtosecond)
        self.ref_temperature = self.ref_energy / self.kB  # Kelvin
        self.ref_pressure = (self.ref_energy / self.ref_length**3 / self.Na).to(self.ureg.atmosphere)

        self.ref_quantities = [
            self.ref_length, self.ref_energy, self.ref_mass,
            self.ref_time, self.ref_pressure, self.ref_temperature]
        self.ref_units = [ref.units for ref in self.ref_quantities]

    def nondimensionalize_units(self, quantities_to_normalize: List[str]):
        """Converts specified quantities to nondimensionalized form, preserving their shape."""
        for name in quantities_to_normalize:
            quantity = getattr(self, name)

            if isinstance(quantity, list):
                if isinstance(quantity[0], float):
                    continue  # already nondimensional
                ref_value = self.get_ref_value(quantity[0])
                nondimensionalized = [nondimensionalize_single(q, ref_value) for q in quantity]
                setattr(self, name, nondimensionalized)

            elif isinstance(quantity, np.ndarray):
                if np.issubdtype(quantity.dtype, np.floating):
                    continue  # already nondimensional
                ref_value = self.get_ref_value(quantity.flat[0])
                nondimensionalized = nondimensionalize_array(quantity, ref_value)
                setattr(self, name, nondimensionalized)

            elif isinstance(quantity, pint.Quantity):
                ref_value = self.get_ref_value(quantity)
                nondimensionalized = nondimensionalize_single(quantity, ref_value)
                setattr(self, name, nondimensionalized)

            else:
                raise TypeError(f"Unsupported type for nondimensionalization: {type(quantity)}: {name}")

    def get_ref_value(self, value: pint.Quantity) -> pint.Quantity:
        """Get the reference value for a given unit."""
        assert value.units in self.ref_units, f"Error: Units not part of the reference units"
        return self.ref_quantities[self.ref_units.index(value.units)]
