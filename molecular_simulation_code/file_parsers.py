import pint
from typing import Tuple, List
import numpy as np

def read_inc_file(
    filepath: str,
    ureg: pint.UnitRegistry
) -> tuple[List[pint.Quantity], List[pint.Quantity], List[pint.Quantity]]:
    """
    Parse a LAMMPS-style `.inc` parameter file and return lists of mass, epsilon, and sigma
    in the order the species are first encountered in the file.

    The function expects the `.inc` file to contain lines of the form:

        mass <species> <value>
        pair_coeff <species1> <species2> <epsilon> <sigma>

    - `mass` lines define the mass of each species in g/mol.
    - `pair_coeff` lines define Lennard-Jones parameters (epsilon in kcal/mol, sigma in Ångström)
      for each pair of species. Only the diagonal terms (`species-species`) are used.

    Comment lines starting with `#` and blank lines are ignored.
    The output lists are ordered by the order of first `mass` appearances in the file.

    Args:
        filepath: Path to `.inc` parameter file.
        ureg: pint.UnitRegistry to assign units to the parsed quantities.

    Returns:
        atom_mass_list: list of pint.Quantity for species masses (g/mol)
        epsilon_list: list of pint.Quantity for species epsilon (kcal/mol)
        sigma_list: list of pint.Quantity for species sigma (Ångström)
    """
    species_order = []
    masses = {}
    pair_coeffs = {}

    with open(filepath, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens or tokens[0].startswith("#"):
                continue

            if tokens[0] == "mass":
                elem, val = tokens[1], float(tokens[2])
                if elem not in species_order:
                    species_order.append(elem)
                masses[elem] = val * ureg.g / ureg.mol

            elif tokens[0] == "pair_coeff":
                elem1, elem2 = tokens[1], tokens[2]
                epsilon, sigma = float(tokens[3]), float(tokens[4])
                pair_coeffs[(elem1, elem2)] = (
                    epsilon * ureg.kcal / ureg.mol,
                    sigma * ureg.angstrom
                )

    return masses, pair_coeffs

def read_data_file(
    filepath: str
) -> Tuple[List[int], np.ndarray]:
    """
    Read a LAMMPS-style `.data` file and return atom types and positions.

    The function expects the `.data` file to follow the standard LAMMPS
    `write_data` output format, where the `Atoms` section looks like:

        Atoms

        atom-ID  atom-type  x  y  z  [other optional flags …]

    Only the atom-type (column 2) and x, y, z coordinates (columns 3-5)
    are extracted. Additional columns in the `Atoms` section are ignored.
    Parsing stops when another section (e.g., `Velocities`) is encountered.

    Args:
        filepath: Path to LAMMPS `.data` file

    Returns:
        atom_types: list of integers, atom type for each atom
        positions: (Nx3) NumPy array of atom positions (floats)
    """
    in_atoms_section = False
    atom_types = []
    atom_ids = []
    positions = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.lower().startswith("atoms"):
                in_atoms_section = True
                continue

            if line.lower().startswith("velocities") or line.lower().startswith("bonds") or line.lower().startswith("angles"):
                in_atoms_section = False
                continue

            if in_atoms_section:
                tokens = line.split()
                atom_id = int(tokens[0])
                atom_type = int(tokens[1])
                x, y, z = map(float, tokens[2:5])

                atom_ids.append(atom_id)
                atom_types.append(atom_type)
                positions.append([x, y, z])

    positions_array = np.array(positions)
    return atom_ids, atom_types, positions_array
