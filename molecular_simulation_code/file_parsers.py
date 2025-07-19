import pint
from typing import Tuple, List
import numpy as np

def read_inc_file( filepath: str, ureg: pint.UnitRegistry):
    """
    Parse a LAMMPS-style `.inc` parameter file and return dicts of mass, epsilon, and sigma.
    """
    masses_dict = {}
    epsilons_dict = {}
    sigmas_dict = {}

    with open(filepath, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens or tokens[0].startswith("#"):
                continue

            if tokens[0] == "mass":
                species, val = tokens[1], float(tokens[2])
                masses_dict[species] = val * ureg.g / ureg.mol

            elif tokens[0] == "pair_coeff":
                species1, species2 = tokens[1], tokens[2]
                epsilon, sigma = float(tokens[3]), float(tokens[4])
                epsilons_dict[(species1, species2)] = epsilon * ureg.kcal / ureg.mol
                sigmas_dict[(species1, species2)] = sigma * ureg.angstrom

    N = len(masses_dict)
    masses_array = np.zeros(N, dtype=object)
    epsilons_array = np.zeros((N, N), dtype=object)
    sigmas_array = np.zeros((N, N), dtype=object)

    for idx in range(1, N+1):
        masses_array[idx-1] = masses_dict[str(idx)]

    for i in range(1, N+1):
        for j in range(1, N+1):
            key = (str(i), str(j))
            key_sym = (str(j), str(i))
            if key in epsilons_dict:
                epsilons_array[i-1, j-1] = epsilons_dict[key]
                sigmas_array[i-1, j-1] = sigmas_dict[key]
            elif key_sym in epsilons_dict:
                epsilons_array[i-1, j-1] = epsilons_dict[key_sym]
                sigmas_array[i-1, j-1] = sigmas_dict[key_sym]

    return masses_array, epsilons_array, sigmas_array

def read_data_file(filepath: str, ureg: pint.UnitRegistry):
    """
    Read a LAMMPS-style `.data` file and return atom IDs, types, positions, and box bounds.
    """
    in_atoms_section = False
    atom_types = []
    atom_ids = []
    positions = []
    box_bounds = np.zeros((3, 2))

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            tokens = line.split()

            # Parse box boundaries
            if len(tokens) == 4 and tokens[2] in {"xlo", "ylo", "zlo"} and tokens[3] in {"xhi", "yhi", "zhi"}:
                idx = {"xlo": 0, "ylo": 1, "zlo": 2}[tokens[2]]
                box_bounds[idx, 0] = float(tokens[0])
                box_bounds[idx, 1] = float(tokens[1])
                continue

            if line.lower().startswith("atoms"):
                in_atoms_section = True
                continue

            if in_atoms_section and (
                line.lower().startswith("velocities") or 
                line.lower().startswith("bonds") or 
                line.lower().startswith("angles")
            ):
                in_atoms_section = False
                continue

            if in_atoms_section:
                atom_id = int(tokens[0])
                atom_type = int(tokens[1])
                x, y, z = map(float, tokens[2:5])

                atom_ids.append(atom_id)
                atom_types.append(atom_type)
                positions.append([x, y, z])

    box_bounds = box_bounds * ureg.angstroms
    positions_array = np.array(positions) * ureg.angstroms
    _, counts = np.unique(atom_types, return_counts=True)
    number_atoms = counts.tolist()

    return number_atoms, atom_ids, atom_types, positions_array, box_bounds

