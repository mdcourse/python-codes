import numpy as np

def define_box(box_dimensions):
    """Define simulation box: returns boundaries and box info for LAMMPS."""
    if len(box_dimensions) not in [2, 3]:
        raise ValueError("box_dimensions must have 2 (2D) or 3 (3D) elements.")

    box_dim_3d = box_dimensions + [0] * (3 - len(box_dimensions))
    box_boundaries = np.zeros((3, 2))

    for dim, length in enumerate(box_dim_3d):
        box_boundaries[dim, 0] = -length / 2
        box_boundaries[dim, 1] = length / 2

    box_mda = np.array(box_dim_3d + [90, 90, 90])
    return box_boundaries, box_mda

def populate_box(box_boundaries, n_atoms, n_types, type_fractions=None):
    """
    Populate the simulation box with random atom positions and types.
    Returns: array of shape (N, 5): [atom_id, type, x, y, z]
    """
    atoms = np.zeros((n_atoms, 5))  # id, type, x, y, z

    if type_fractions is None:
        type_fractions = [1/n_types] * n_types

    type_choices = np.random.choice(
        np.arange(1, n_types+1),
        size=n_atoms,
        p=type_fractions
    )

    for i in range(n_atoms):
        atoms[i, 0] = i+1          # atom id
        atoms[i, 1] = type_choices[i]
        for dim in range(3):
            lo, hi = box_boundaries[dim]
            atoms[i, dim+2] = np.random.uniform(lo, hi)

    return atoms

def write_lammps_data(filename, box_boundaries, masses, atoms):
    """
    Write LAMMPS data file.
    """
    with open(filename, "w") as f:
        f.write("# LAMMPS-compatible data file\n\n")

        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(masses)} atom types\n\n")

        for i, (lo, hi) in enumerate(box_boundaries):
            label = ["x", "y", "z"][i]
            f.write(f"{lo:.6f} {hi:.6f} {label}lo {label}hi\n")

        f.write("\nMasses\n\n")
        for atom_type, mass in masses.items():
            f.write(f"{atom_type} {mass:.6f}\n")

        f.write("\nAtoms\n\n")
        for atom in atoms:
            atom_id, atom_type, x, y, z = atom
            f.write(f"{int(atom_id)} {int(atom_type)} {x:.6f} {y:.6f} {z:.6f} 0 0 0\n")


def write_lammps_inc(filename, masses, pair_coeffs):
    """
    Write LAMMPS include file.
    """
    with open(filename, "w") as f:
        f.write("# LAMMPS-compatible parameter file\n\n")

        for atom_type, mass in masses.items():
            f.write(f"mass {atom_type} {mass:.6f}\n")

        f.write("\n")

        for (i, j), (epsilon, sigma) in pair_coeffs.items():
            f.write(f"pair_coeff {i} {j} {epsilon:.6f} {sigma:.4f}\n")


if __name__ == "__main__":
    # Example parameters
    box_dimensions = [20, 20, 20]      # in whatever units you want
    n_atoms = 500
    n_types = 2
    masses = {1: 15.9994, 2: 1.008}

    # Pair coefficients: {(type_i, type_j): (epsilon, sigma)}
    pair_coeffs = {
        (1, 1): (0.185199, 3.1589),
        (2, 2): (0.0, 1.0),
        (1, 2): (0.0, 1.0),
    }

    # Define box
    box_boundaries, box_mda = define_box(box_dimensions)

    # Populate box randomly
    atoms = populate_box(box_boundaries, n_atoms, n_types, type_fractions=[0.8, 0.2])

    # Write files
    write_lammps_data("topology.data", box_boundaries, masses, atoms)
    write_lammps_inc("parameters.inc", masses, pair_coeffs)
