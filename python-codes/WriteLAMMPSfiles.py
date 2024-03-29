import numpy as np


def write_topology_file(dictionary,
                        filename="lammps.data",
                        velocities=None):
    """Write a LAMMPS data file containing atoms positions and velocities.

    The charge of the atoms is assumed to be 0, and the same
    molecule id is used for all atoms.
    """
    atoms_types = dictionary.atoms_type
    atoms_positions = dictionary.atoms_positions\
        * dictionary.reference_distance
    if velocities is not None:
        atoms_velocities = dictionary.atoms_velocities\
            * dictionary.reference_distance/dictionary.reference_time
    f = open(filename, "w")
    f.write('# LAMMPS data file\n\n')
    f.write("%d %s" % (dictionary.total_number_atoms, 'atoms\n'))
    f.write("%d %s" % (np.max(dictionary.atoms_type), 'atom types\n\n'))
    for LminLmax, dim in zip(dictionary.box_boundaries
                             * dictionary.reference_distance,
                             ["x", "y", "z"]):
        f.write("%.3f %.3f %s %s" % (LminLmax[0],
                                     LminLmax[1],
                                     dim+'lo',
                                     dim+'hi\n'))
    f.write('\nAtoms\n\n')
    cpt = 1
    for type, xyz in zip(atoms_types, atoms_positions):
        q, mol = 0, 1
        characters = "%d %d %d %.3f %.3f %.3f %.3f %s"
        v = [cpt, mol, type, q, xyz[0], xyz[1], xyz[2]]
        f.write(characters % (v[0], v[1], v[2], v[3], v[4], v[5], v[6], '\n'))
        cpt += 1
    if velocities is not None:
        f.write('\nVelocities\n\n')
        cpt = 1
        for vxyz in atoms_velocities:
            characters = "%d %.3f %.3f %.3f %s"
            v = [cpt, vxyz[0], vxyz[1], vxyz[2]]
            f.write(characters % (cpt, v[0], v[1], v[2], '\n'))
            cpt += 1
    f.close()
