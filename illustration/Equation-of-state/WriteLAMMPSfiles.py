import numpy as np
from scipy import constants as cst


def write_topology_file(dictionary,
                        filename="lammps.data",
                        velocities=None,
                        atom_style="atomic"):
    """Write a LAMMPS data file containing atoms positions and velocities.

    The charge of the atoms is assumed to be 0, and the same
    molecule id is used for all atoms.
    """
    box_boundaries = dictionary.box_boundaries\
        * dictionary.reference_distance
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
    for l0, dim in zip(box_boundaries, ["x", "y", "z"]):
        characters = "%.3f %.3f %s %s"
        f.write(characters % (l0[0], l0[1], dim+'lo', dim+'hi\n'))
    f.write('\nAtoms\n\n')
    cpt = 1
    for type, xyz in zip(atoms_types, atoms_positions):
        if atom_style == "atomic":
            characters = "%d %d %.3f %.3f %.3f %s"
            v = [cpt, type, xyz[0], xyz[1], xyz[2]]
            f.write(characters % (v[0], v[1], v[2], v[3], v[4], '\n'))
        elif atom_style == "molecular":
            q, mol = 0, cpt
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


def write_lammps_parameters(dictionary,
                            filename="PARM.lammps"):
    """Write a LAMMPS-format parameter file"""
    f = open(filename, "w")
    f.write('# LAMMPS parameter file \n\n')
    for type, mass in zip(np.unique(dictionary.atoms_type),
                          dictionary.atom_mass):
        mass *= dictionary.reference_mass
        f.write("mass "+str(type)+" "+str(mass)+"\n")
    f.write('\n')
    for type, epsilon, sigma in zip(np.unique(dictionary.atoms_type),
                                    dictionary.epsilon,
                                    dictionary.sigma):
        epsilon *= dictionary.reference_energy
        sigma *= dictionary.reference_distance
        f.write("pair_coeff " + str(type) + " " + str(type) + " " +
                str(epsilon) + " " + str(sigma) + "\n")
    f.write('\n')
    f.close()


def write_lammps_variables(self, filename="variable.lammps"):
    """Write a LAMMPS-format variable file"""
    f = open(filename, "w")
    f.write('# LAMMPS variable file \n\n')
    f.write('variable neighbor equal '
            + str(self.neighbor) + '\n')
    f.write('variable thermo equal '
            + str(self.thermo_period) + '\n')
    f.write('variable dump equal '
            + str(self.dumping_period) + '\n')
    f.write('variable cut_off equal '
            + str(self.cut_off*self.reference_distance) + '\n')
    try:
        f.write('variable displace_mc equal '
                + str(self.displace_mc*self.reference_distance) + '\n')
    except:
        pass
    try:
        f.write('variable time_step equal '
                + str(self.time_step*self.reference_time) + '\n')
    except:
        pass
    # f.write('variable minimization_steps equal '
    # + str(self.minimization_steps) + '\n')
    f.write('variable maximum_steps equal '
            + str(self.maximum_steps) + '\n')
    kB = cst.Boltzmann*cst.Avogadro/cst.calorie/cst.kilo  # kCal/mol/K
    f.write('variable temp equal '
            + str(self.desired_temperature*self.reference_energy/kB) + '\n')
    try:
        f.write('variable tau_temp equal '
                + str(self.tau_temp*self.reference_time) + '\n')
    except:
        pass
    try:
        if self.tau_press is not None:
            f.write('variable press equal '
                    + str(self.desired_pressure * self.reference_pressure) + '\n')
            f.write('variable tau_press equal '
                    + str(self.tau_press*self.reference_time) + '\n')
            f.write('variable pber equal 1')
        else:
            f.write('variable pber equal 0')
    except:
         pass
    f.write('\n')
    f.close()
