from scipy import constants as cst
import numpy as np
import os
from Measurements import Measurements

import warnings
warnings.filterwarnings('ignore')


class Outputs(Measurements):
    def __init__(self,
                 thermo_period=None,
                 dumping_period=None,
                 data_folder="Outputs/",
                 *args,
                 **kwargs):
        self.thermo_period = thermo_period
        self.dumping_period = dumping_period
        self.data_folder = data_folder
        super().__init__(*args, **kwargs)
        if os.path.exists(self.data_folder) is False:
            os.mkdir(self.data_folder)

    def update_log_minimize(self, Epot, maxForce):
        """Update the log file during minimization"""
        if (self.thermo_period is not None):
            if ((self.step % self.thermo_period == 0)
                    | (self.thermo_period == 0)):
                epot_kcalmol = Epot * self.reference_energy
                max_force_kcalmolA = maxForce \
                    * self.reference_energy / self.reference_distance
                if self.step == 0:
                    characters = "%s %s %s"
                    print(characters % ("step",
                                        "epot",
                                        "maxF"))
                characters = "%d %.3f %.3f"
                print(characters % (self.step,
                                    epot_kcalmol,
                                    max_force_kcalmolA))

    def update_log_md_mc(self, velocity=True):
        """Update the log file during MD or MC simulations"""
        if self.thermo_period is not None:
            if (self.step % self.thermo_period == 0) \
                    | (self.thermo_period == 0):
                # refresh values
                if velocity:
                    self.calculate_temperature()
                    self.calculate_kinetic_energy()
                    self.calculate_pressure()
                # convert the units
                if velocity:
                    temperature = self.temperature \
                        * self.reference_temperature  # K
                    pressure = self.pressure \
                        * self.reference_pressure  # Atm
                    Ekin = self.Ekin*self.reference_energy  # kcal/mol
                else:
                    temperature = self.desired_temperature \
                        * self.reference_temperature  # K
                    pressure = 0.0
                    Ekin = 0.0
                volume = np.prod(self.box_size[:3]) \
                    *self.reference_distance**3  # A3
                Epot = self.compute_potential(output="potential") \
                    * self.reference_energy  # kcal/mol
                density = self.calculate_density()  # Unitless
                density *= self.reference_mass \
                    /self.reference_distance**3  # g/mol/A3
                Na = cst.Avogadro
                density *= (cst.centi/cst.angstrom)**3 / Na  # g/cm3
                if self.step == 0:
                    characters = '{:<5} {:<5} {:<9} {:<9} {:<9} {:<13} {:<13} {:<13}'
                    print(characters.format(
                        '%s' % ("step"),
                        '%s' % ("N"),
                        '%s' % ("T (K)"),
                        '%s' % ("p (atm)"),
                        '%s' % ("V (A3)"),
                        '%s' % ("Ep (kcal/mol)"),
                        '%s' % ("Ek (kcal/mol)"),
                        '%s' % ("dens (g/cm3)"),
                        ))
                characters = '{:<5} {:<5} {:<9} {:<9} {:<9} {:<13} {:<13} {:<13}'
                print(characters.format(
                    '%s' % (self.step),
                    '%s' % (self.total_number_atoms),
                    '%s' % (f"{temperature:.3}"),
                    '%s' % (f"{pressure:.3}"),
                    '%s' % (f"{volume:.3}"),
                    '%s' % (f"{Epot:.3}"),
                    '%s' % (f"{Ekin:.3}"),
                    '%s' % (f"{density:.3}"),
                    ))
                for output_value, filename in zip([self.total_number_atoms,
                                                   Epot,
                                                   Ekin,
                                                   pressure,
                                                   temperature,
                                                   density,
                                                   volume],
                                                  ["atom_number.dat",
                                                   "Epot.dat",
                                                   "Ekin.dat",
                                                   "pressure.dat",
                                                   "temperature.dat",
                                                   "density.dat",
                                                   "volume.dat"]):
                    self.update_data_file(output_value, filename)

    def update_data_file(self, output_value, filename):
        if self.step == 0:
            f = open(self.data_folder + filename, "w")
        else:
            f = open(self.data_folder + filename, "a")
        characters = "%d %.3f %s"
        v = [self.step, output_value]
        f.write(characters % (v[0], v[1], '\n'))
        f.close()

    def update_dump_file(self, filename, velocity=False):
        """Update the dump file with the atom positions.

        If velocity is True, atom velocities are also printed in the file."""
        if self.dumping_period is not None:
            box_boundaries = self.box_boundaries\
                * self.reference_distance
            atoms_positions = self.atoms_positions\
                * self.reference_distance
            atoms_types = self.atoms_type
            if velocity:
                atoms_velocities = self.atoms_velocities \
                    * self.reference_distance/self.reference_time
            if self.step % self.dumping_period == 0:
                if self.step == 0:
                    f = open(self.data_folder + filename, "w")
                else:
                    f = open(self.data_folder + filename, "a")
                f.write("ITEM: TIMESTEP\n")
                f.write(str(self.step) + "\n")
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write(str(self.total_number_atoms) + "\n")
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                for dim in np.arange(self.dimensions):
                    f.write(str(box_boundaries[dim][0]) + " "
                            + str(box_boundaries[dim][1]) + "\n")
                cpt = 1
                if velocity:
                    f.write("ITEM: ATOMS id type x y z vx vy vz\n")
                    characters = "%d %d %.3f %.3f %.3f %.3f %.3f %.3f %s"
                    for type, xyz, vxyz in zip(atoms_types,
                                               atoms_positions,
                                               atoms_velocities):
                        v = [cpt, type, xyz[0], xyz[1], xyz[2],
                             vxyz[0], vxyz[1], vxyz[2]]
                        f.write(characters % (v[0], v[1], v[2], v[3], v[4],
                                              v[5], v[6], v[7], '\n'))
                        cpt += 1
                else:
                    f.write("ITEM: ATOMS id type x y z\n")
                    characters = "%d %d %.3f %.3f %.3f %s"
                    for type, xyz in zip(atoms_types,
                                         atoms_positions):
                        v = [cpt, type, xyz[0], xyz[1], xyz[2]]
                        f.write(characters % (v[0], v[1], v[2],
                                              v[3], v[4], '\n'))
                        cpt += 1
                f.close()
