from scipy import constants as cst
from decimal import Decimal
import numpy as np
import copy, sys, os

import warnings
warnings.filterwarnings('ignore')

class Outputs:
    def __init__(self,
                 thermo = None,
                 data_folder = "./",
                 dump = None,
                 *args,
                 **kwargs):
        self.thermo = thermo
        self.dump = dump
        self.data_folder = data_folder
        super().__init__(*args, **kwargs)

        if os.path.exists(self.data_folder) is False:
            os.mkdir(self.data_folder)

    def write_lammps_data(self, filename="lammps.data"):
        """Write a LAMMPS-format data file containing atoms positions and velocities"""
        f = open(filename, "w")
        f.write('# LAMMPS data file \n\n')
        f.write("%d %s"%(self.total_number_atoms, 'atoms\n')) 
        f.write("%d %s"%(np.max(self.atoms_type), 'atom types\n')) 
        f.write('\n')
        for LminLmax, dim in zip(self.box_boundaries*self.reference_distance, ["x", "y", "z"]):
            f.write("%.3f %.3f %s %s"%(LminLmax[0],LminLmax[1], dim+'lo', dim+'hi\n')) 
        f.write('\n')
        f.write('Atoms\n')
        f.write('\n')
        cpt = 1
        for type, xyz in zip(self.atoms_type,
                             self.atoms_positions*self.reference_distance):
            q = 0
            f.write("%d %d %d %.3f %.3f %.3f %.3f %s"%(cpt, 1, type, q, xyz[0], xyz[1], xyz[2], '\n')) 
            cpt += 1

    def update_dump(self, filename, velocity = True, minimization = False):
        if self.dump is not None:
            if minimization:
                dumping = self.dumping_minimize
            else:
                dumping = self.dump
            if self.step % dumping == 0:
                if self.step==0:
                    f = open(self.data_folder + filename, "w")
                else:
                    f = open(self.data_folder + filename, "a")
                f.write("ITEM: TIMESTEP\n")
                f.write(str(self.step) + "\n")
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write(str(self.total_number_atoms) + "\n")
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                for dim in np.arange(self.dimensions):
                    f.write(str(self.box_boundaries[dim][0]*self.reference_distance)
                            + " " + str(self.box_boundaries[dim][1]*self.reference_distance) + "\n")
                f.write("ITEM: ATOMS id type x y z vx vy vz\n")
                cpt = 1
                atoms_positions = copy.deepcopy(self.atoms_positions) * self.reference_distance
                if velocity:
                    atoms_velocities = copy.deepcopy(self.atoms_velocities) \
                        * self.reference_distance/self.reference_time 
                else:
                    atoms_velocities = np.zeros((self.total_number_atoms, self.dimensions))
                for type, xyz, vxyz in zip(self.atoms_type, atoms_positions, atoms_velocities):
                    f.write(str(cpt) + " " + str(type)
                            + " " + str(np.round(xyz[0],3))
                            + " " + str(np.round(xyz[1],3))
                            + " " + str(np.round(xyz[2],3))
                            + " " + str(vxyz[0])
                            + " " + str(vxyz[1])
                            + " " + str(vxyz[2])+"\n") 
                    cpt += 1
                f.close()

    def update_log(self):
        if self.thermo is not None:
            if (self.step % self.thermo == 0) | (self.step == 0):
                # convert the units
                volume_A3 = np.prod(self.box_size)*self.reference_distance**3
                epot_kcalmol = self.calculate_potential_energy(self.atoms_positions) \
                    * self.reference_energy

                if self.step == 0:         
                    print('{:<5} {:<5} {:<9} {:<13}'.format(
                        '%s' % ("step"),
                        '%s' % ("N"),
                        '%s' % ("V (A3)"),
                        '%s' % ("Ep (kcal/mol)"),
                        ))
                print('{:<5} {:<5} {:<9} {:<13}'.format(
                    '%s' % (self.step),
                    '%s' % (self.total_number_atoms),
                    '%s' % (f"{volume_A3:.3}"),
                    '%s' % (f"{epot_kcalmol:.3}"),                      
                    ))

                for output_value, filename in zip([self.total_number_atoms,
                                                    epot_kcalmol,
                                                    volume_A3],
                                                    ["atom_number.dat",
                                                    "Epot.dat",
                                                    "volume.dat"]):
                    self.write_data_file(output_value, filename)

    def write_data_file(self, output_value, filename):
        if self.step == 0:
            myfile = open(self.data_folder + filename, "w")
        else:
            myfile = open(self.data_folder + filename, "a")
        myfile.write(str(self.step) + " " + str(output_value) + "\n")
        myfile.close()