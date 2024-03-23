from scipy import constants as cst
from decimal import Decimal
import numpy as np
import copy, sys, os

import warnings
warnings.filterwarnings('ignore')

class Outputs:
    def __init__(self,
                 data_folder = "./",
                 *args,
                 **kwargs):
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