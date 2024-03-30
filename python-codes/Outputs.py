import numpy as np
import copy, os
from Measurements import Measurements

import warnings
warnings.filterwarnings('ignore')

class Outputs(Measurements):
    def __init__(self,
                 thermo_period = None,
                 dumping_period = None,
                 data_folder = "Outputs/",
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
            if ((self.step % self.thermo_period == 0) \
                | (self.thermo_period == 0)):
                epot_kcalmol = Epot * self.reference_energy
                max_force_kcalmolA = maxForce * self.reference_energy / self.reference_distance
                if self.step == 0:
                    print("%s %s %s"%("step", "epot", "maxF")) 
                print("%d %.3f %.3f"%(self.step, epot_kcalmol, max_force_kcalmolA)) 

    def update_log_md_mc(self):
        """Update the log file during MD or MC simulations"""
        if self.thermo_period is not None:
            if (self.step % self.thermo_period == 0) \
                | (self.thermo_period == 0):
                # refresh values
                self.calculate_temperature()
                self.calculate_pressure()
                self.calculate_kinetic_energy()
                # convert the units
                temperature_K = self.temperature * self.reference_temperature
                pressure_atm = self.pressure * self.reference_pressure
                volume_A3 = np.prod(self.box_size)*self.reference_distance**3
                epot_kcalmol = self.calculate_LJ_potential_force(output="potential") \
                    * self.reference_energy
                ekin_kcalmol = self.Ekin*self.reference_energy
                density_gmolA3 = self.calculate_density() * self.reference_mass/self.reference_distance**3
                density_gcm3 = density_gmolA3/6.022e23*(1e8)**3

                if self.step == 0:         
                    print('{:<5} {:<5} {:<9} {:<9} {:<9} {:<13} {:<13} {:<13}'.format(
                        '%s' % ("step"),
                        '%s' % ("N"),
                        '%s' % ("T (K)"),
                        '%s' % ("p (atm)"),
                        '%s' % ("V (A3)"),
                        '%s' % ("Ep (kcal/mol)"),
                        '%s' % ("Ek (kcal/mol)"),      
                        '%s' % ("dens (g/cm3)"),                        
                        ))
                print('{:<5} {:<5} {:<9} {:<9} {:<9} {:<13} {:<13} {:<13}'.format(
                    '%s' % (self.step),
                    '%s' % (self.total_number_atoms),
                    '%s' % (f"{temperature_K:.3}"),
                    '%s' % (f"{pressure_atm:.3}"),
                    '%s' % (f"{volume_A3:.3}"),
                    '%s' % (f"{epot_kcalmol:.3}"),
                    '%s' % (f"{ekin_kcalmol:.3}"),      
                    '%s' % (f"{density_gcm3:.3}"),                        
                    ))

                for output_value, filename in zip([self.total_number_atoms,
                                                    epot_kcalmol,
                                                    ekin_kcalmol,
                                                    pressure_atm,
                                                    temperature_K,
                                                    density_gcm3,
                                                    volume_A3],
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
            myfile = open(self.data_folder + filename, "w")
        else:
            myfile = open(self.data_folder + filename, "a")
        myfile.write(str(self.step) + " " + str(output_value) + "\n")
        myfile.close()

    def update_dump_file(self, filename, velocity = True):
        if self.dump is not None:
            if self.step % self.dump == 0:
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
