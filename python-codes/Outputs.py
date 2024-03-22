from scipy import constants as cst
from decimal import Decimal
import numpy as np
import copy, sys, os

import warnings
warnings.filterwarnings('ignore')

class Outputs:
    def __init__(self,
                 thermo = None,
                 dump = None,
                 thermo_minimize = 25,
                 dumping_minimize = 10,
                 data_folder = "./",
                 *args,
                 **kwargs):
        self.thermo_minimize = thermo_minimize
        self.dumping_minimize = dumping_minimize
        self.thermo = thermo
        self.dump = dump
        self.data_folder = data_folder
        super().__init__(*args, **kwargs)

        if os.path.exists(self.data_folder) is False:
            os.mkdir(self.data_folder)

    def update_log(self, minimization = False):
        if minimization:
            if self.thermo_minimize is not None:
                if (self.step % self.thermo_minimize == 0) | (self.step == 0):
                    epot_kcalmol = self.calculate_potential_energy(self.atoms_positions) \
                        * self.reference_energy
                    max_force_kcalmolA = self.max_forces * self.reference_energy / self.reference_distance
                    if self.step == 0:
                        print("%s %s %s"%("step", "epot", "maxF")) 
                    print("%d %.3f %.3f"%(self.step, epot_kcalmol, max_force_kcalmolA)) 
        else:
            if self.thermo is not None:
                if (self.step % self.thermo == 0) | (self.step == 0):
                    # refresh values
                    self.calculate_temperature()
                    self.calculate_pressure()
                    self.calculate_kinetic_energy()
                    # convert the units
                    temperature_K = self.temperature * self.reference_temperature
                    pressure_atm = self.pressure * self.reference_pressure
                    volume_A3 = np.prod(self.box_size)*self.reference_distance**3
                    epot_kcalmol = self.calculate_potential_energy(self.atoms_positions) \
                        * self.reference_energy
                    ekin_kcalmol = self.Ekin*self.reference_energy
                    density_gmolA3 = self.evaluate_density() * self.reference_mass/self.reference_distance**3
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
                                                       density_gcm3],
                                                      ["atom_number.dat",
                                                       "Epot.dat",
                                                       "Ekin.dat",
                                                       "pressure.dat",
                                                       "temperature.dat",
                                                       "density.dat"]):
                        self.write_data_file(output_value, filename)

    def write_data_file(self, output_value, filename):
        if self.step == 0:
            myfile = open(self.data_folder + filename, "w")
        else:
            myfile = open(self.data_folder + filename, "a")
        myfile.write(str(self.step) + " " + str(output_value) + "\n")
        myfile.close()

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
        for type, q, xyz in zip(self.atoms_type, self.atoms_charge,
                             self.atoms_positions*self.reference_distance):
            f.write("%d %d %d %.3f %.3f %.3f %.3f %s"%(cpt, 1, type, q, xyz[0], xyz[1], xyz[2], '\n')) 
            cpt += 1
        f.write('\n')
        f.write('Velocities\n')
        f.write('\n')
        cpt = 1
        for vxyz in self.atoms_velocities*self.reference_distance/self.reference_time:
            f.write("%d %.3f %.3f %.3f %s"%(cpt, vxyz[0], vxyz[1], vxyz[2], '\n')) 
            cpt += 1
        f.close()

    def write_lammps_parameters(self, filename="PARM.lammps"):
        """Write a LAMMPS-format parameters file"""
        f = open(filename, "w")
        f.write('# LAMMPS parameter file \n\n')
        for type, mass in zip(np.unique(self.atoms_type), self.atom_mass):
            mass *= self.reference_mass
            f.write("mass "+str(type)+" "+str(mass)+"\n")  
        f.write('\n')   
        for type, epsilon, sigma in zip(np.unique(self.atoms_type),
                                        self.epsilon,
                                        self.sigma):
            epsilon *= self.reference_energy
            sigma *= self.reference_distance
            f.write("pair_coeff "+str(type)+" "+str(type)+" "+
                    str(epsilon) + " " + str(sigma) + "\n")
        f.write('\n')
        f.close()

    def write_lammps_variables(self, filename="variable.lammps"):
        """Write a LAMMPS-format variable file"""
        f = open(filename, "w")
        f.write('# LAMMPS variable file \n\n')
        f.write('variable thermo equal ' + str(self.thermo) + '\n')
        f.write('variable dump equal ' + str(self.dump) + '\n')
        f.write('variable thermo_minimize equal ' + str(self.thermo_minimize) + '\n')
        f.write('variable dumping_minimize equal ' + str(self.dumping_minimize) + '\n')
        f.write('variable time_step equal ' + str(self.time_step*self.reference_time) + '\n')
        f.write('variable minimization_steps equal ' + str(self.minimization_steps) + '\n')
        f.write('variable maximum_steps equal ' + str(self.maximum_steps) + '\n')
        kB = cst.Boltzmann*cst.Avogadro/cst.calorie/cst.kilo # kCal/mol/K
        f.write('variable temp equal ' + str(self.temperature*self.reference_energy/kB) + '\n')
        f.write('variable tau_temp equal ' + str(self.tau_temp*self.reference_time) + '\n')
        f.write('\n')
        f.close()
