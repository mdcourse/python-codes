from scipy import constants as cst
import numpy as np
import copy
from Outputs import Outputs
from WriteLAMMPSfiles import write_topology_file

import warnings
warnings.filterwarnings('ignore')

class MonteCarlo(Outputs):
    def __init__(self,
                 maximum_steps,
                 cut_off = 9,
                 displace_mc = None,
                 desired_mu = None,
                 inserted_type = 0,
                 desired_temperature = 300,
                 neighbor = 10,
                 *args,
                 **kwargs):
        self.maximum_steps = maximum_steps
        self.cut_off = cut_off
        self.displace_mc = displace_mc
        self.desired_mu = desired_mu
        self.inserted_type = inserted_type
        self.desired_temperature = desired_temperature
        self.beta =  1/self.desired_temperature
        if self.desired_mu is not None:
            self.neighbor = 1
        else:
            # Rebuild neighbor lists every step in GCMC
            self.neighbor = neighbor
        super().__init__(*args, **kwargs)
        self.nondimensionalize_units_3()

    def nondimensionalize_units_3(self):
        """Use LJ prefactors to convert units into non-dimensional."""
        self.cut_off = self.cut_off/self.reference_distance
        self.desired_temperature = self.desired_temperature \
            /self.reference_temperature
        if self.displace_mc is not None:
            self.displace_mc /= self.reference_distance
        if self.desired_mu is not None:
            self.desired_mu /= self.reference_energy

    def run(self):
        """Perform the loop over time."""
        write_topology_file(self, filename="initial.data")
        for self.step in range(0, self.maximum_steps+1):
            self.update_neighbor_lists()
            self.monte_carlo_displacement()
            self.monte_carlo_insert_delete()
            self.wrap_in_box()
            self.update_log_md_mc(velocity=False)
            self.update_dump_file(filename="dump.mc.lammpstrj")
        #self.write_data_file(filename="final.data",  velocity = False)
        #self.write_lammps_parameters()
        #self.write_lammps_variables()

    def monte_carlo_displacement(self):
        if self.displace_mc is not None:
            initial_Epot = self.compute_potential(output="potential")
            initial_positions = copy.deepcopy(self.atoms_positions)
            atom_id = np.random.randint(self.total_number_atoms)
            # Test a new position
            self.atoms_positions[atom_id] += (np.random.random(3)-0.5)*self.displace_mc
            trial_Epot = self.compute_potential(output="potential")
            acceptation_probability = np.min([1, np.exp(-self.beta*(trial_Epot-initial_Epot))])
            if np.random.random() <= acceptation_probability: # Accept new position
                pass
            else: # Reject new position
                self.atoms_positions = initial_positions

    def calculate_Lambda(self, mass):
        """Estimate the de Broglie wavelength."""
        # Is it normal that mass is unitless ???
        m = mass/cst.Avogadro*cst.milli  # kg
        kB = cst.Boltzmann  # J/K
        Na = cst.Avogadro
        kB *= Na/cst.calorie/cst.kilo #  kCal/mol/K
        T = self.desired_temperature  # N
        T_K = T*self.reference_energy/kB  # K
        Lambda = cst.h/np.sqrt(2*np.pi*kB*m*T_K)  # m
        Lambda /= cst.angstrom  # Angstrom
        return Lambda / self.reference_distance # dimensionless

    def monte_carlo_insert_delete(self):
        if self.desired_mu is not None:
            initial_Epot = self.compute_potential(output="potential")
            initial_positions = copy.deepcopy(self.atoms_positions)
            initial_number_atoms = copy.deepcopy(self.number_atoms)
            volume = np.prod(np.diff(self.box_boundaries))
            if np.random.random() < 0.5:
                # Try adding an atom
                self.number_atoms[self.inserted_type] += 1
                atom_position = np.zeros((1, self.dimensions))
                for dim in np.arange(self.dimensions):
                    atom_position[:, dim] = np.random.random(1)*np.diff(self.box_boundaries[dim]) - np.diff(self.box_boundaries[dim])/2
                shift_id = 0
                for N in self.number_atoms[:self.inserted_type]:
                    shift_id += N
                self.atoms_positions = np.vstack([self.atoms_positions[:shift_id], atom_position, self.atoms_positions[shift_id:]])
                self.update_neighbor_lists()
                self.calculate_cross_coefficients()
                trial_Epot = self.compute_potential(output="potential")
                Lambda = self.calculate_Lambda(self.atom_mass[self.inserted_type])
                acceptation_probability = np.min([1, volume/(Lambda**self.dimensions*(self.total_number_atoms))*np.exp(self.beta*(self.desired_mu-trial_Epot+initial_Epot))])
            else:
                # Pick one atom to delete randomly
                atom_id = np.random.randint(self.number_atoms[self.inserted_type])
                self.number_atoms[self.inserted_type] -= 1
                if self.number_atoms[self.inserted_type] > 0:
                    shift_id = 0
                    for N in self.number_atoms[:self.inserted_type]:
                        shift_id += N
                    self.atoms_positions = np.delete(self.atoms_positions, shift_id+atom_id, axis=0)
                    self.update_neighbor_lists()
                    self.calculate_cross_coefficients()
                    trial_Epot = self.compute_potential(output="potential")
                    Lambda = self.calculate_Lambda(self.atom_mass[self.inserted_type])
                    acceptation_probability = np.min([1, (Lambda**self.dimensions*(self.total_number_atoms-1)/volume)*np.exp(-self.beta*(self.desired_mu+trial_Epot-initial_Epot))])
                else:
                    acceptation_probability = 0
            if np.random.random() < acceptation_probability:
                # Accept the new position
                pass
            else:
                # Reject the new position
                self.atoms_positions = initial_positions
                self.number_atoms = initial_number_atoms
                self.calculate_cross_coefficients()
