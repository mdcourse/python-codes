from scipy import constants as cst
from decimal import Decimal
import numpy as np
import copy

import warnings
warnings.filterwarnings('ignore')

from InitializeSimulation import InitializeSimulation
from Utilities import Utilities
from Outputs import Outputs

class MonteCarlo(InitializeSimulation):
    def __init__(self,
        maximum_steps,
        cut_off = 9,
        displace_mc = None,
        desired_mu = None,
        inserted_type = 0,
        desired_temperature = 300,
        neighbor = 10,
        *args,
        **kwargs,
        ):

        self.maximum_steps = maximum_steps
        self.cut_off = cut_off
        self.displace_mc = displace_mc
        self.desired_mu = desired_mu
        self.inserted_type = inserted_type
        self.desired_temperature = desired_temperature
        self.beta =  1/self.desired_temperature
        self.neighbor = neighbor
        
        super().__init__(*args, **kwargs)

        self.nondimensionalize_units_3()

    def nondimensionalize_units_3(self):
        """Use LJ prefactors to convert units into non-dimensional."""
        self.cut_off = self.cut_off/self.reference_distance
        self.desired_temperature = self.desired_temperature/self.reference_temperature
        if self.displace_mc is not None:
            self.displace_mc /= self.reference_distance
        if self.desired_mu is not None:
            self.desired_mu /= self.reference_energy

    def run(self):
        """Perform the loop over time."""
        
        for self.step in range(0, self.maximum_steps+1):
            self.update_neighbor_lists()
            self.monte_carlo_displacement()
            self.monte_carlo_insert_delete()
            self.wrap_in_box()
            #self.update_log()
            self.update_dump(filename="dump.mc.lammpstrj", velocity=False)
        self.write_data_file(filename="final.data",  velocity = False)
        #self.write_lammps_parameters()
        #self.write_lammps_variables()

    def monte_carlo_displacement(self):
        if self.displace_mc is not None:
            initial_Epot = self.calculate_LJ_potential_force(output="potential")
            initial_positions = copy.deepcopy(self.atoms_positions)
            atom_id = np.random.randint(self.total_number_atoms)
            # Test a new position
            self.atoms_positions[atom_id] += (np.random.random(3)-0.5)*self.displace_mc
            trial_Epot = self.calculate_LJ_potential_force(output="potential")
            acceptation_probability = np.min([1, np.exp(-self.beta*(trial_Epot-initial_Epot))])
            if np.random.random() <= acceptation_probability: # Accept new position
                pass
            else: # Reject new position
                self.atoms_positions = initial_positions

    def calculate_Lambda(self, mass):
        """Estimate de Broglie wavelength in LJ units."""
        m_kg = mass/cst.Avogadro*cst.milli
        kB_kCal_mol_K = cst.Boltzmann*cst.Avogadro/cst.calorie/cst.kilo
        T_K = self.desired_temperature*self.reference_energy/kB_kCal_mol_K
        Lambda = cst.h/np.sqrt(2*np.pi*cst.Boltzmann*m_kg*T_K)/cst.angstrom
        return self.nondimensionalise_units(Lambda, "distance")

    def monte_carlo_insert_delete(self):
        if self.desired_mu is not None:
            initial_Epot = self.calculate_LJ_potential_force(output="potential")
            initial_positions = copy.deepcopy(self.atoms_positions)
            initial_number_atoms = self.number_atoms
            if np.random.random() < 0.5:
                # Try adding an atom
                try_number_atoms = self.total_number_atoms + 1
                initial_number_atoms[self.inserted_type] += 1
                atom_position = np.zeros((1, self.dimensions))
                for dim in np.arange(self.dimensions):
                    atom_position[:, dim] = np.random.random(1)*np.diff(self.box_boundaries[dim]) - np.diff(self.box_boundaries[dim])/2
                self.atoms_positions = np.vstack([initial_positions, atom_position])

                trial_Epot = self.calculate_LJ_potential_force(output="potential")
                Lambda = self.calculate_Lambda(self.atom_mass[self.inserted_type])
                volume = np.prod(np.diff(self.box_boundaries))
                acceptation_probability = np.min([1, volume/(Lambda**self.dimensions*(self.number_atoms + 1))*np.exp(beta*(self.desired_mu-trial_Epot+Epot))])



            Epot = self.calculate_potential_energy(self.atoms_positions)
            trial_atoms_positions = copy.deepcopy(self.atoms_positions)
            if np.random.random() < 0.5:
                total_number_atoms = self.total_number_atoms + 1
                atom_position = np.zeros((1, self.dimensions))
                for dim in np.arange(self.dimensions):
                    atom_position[:, dim] = np.random.random(1)*np.diff(self.box_boundaries[dim]) - np.diff(self.box_boundaries[dim])/2
                trial_atoms_positions = np.vstack([trial_atoms_positions, atom_position])
                trial_Epot = self.calculate_potential_energy(trial_atoms_positions) # tocheck number_atoms = total_number_atoms)
                Lambda = self.calculate_Lambda(self.atom_mass)
                volume = np.prod(np.diff(self.box_boundaries))
                beta = 1/self.desired_temperature
                acceptation_probability = np.min([1, volume/(Lambda**self.dimensions*(self.number_atoms + 1))*np.exp(beta*(self.desired_mu-trial_Epot+Epot))])
            else:
                number_atoms = self.total_number_atoms - 1
                if number_atoms > 0:
                    atom_id = np.random.randint(self.total_number_atoms)
                    trial_atoms_positions = np.delete(trial_atoms_positions, atom_id, axis=0)
                    trial_Epot = self.calculate_potential_energy(trial_atoms_positions) # tocheck number_atoms = number_atoms)
                    Lambda = self.calculate_Lambda(self.atom_mass)
                    volume = np.prod(np.diff(self.box_boundaries))
                    beta = 1/self.desired_temperature
                    acceptation_probability = np.min([1, (Lambda**self.dimensions*(self.number_atoms)/volume)*np.exp(-beta*(self.desired_mu+trial_Epot-Epot))])
                else:
                    acceptation_probability = 0
            if np.random.random() < acceptation_probability:
                self.atoms_positions = trial_atoms_positions
                self.Epot = trial_Epot
                self.total_number_atoms = number_atoms # will have to be fixed ...
            else:
                self.Epot = Epot
