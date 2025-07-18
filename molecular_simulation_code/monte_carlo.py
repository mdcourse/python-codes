from trajectory_dumper import update_dump_file
from simulation_logger import log_simulation_data

import numpy as np
import copy
from initialize_simulation import InitializeSimulation
from measurements_utilities import compute_epot
from monte_carlo_utilities import calculate_Lambda

import warnings
warnings.filterwarnings('ignore')


class MonteCarlo(InitializeSimulation):
    def __init__(self,
                maximum_steps,
                desired_temperature,
                displace_mc = None,
                desired_mu = None,
                inserted_type = 0,
                swap_type = [None, None],
                *args,
                **kwargs):
        self.maximum_steps = maximum_steps
        self.displace_mc = displace_mc
        self.desired_mu = desired_mu
        self.inserted_type = inserted_type
        self.swap_type = swap_type
        self.desired_temperature = desired_temperature
        super().__init__(*args, **kwargs)
        self.nondimensionalize_units(["desired_temperature", "displace_mc"])
        self.successful_move = 0
        self.failed_move = 0
        self.nondimensionalize_units(["desired_mu"])
        self.successful_insert = 0
        self.failed_insert = 0
        self.successful_delete = 0
        self.failed_delete = 0
        self.successful_swap = 0
        self.failed_swap = 0


    def monte_carlo_move(self):
        """Monte Carlo move trial."""
        if self.displace_mc is not None: # only trigger if displace_mc was provided by the user
            # When needed, recalculate neighbor/coeff lists
            self.update_neighbor_lists()
            self.update_cross_coefficients()
            # If self.Epot does not exist yet, calculate it
            # It should only be necessary when step = 0
            if hasattr(self, 'Epot') is False:
                self.Epot = compute_epot(self.neighbor_lists, self.atoms_positions, self.box_mda, self.cross_coefficients)
            # Make a copy of the initial atom positions and initial energy
            initial_Epot = self.Epot
            # initial_positions = copy.deepcopy(self.atoms_positions)
            # initial_positions = np.copy(self.atoms_positions)
            # initial_positions = numba_copy(self.atoms_positions)
            # Pick an atom id randomly
            atom_id = np.random.randint(np.sum(self.number_atoms))
            # Move the chosen atom in a random direction
            # The maximum displacement is set by self.displace_mc
            if self.box_mda[2] == 0:  # 2D case
                move = (np.random.random(2) - 0.5) * self.displace_mc
                move = np.append(move, 0.0)  # Pad with zero for the z-component
            else:  # 3D case
                move = (np.random.random(3) - 0.5) * self.displace_mc
            initial_position_atom = np.copy(self.atoms_positions[atom_id])
            self.atoms_positions[atom_id] += move

            # Measure the potential energy of the new configuration
            trial_Epot = compute_epot(self.neighbor_lists, self.atoms_positions, self.box_mda, self.cross_coefficients)

            # Evaluate whether the new configuration should be kept or not
            beta =  1/self.desired_temperature
            delta_E = trial_Epot-initial_Epot
            random_number = np.random.random() # random number between 0 and 1

            if beta * delta_E > 700:
                acceptation_probability = 0  # exp(-700) is effectively 0.
            elif beta * delta_E < -700:  # Avoid overflow for large negative exponents
                acceptation_probability = 1  # exp(-(-700)) is effectively infinite
            else:
                acceptation_probability = np.min([1, np.exp(-beta * delta_E)])

            if random_number <= acceptation_probability: # Accept new position
                self.Epot = trial_Epot
                self.successful_move += 1
            else: # Reject new position
                self.atoms_positions[atom_id] = initial_position_atom
                # self.atoms_positions = initial_positions # Revert to initial positions
                self.failed_move += 1

    def monte_carlo_swap(self):
        if self.swap_type[0] is not None:
            self.update_neighbor_lists()
            self.update_cross_coefficients()
            if hasattr(self, 'Epot') is False:
                self.Epot = compute_epot(self.neighbor_lists, self.atoms_positions, self.box_mda, self.cross_coefficients)
            initial_Epot = self.Epot
            initial_positions = copy.deepcopy(self.atoms_positions)
            # Pick an atom of type one randomly
            atom_id_1 = np.random.randint(self.number_atoms[self.swap_type[0]])
            # Pick an atom of type two randomly
            atom_id_2 = np.random.randint(self.number_atoms[self.swap_type[1]])
            shift_1 = 0
            for N in self.number_atoms[:self.swap_type[0]]:
                shift_1 += N
            shift_2 = 0
            for N in self.number_atoms[:self.swap_type[1]]:
                shift_2 += N
            # attempt to swap the position of the atoms
            position1 = copy.deepcopy(self.atoms_positions[shift_1+atom_id_1])
            position2 = copy.deepcopy(self.atoms_positions[shift_2+atom_id_2])
            self.atoms_positions[shift_2+atom_id_2] = position1
            self.atoms_positions[shift_1+atom_id_1] = position2
            # force the recalculation of neighbor list
            initial_atoms_sigma = self.atoms_sigma
            initial_atoms_epsilon = self.atoms_epsilon
            initial_atoms_mass = self.atoms_mass
            initial_atoms_type = self.atoms_type
            initial_cross_coefficients = self.cross_coefficients
            initial_neighbor_lists = self.neighbor_lists
            self.update_neighbor_lists(force_update=True)
            self.assign_atom_properties()
            self.update_cross_coefficients(force_update=True)
            # Measure the potential energy of the new configuration
            trial_Epot = compute_epot(self.neighbor_lists, self.atoms_positions, self.box_mda, self.cross_coefficients)
            # Evaluate whether the new configuration should be kept or not
            beta =  1/self.desired_temperature
            delta_E = trial_Epot-initial_Epot
            random_number = np.random.random() # random number between 0 and 1

            if beta * delta_E > 700:
                acceptation_probability = 0  # exp(-700) is effectively 0.
            elif beta * delta_E < -700:  # Avoid overflow for large negative exponents
                acceptation_probability = 1  # exp(-(-700)) is effectively infinite
            else:
                acceptation_probability = np.min([1, np.exp(-beta * delta_E)])

            if random_number <= acceptation_probability: # Accept new position
                self.Epot = trial_Epot
                self.successful_swap += 1
            else: # Reject new position
                self.atoms_positions = initial_positions # Revert to initial positions
                self.failed_swap += 1
                self.atoms_sigma = initial_atoms_sigma
                self.atoms_epsilon = initial_atoms_epsilon
                self.atoms_mass = initial_atoms_mass
                self.atoms_type = initial_atoms_type
                self.cross_coefficients = initial_cross_coefficients
                self.neighbor_lists = initial_neighbor_lists

    def monte_carlo_exchange(self):
        if self.desired_mu is not None:
            # The first step is to make a copy of the previous state
            # Since atoms numbers are evolving, its also important to store the
            # neighbor, sigma, and epsilon lists
            self.Epot = compute_epot(self.neighbor_lists, self.atoms_positions, self.box_mda, self.cross_coefficients)# TOFIX: not necessary every time
            initial_positions = copy.deepcopy(self.atoms_positions)
            initial_number_atoms = copy.deepcopy(self.number_atoms)
            initial_neighbor_lists = copy.deepcopy(self.neighbor_lists)
            initial_cross_coefficients = copy.deepcopy(self.cross_coefficients)
            # Apply a 50-50 probability to insert or delete
            insert_or_delete = np.random.random()
            if np.random.random() < insert_or_delete:
                self.monte_carlo_insert()
            else:
                self.monte_carlo_delete()
            if np.random.random() < self.acceptation_probability: # accepted move
                # Update the success counters
                if np.random.random() < insert_or_delete:
                    self.successful_insert += 1
                else:
                    self.successful_delete += 1
            else:
                # Reject the new position, revert to inital position
                self.neighbor_lists = initial_neighbor_lists
                self.cross_coefficients = initial_cross_coefficients
                self.atoms_positions = initial_positions
                self.number_atoms = initial_number_atoms
                # Update the failed counters
                if np.random.random() < insert_or_delete:
                    self.failed_insert += 1
                else:
                    self.failed_delete += 1

    def monte_carlo_insert(self):
        self.number_atoms[self.inserted_type] += 1
        new_atom_position = np.random.random(3)*np.diff(self.box_boundaries).T \
            - np.diff(self.box_boundaries).T/2
        shift_id = 0 
        for N in self.number_atoms[:self.inserted_type]:
            shift_id += N
        self.atoms_positions = np.vstack([self.atoms_positions[:shift_id],
                                        new_atom_position,
                                        self.atoms_positions[shift_id:]])
        self.update_neighbor_lists()
        self.assign_atom_properties()
        self.update_cross_coefficients()
        trial_Epot = compute_epot(self.neighbor_lists, self.atoms_positions, self.box_mda, self.cross_coefficients)
        Lambda = calculate_Lambda(self.desired_temperature, self.atom_mass[self.inserted_type])
        beta =  1/self.desired_temperature
        Nat = np.sum(self.number_atoms) # Number atoms, should it really be N? of N (type) ?
        Vol = np.prod(self.box_mda[:3]) # box volume
        # dimension of 3 is enforced in the power of the Lambda
        self.acceptation_probability = np.min([1, Vol/(Lambda**3*Nat) \
            *np.exp(beta*(self.desired_mu-trial_Epot+self.Epot))])

    def monte_carlo_delete(self):
        # Pick one atom to delete randomly
        atom_id = np.random.randint(self.number_atoms[self.inserted_type])
        self.number_atoms[self.inserted_type] -= 1
        if self.number_atoms[self.inserted_type] > 0:
            shift_id = 0
            for N in self.number_atoms[:self.inserted_type]:
                shift_id += N
            self.atoms_positions = np.delete(self.atoms_positions, shift_id+atom_id, axis=0)
            self.update_neighbor_lists()
            self.assign_atom_properties()
            self.update_cross_coefficients()
            trial_Epot = compute_epot(self.neighbor_lists, self.atoms_positions, self.box_mda, self.cross_coefficients)
            Lambda = calculate_Lambda(self.desired_temperature, self.atom_mass[self.inserted_type])
            beta =  1/self.desired_temperature
            Nat = np.sum(self.number_atoms) # Number atoms, should it really be N? of N (type) ?
            Vol = np.prod(self.box_mda[:3]) # box volume
            # dimension of 3 is enforced in the power of the Lambda
            self.acceptation_probability = np.min([1, (Lambda**3 *(Nat-1)/Vol) \
                *np.exp(-beta*(self.desired_mu+trial_Epot-self.Epot))])
        else:
            print("Error: no more atoms to delete")

    def run(self):
        """Perform the loop over time."""
        for self.step in range(0, self.maximum_steps+1):
            self.monte_carlo_move()
            self.monte_carlo_swap()
            self.wrap_in_box()
            self.monte_carlo_exchange()
            log_simulation_data(self)
            update_dump_file(self, "dump.mc.lammpstrj")
