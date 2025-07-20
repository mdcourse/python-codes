from trajectory_dumper import update_dump_file
from simulation_logger import log_simulation_data

import numpy as np
import copy
from initialize_simulation import InitializeSimulation
from measurements_utilities import compute_epot
from monte_carlo_utilities import calculate_Lambda
from numba.typed import List as NumbaList

import warnings
warnings.filterwarnings('ignore')


class MonteCarlo(InitializeSimulation):
    def __init__(self,
                maximum_steps,
                desired_temperature,
                displace_mc = None,
                desired_mu = None,
                inserted_type = 0,
                # swap_type = [None, None],
                *args,
                **kwargs):
        self.maximum_steps = maximum_steps
        self.displace_mc = displace_mc
        self.desired_mu = desired_mu
        self.inserted_type = inserted_type
        # self.swap_type = swap_type
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
        # self.successful_swap = 0
        # self.failed_swap = 0

    def backup_state(self):
        return {
            "positions": np.copy(self.positions),
            "neighbor_lists": copy.deepcopy(self.neighbor_lists),
            "number_atoms": copy.deepcopy(self.number_atoms),
            "atom_types": np.copy(self.atom_types),
            "cross_sigmas": np.copy(self.cross_sigmas),
            "cross_epsilons": np.copy(self.cross_epsilons),
            "masses": np.copy(self.masses),
        }
        
    def restore_state(self, state):
        self.positions = state["positions"]
        self.neighbor_lists = state["neighbor_lists"]
        self.number_atoms = state["number_atoms"]
        self.atom_types = state["atom_types"]
        self.cross_sigmas = state["cross_sigmas"]
        self.cross_epsilons = state["cross_epsilons"]
        self.masses = state["masses"]

    def evaluate_trial(self, trial_Epot, initial_Epot, success_attr, fail_attr):
        """Evaluate whether to accept trial configuration based on energies."""
        beta = 1 / self.desired_temperature
        delta_E = trial_Epot - initial_Epot
        rnd = np.random.random()

        if beta * delta_E > 700:
            acc_prob = 0
        elif beta * delta_E < -700:
            acc_prob = 1
        else:
            acc_prob = min(1, np.exp(-beta * delta_E))

        if rnd <= acc_prob:
            self.Epot = trial_Epot
            setattr(self, success_attr, getattr(self, success_attr) + 1)
            return True
        else:
            setattr(self, fail_attr, getattr(self, fail_attr) + 1)
            return False

    def propose_displacement(self):
        """
        Pick a random atom and generate a random displacement vector 
        for it, according to self.displace_mc settings.
        """
        atom_id = None
        displace_mc = 0

        # Loop until we pick an atom of a type that allows displacement
        while displace_mc == 0:
            # Pick atom randomly
            atom_id = np.random.randint(np.sum(self.number_atoms))
            # Evaluate atom type
            atom_type = self.atom_types[atom_id]
            # Check desired displace_mc for this atom type
            displace_mc = self.displace_mc[atom_type - 1]

        # Generate displacement vector
        if self.box_mda[2] == 0:  # 2D
            move = (np.random.random(2) - 0.5) * displace_mc
            move = np.append(move, 0.0)
        else:  # 3D
            move = (np.random.random(3) - 0.5) * displace_mc

        return atom_id, move

    def monte_carlo_move(self):
        if self.displace_mc is None:
            return

        self.update_neighbor_lists()
        self.update_cross_coefficients()

        if not hasattr(self, 'Epot'):
            self.Epot = compute_epot(self.neighbor_lists, self.positions,
                                    self.box_mda[:3], self.cross_sigmas,
                                    self.cross_epsilons)

        # Measure the initial energy, and store the initial state
        initial_Epot = self.Epot
        state_backup = self.backup_state()

        # Do the MC move
        atom_id, move = self.propose_displacement()
        self.positions[atom_id] += move

        # Remeasure the energy potentielle
        trial_Epot = compute_epot(self.neighbor_lists, self.positions,
                                self.box_mda[:3], self.cross_sigmas,
                                self.cross_epsilons)
    
        accepted = self.evaluate_trial(trial_Epot, initial_Epot, "successful_move", "failed_move")

        if not accepted:
            self.restore_state(state_backup)
        else:
            self.Epot = trial_Epot

    def propose_delete(self):
        """Propose deleting a particle and return trial energy & acceptance probability."""
        if self.number_atoms[self.inserted_type] <= 0:
            raise RuntimeError("No more atoms to delete.")

        atom_id = np.random.randint(self.number_atoms[self.inserted_type])
        self.number_atoms[self.inserted_type] -= 1
        shift_id = sum(self.number_atoms[:self.inserted_type])

        self.positions = np.delete(self.positions, shift_id + atom_id, axis=0)

        self.update_neighbor_lists()
        # self.assign_atom_properties()
        self.update_cross_coefficients()

        trial_Epot = compute_epot(self.neighbor_lists, self.positions,
                                self.box_mda[:3], self.cross_sigmas,
                                self.cross_epsilons)
        
        Lambda = calculate_Lambda(self.desired_temperature,
                                self.masses[self.inserted_type])
        beta = 1 / self.desired_temperature
        Nat = np.sum(self.number_atoms)
        Vol = np.prod(self.box_mda[:3])

        acc_prob = min(1, (Lambda**3 * (Nat) / Vol) *
                        np.exp(-beta * (self.desired_mu[self.inserted_type]
                                        + trial_Epot - self.Epot)))
        
        print("acc_prob", acc_prob)

        return trial_Epot, acc_prob


    def propose_insert(self):
        """Propose inserting a particle and return trial energy & acceptance probability."""
        self.number_atoms[self.inserted_type] += 1

        new_atom_pos = np.random.random(3) * np.diff(self.box_bounds).T \
                    - np.diff(self.box_bounds).T / 2
        shift_id = sum(self.number_atoms[:self.inserted_type])

        self.positions = np.insert(self.positions, shift_id, new_atom_pos, axis=0)

        self.update_neighbor_lists()
        # self.assign_atom_properties()
        self.update_cross_coefficients()

        trial_Epot = compute_epot(self.neighbor_lists, self.positions,
                                self.box_mda[:3], self.cross_sigmas,
                                self.cross_epsilons)
        
        Lambda = calculate_Lambda(self.desired_temperature,
                                self.atom_mass[self.inserted_type])
        beta = 1 / self.desired_temperature
        Nat = np.sum(self.number_atoms)
        Vol = np.prod(self.box_mda[:3])

        acc_prob = min(1, Vol / (Lambda**3 * Nat) *
                        np.exp(beta * (self.desired_mu - trial_Epot + self.Epot)))
        return trial_Epot, acc_prob


    def monte_carlo_exchange(self):
        if self.desired_mu is None:
            return

        state_backup = self.backup_state()
        insert = np.random.random() < 0.5  # 50/50 choice

        if insert:
            trial_Epot, acc_prob = self.propose_insert()
            success_attr, fail_attr = "successful_insert", "failed_insert"
        else:
            trial_Epot, acc_prob = self.propose_delete()
            success_attr, fail_attr = "successful_delete", "failed_delete"

        accepted = self.evaluate_trial(trial_Epot, self.Epot, success_attr, fail_attr, acc_prob)
        if not accepted:
            self.restore_state(state_backup)
        else:
            self.Epot = trial_Epot

    def run(self):
        """Perform the loop over time."""
        for self.step in range(0, self.maximum_steps+1):
            self.monte_carlo_move()
            self.monte_carlo_exchange()
            self.wrap_in_box()
            log_simulation_data(self)
            update_dump_file(self, "dump.mc.lammpstrj")
