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
                # desired_mu = None,
                inserted_type = 0,
                # swap_type = [None, None],
                *args,
                **kwargs):
        self.maximum_steps = maximum_steps
        self.displace_mc = displace_mc
        # self.desired_mu = desired_mu
        self.inserted_type = inserted_type
        # self.swap_type = swap_type
        self.desired_temperature = desired_temperature
        super().__init__(*args, **kwargs)
        self.nondimensionalize_units(["desired_temperature", "displace_mc"])
        self.successful_move = 0
        self.failed_move = 0
        # self.nondimensionalize_units(["desired_mu"])
        # self.successful_insert = 0
        # self.failed_insert = 0
        # self.successful_delete = 0
        # self.failed_delete = 0
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
        
        Returns:
            atom_id (int): index of chosen atom
            move (np.ndarray): displacement vector
        """
        atom_id = None
        displace_mc = 0

        # loop until we pick an atom of a type that allows displacement
        while displace_mc == 0:
            atom_id = np.random.randint(np.sum(self.number_atoms))
            atom_type = self.atom_types[atom_id]
            displace_mc = self.displace_mc[atom_type - 1]

        # generate displacement vector
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

        initial_Epot = self.Epot
        state_backup = self.backup_state()

        # Do the actual move:
        atom_id, move = self.propose_displacement()
        self.positions[atom_id] += move

        trial_Epot = compute_epot(self.neighbor_lists,
                                    self.positions,
                                    self.box_mda[:3],
                                    self.cross_sigmas,
                                    self.cross_epsilons)
        accepted = self.evaluate_trial(trial_Epot, initial_Epot, "successful_move", "failed_move")

        if not accepted:
            self.restore_state(state_backup)

    if False:
    
        def monte_carlo_move(self):
            """Monte Carlo move trial."""
            if self.displace_mc is not None: # only trigger if displace_mc was provided by the user
                # When needed, recalculate neighbor/coeff lists
                self.update_neighbor_lists()
                self.update_cross_coefficients()

                # If self.Epot does not exist yet, calculate it
                # It should only be necessary when step = 0
                if hasattr(self, 'Epot') is False:

                    self.Epot = compute_epot(self.neighbor_lists,
                                            self.positions,
                                            self.box_mda[:3],
                                            self.cross_sigmas,
                                            self.cross_epsilons)
                    
                # Make a copy of the initial atom positions and initial energy
                initial_Epot = self.Epot
                # initial_positions = copy.deepcopy(self.atoms_positions)
                # initial_positions = np.copy(self.atoms_positions)
                # initial_positions = numba_copy(self.atoms_positions)
                # Pick an atom id randomly

                displace_mc = 0
                while displace_mc == 0:
                    atom_id = np.random.randint(np.sum(self.number_atoms))
                    atom_type = self.atom_types[atom_id]
                    displace_mc = self.displace_mc[atom_type - 1]
                
                # Move the chosen atom in a random direction
                # The maximum displacement is set by self.displace_mc
                if self.box_mda[2] == 0:  # 2D case
                    move = (np.random.random(2) - 0.5) * displace_mc
                    move = np.append(move, 0.0)  # Pad with zero for the z-component
                else:  # 3D case
                    move = (np.random.random(3) - 0.5) * displace_mc
                initial_position_atom = np.copy(self.positions[atom_id])
                self.positions[atom_id] += move

                # Measure the potential energy of the new configuration
                trial_Epot = compute_epot(self.neighbor_lists,
                                            self.positions,
                                            self.box_mda[:3],
                                            self.cross_sigmas,
                                            self.cross_epsilons)

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
                    self.positions[atom_id] = initial_position_atom
                    # self.atoms_positions = initial_positions # Revert to initial positions
                    self.failed_move += 1

    def run(self):
        """Perform the loop over time."""
        for self.step in range(0, self.maximum_steps+1):
            self.monte_carlo_move()
            self.wrap_in_box()
            log_simulation_data(self)
            update_dump_file(self, "dump.mc.lammpstrj")
