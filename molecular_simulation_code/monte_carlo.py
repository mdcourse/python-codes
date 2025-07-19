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
            atom_id = np.random.randint(np.sum(self.number_atoms))
            # Move the chosen atom in a random direction
            # The maximum displacement is set by self.displace_mc
            if self.box_mda[2] == 0:  # 2D case
                move = (np.random.random(2) - 0.5) * self.displace_mc
                move = np.append(move, 0.0)  # Pad with zero for the z-component
            else:  # 3D case
                move = (np.random.random(3) - 0.5) * self.displace_mc
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
