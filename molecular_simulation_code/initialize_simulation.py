import os
import numpy as np
from prepare_simulation import Prepare
# from simulation_handler import Utilities


class InitializeSimulation(Prepare): # , Utilities):
    def __init__(self,
                cut_off, # Angstroms
                neighbor=1, # Integer - frequency for neighbor list rebuild
                thermo_period = None,
                dumping_period = None,
                thermo_outputs = None,
                data_folder="Outputs/",
                *args,
                **kwargs,
                ):
        super().__init__(*args, **kwargs)
        self.cut_off = cut_off
        self.neighbor = neighbor
        self.step = 0 # initialize simulation step
        self.thermo_period = thermo_period
        self.dumping_period = dumping_period
        self.thermo_outputs = thermo_outputs
        self.data_folder = data_folder

        self.nondimensionalize_units(["cut_off"])
