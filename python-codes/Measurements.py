import numpy as np
from InitializeSimulation import InitializeSimulation
from Utilities import Utilities

import warnings
warnings.filterwarnings('ignore')


class Measurements(InitializeSimulation, Utilities):
    def __init__(self,
                 min_bin=1,
                 max_bin=15,
                 number_bins=40,
                 *args,
                 **kwargs):
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.number_bins = number_bins
        super().__init__(*args, **kwargs)
        self.nondimensionalize_units_2()

    def nondimensionalize_units_2(self):
        """Use LJ prefactors to convert units into non-dimensional."""
        self.min_bin /= self.reference_distance
        self.max_bin /= self.reference_distance

    def calculate_histogram(self):
        r = np.linalg.norm(self.evaluate_rij_matrix(), axis=2)
        histogram, bin_boundaries = np.histogram(r,
                                                 bins=self.number_bins,
                                                 range=(self.min_bin, 3))
        bin_centers = (bin_boundaries[1:]+bin_boundaries[:-1])/2
        return bin_centers, histogram
