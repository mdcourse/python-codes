from scipy import constants as cst
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from InitializeSimulation import InitializeSimulation

class Measurements(InitializeSimulation):
    def __init__(self,
                 min_bin = 1,
                 max_bin = 15,
                 number_bins = 40,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.min_bin = min_bin
        self.max_bin = max_bin
        self.number_bins = number_bins
        self.nondimensionalize_units_0()

    def nondimensionalize_units_0(self):
        """Use LJ prefactors to convert units into non-dimensional."""
        self.min_bin /= self.reference_distance
        self.max_bin /= self.reference_distance

    def calculate_rdf(self):
        r = np.linalg.norm(self.evaluate_rij_matrix(), axis=2)
        histogram, bin_boundaries = np.histogram(r, bins = self.number_bins, range= (self.min_bin,3))
        bin_centers = (bin_boundaries[1:]+bin_boundaries[:-1])/2
        return bin_centers, histogram

