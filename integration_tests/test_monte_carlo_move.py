import sys
import os
import time
import pstats
import cProfile
import unittest
import numpy as np
from pint import UnitRegistry

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../molecular_simulation_code")))

from monte_carlo import MonteCarlo


class TestMonteCarloSimulation(unittest.TestCase):
    """Global test for the MonteCarlo simulation"""

    def setUp(self):
        """Setup for the MonteCarlo test."""

        ureg = UnitRegistry()
        
        # Configuration A
        nmb_1= 50  # Define atom number
        sig_1 = 3 * ureg.angstrom  # Define LJ parameters (sigma)
        eps_1 = 0.1 * ureg.kcal/ureg.mol  # Define LJ parameters (epsilon)
        mss_1 = 10 * ureg.gram/ureg.mol  # Define atom mass        
        L = 14 * ureg.angstrom  # Define box size
        rc = 2.5 * sig_1  # Define cut_off
        T = 300 * ureg.kelvin  # Pick the desired temperature
        displace_mc = sig_1/4  # choose the displace_mc

        # Initialize the MonteCarlo object
        self.mc = MonteCarlo(
            ureg = ureg,
            maximum_steps = 5000,
            thermo_period = 100,
            dumping_period = 100,
            number_atoms = [nmb_1],
            epsilon = [eps_1],
            sigma = [sig_1],
            atom_mass = [mss_1],
            box_dimensions = [L, L, L],
            cut_off = rc,
            thermo_outputs = "Epot-press",
            desired_temperature = T,
            neighbor = 50,
            displace_mc = displace_mc,
        )

    def test_monte_carlo_run(self):
        """Test if the Monte Carlo simulation runs without errors."""
        profiler = cProfile.Profile()
        try:
            # Start profiling before running the Monte Carlo simulation
            profiler.enable()

            # Run the Monte Carlo simulation
            ti = time.time()
            self.mc.run()  # Assuming self.mc is your Monte Carlo simulation object

            # Stop profiling after the run
            profiler.disable()

            tf = time.time()
            print("Duration:", np.round(tf - ti, 2), "s")

            # Convert the profiler stats into a readable format
            stats = pstats.Stats(profiler)
            stats.strip_dirs()  # Remove extraneous directory information
            stats.sort_stats('time')  # Sort by time spent in function
            stats.print_stats(10)  # Print top 10 slowest functions

            self.assertTrue(True)
        except Exception as e:
            # If any exception occurs, fail the test and print the error
            self.fail(f"Monte Carlo simulation failed with error: {e}")


if __name__ == "__main__":
    unittest.main()
