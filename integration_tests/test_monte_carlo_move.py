import sys
import os
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
        
        nmb_1= 50  # Define atom number
        sig_1 = 3 * ureg.angstrom  # Define LJ parameters (sigma)
        eps_1 = 0.1 * ureg.kcal/ureg.mol  # Define LJ parameters (epsilon)
        mss_1 = 10 * ureg.gram/ureg.mol  # Define atom mass        
        L = 20 * ureg.angstrom  # Define box size
        rc = 2.5 * sig_1  # Define cut_off
        T = 300 * ureg.kelvin  # Pick the desired temperature
        displace_mc = sig_1/4  # choose the displace_mc

        # Initialize the MonteCarlo object
        self.mc = MonteCarlo(
            ureg = ureg,
            maximum_steps=100,
            thermo_period=10,
            dumping_period=10,
            number_atoms=[nmb_1],
            epsilon=[eps_1], # kcal/mol
            sigma=[sig_1], # A
            atom_mass=[mss_1], # g/mol
            box_dimensions=[L, L, L], # A
            cut_off=rc,
            thermo_outputs="Epot-press",
            desired_temperature=T, # K
            neighbor=20,
            displace_mc = displace_mc,
        )

    def test_monte_carlo_run(self):
        """Test if the Monte Carlo simulation runs without errors."""
        try:
            # Run the Monte Carlo simulation (this should not raise an exception)
            self.mc.run()
            # If it runs successfully, assert True
            self.assertTrue(True)
        except Exception as e:
            # If any exception occurs, fail the test and print the error
            self.fail(f"Monte Carlo simulation failed with error: {e}")

if __name__ == "__main__":
    unittest.main()
