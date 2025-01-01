from MonteCarlo import MonteCarlo
from pint import UnitRegistry
ureg = UnitRegistry()
import os
import time

# Define atom number of each group
nmb_1= 100
# Define LJ parameters (sigma)
sig_1 = 3*ureg.angstrom
# Define LJ parameters (epsilon)
eps_1 = 0.1*ureg.kcal/ureg.mol
# Define atom mass
mss_1 = 10*ureg.gram/ureg.mol
# Define box size
L = 20*ureg.angstrom
# Define a cut off
rc = 2.5*sig_1
# Pick the desired temperature
T = 300*ureg.kelvin
# choose the desired_mu
desired_mu = -3*ureg.kcal/ureg.mol

# Initialize the prepare object
mc = MonteCarlo(
    ureg = ureg,
    maximum_steps=1000,
    thermo_period=10,
    dumping_period=10,
    number_atoms=[nmb_1],
    epsilon=[eps_1], # kcal/mol
    sigma=[sig_1], # A
    atom_mass=[mss_1], # g/mol
    box_dimensions=[L, L], # A
    cut_off=rc,
    thermo_outputs="Epot-press",
    desired_temperature=T, # K
    neighbor=1,
    displace_mc = 0.25*sig_1
    # desired_mu = desired_mu,
)
mc.run()

# Test function using pytest
def test_output_files():
    assert os.path.exists("Outputs/dump.mc.lammpstrj"), \
    "Test failed: dump file was not created"
    assert os.path.exists("Outputs/simulation.log"), \
    "Test failed: log file was not created"
    print("Test passed")

# If the script is run directly, execute the tests
if __name__ == "__main__":
    import pytest
    # Run pytest programmatically
    pytest.main(["-s", __file__])
tf = time.time()
