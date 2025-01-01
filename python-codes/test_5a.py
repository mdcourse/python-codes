from MinimizeEnergy import MinimizeEnergy
from pint import UnitRegistry
ureg = UnitRegistry()
import os
import time

# Define atom number of each group
nmb_1, nmb_2= [50, 50]
# Define LJ parameters (sigma)
sig_1, sig_2 = [3, 4]*ureg.angstrom
# Define LJ parameters (epsilon)
eps_1, eps_2 = [0.2, 0.4]*ureg.kcal/ureg.mol
# Define atom mass
mss_1, mss_2 = [10, 20]*ureg.gram/ureg.mol
# Define box size
L = 20*ureg.angstrom
# Define a cut off
rc = 2.5*sig_1

# Initialize the prepare object
minimizer = MinimizeEnergy(
    ureg = ureg,
    maximum_steps=100,
    thermo_period=1,
    dumping_period=1,
    number_atoms=[nmb_1, nmb_2],
    epsilon=[eps_1, eps_2], # kcal/mol
    sigma=[sig_1, sig_2], # A
    atom_mass=[mss_1, mss_2], # g/mol
    box_dimensions=[L, L, L], # A
    cut_off=rc,
    data_folder="Outputs/",
    thermo_outputs="Epot-MaxF",
)
minimizer.run()

# Test function using pytest
def test_output_files():
    assert os.path.exists("Outputs/dump.min.lammpstrj"), \
    "Test failed: the dump file was not created"
    assert os.path.exists("Outputs/simulation.log"), \
    "Test failed: the log file was not created"
    print("Test passed")

ti = time.time()

# If the script is run directly, execute the tests
if __name__ == "__main__":
    import pytest
    # Run pytest programmatically
    pytest.main(["-s", __file__])

tf = time.time()
# print(tf-ti, "second")