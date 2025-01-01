import numpy as np
from InitializeSimulation import InitializeSimulation
from pint import UnitRegistry
ureg = UnitRegistry()

# Define atom number of each group
nmb_1, nmb_2= [10, 10]
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
init = InitializeSimulation(
    ureg = ureg,
    number_atoms=[nmb_1, nmb_2],
    epsilon=[eps_1, eps_2], # kcal/mol
    sigma=[sig_1, sig_2], # A
    atom_mass=[mss_1, mss_2], # g/mol
    box_dimensions=[L, L, L], # A
    cut_off=rc,
)

# Test function using pytest
def test_placement():
    box_boundaries = init.box_boundaries
    atoms_positions = init.atoms_positions
    for atom_position in atoms_positions:
        for x, boundary in zip(atom_position, box_boundaries):
            assert (x >= boundary[0]) and (x <= boundary[1]), \
            f"Test failed: Atoms outside of the box at position {atom_position}"
    print("Test passed")

# If the script is run directly, execute the tests
if __name__ == "__main__":
    import pytest
    # Run pytest programmatically
    pytest.main(["-s", __file__])

