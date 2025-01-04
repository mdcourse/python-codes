# Configuration B - Particles in the GCMC ensemble

## Parameters

nmb_1 = 50  # Define inital atom number  
sig_1 = 3 * ureg.angstrom  # Define LJ parameters (sigma)  
eps_1 = 0.1 * ureg.kcal/ureg.mol  # Define LJ parameters (epsilon)  
mss_1 = 10 * ureg.gram/ureg.mol  # Define atom mass          
L = 20 * ureg.angstrom  # Define box size  
rc = 2.5 * sig_1  # Define cut_off  
T = 300 * ureg.kelvin  # Pick the desired temperature  
mu = - 4 * ureg.kcal/ureg.mol  # Pick the desired chemical potential

## Measurements

density = 0.014 ± 0.000 g/cm3
