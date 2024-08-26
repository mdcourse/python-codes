#!/usr/bin/env python
# coding: utf-8

# In[3]:


import warnings
warnings.filterwarnings("ignore")

from scipy import constants as cst
from pint import UnitRegistry
ureg = UnitRegistry()
import sys
import multiprocessing
import subprocess
import numpy as np
import shutil
import os

path_to_code = "generated-codes/chapter7/"
sys.path.append(path_to_code)

from MinimizeEnergy import MinimizeEnergy
from MonteCarlo import MonteCarlo
from WriteLAMMPSfiles import write_topology_file, write_lammps_parameters, write_lammps_variables


# "The values of the potential constants for argon have been taken throughout as  
# E*/k= 119.76°K, r*=3.822 A, v*=23.79 cm3/mole, as determined by Michels6 from  
# second virial coefficient data." [1]  
# 
# [1] Wood and Parker. The Journal of Chemical Physics, 27(3):720–733, 1957.  

# In[4]:


kB = cst.Boltzmann*ureg.J/ureg.kelvin # boltzman constant
Na = cst.Avogadro/ureg.mole # avogadro
R = kB*Na # gas constant


# In[7]:


def launch_MC_code(tau):

    epsilon = (119.76*ureg.kelvin*kB*Na).to(ureg.kcal/ureg.mol) # kcal/mol
    r_star = 3.822*ureg.angstrom # angstrom
    sigma = r_star / 2**(1/6) # angstrom
    N_atom = 200 # no units
    m_argon = 39.948*ureg.gram/ureg.mol
    T = (273.15+55)*ureg.kelvin # 55°C
    volume_star = (23.79 * ureg.centimeter**3/ureg.mole).to(ureg.angstrom**3/ureg.mole)
    cut_off = sigma*2.5
    displace_mc = sigma/5 # angstrom
    volume = N_atom*volume_star*tau/Na
    box_size = volume**(1/3)
    folder = "outputs_tau"+str(tau)+"/"

    em = MinimizeEnergy(maximum_steps=100,
                        thermo_period=10,
                        dumping_period=10,
        number_atoms=[N_atom],
        epsilon=[epsilon.magnitude], 
        sigma=[sigma.magnitude],
        atom_mass=[m_argon.magnitude],
        box_dimensions=[box_size.magnitude,
                        box_size.magnitude,
                        box_size.magnitude],
        cut_off=cut_off.magnitude,
        data_folder=folder,
    )
    em.run()

    mc = MonteCarlo(maximum_steps=200000,
        dumping_period=5000,
        thermo_period=5000,
        neighbor=50,
        displace_mc = displace_mc.magnitude,
        desired_temperature = T.magnitude,
        number_atoms=[N_atom],
        epsilon=[epsilon.magnitude], 
        sigma=[sigma.magnitude],
        atom_mass=[m_argon.magnitude],
        box_dimensions=[box_size.magnitude,
                        box_size.magnitude,
                        box_size.magnitude],
        initial_positions = em.atoms_positions*em.reference_distance,
        cut_off=cut_off.magnitude,
        data_folder=folder,
    )
    mc.run()

    folder = "lammps_tau"+str(tau)+"/"
    if os.path.exists(folder) is False:
        os.mkdir(folder)

    write_topology_file(mc, filename=folder+"initial.data")
    write_lammps_parameters(mc, filename=folder+"PARM.lammps")
    write_lammps_variables(mc, filename=folder+"variable.lammps")

    mycwd = os.getcwd() # initial path
    os.chdir(folder)
    shutil.copyfile("../lammps/input.lmp", "input.lmp")
    subprocess.call(["/home/simon/Softwares/lammps-2Aug2023/src/lmp_serial", "-in", "input.lmp"])
    os.chdir(mycwd)


# In[8]:


if __name__ == "__main__":
    tau_values = np.round(np.logspace(-0.126, 0.882, 10),2)
    pool = multiprocessing.Pool()
    squared_numbers = pool.map(launch_MC_code, tau_values)
    pool.close()
    pool.join()

