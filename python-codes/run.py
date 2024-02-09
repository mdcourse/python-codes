#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from InitializeSimulation import InitializeSimulation
from Utilities import Utilities
from Outputs import Outputs
from MolecularDynamics import MolecularDynamics
from MonteCarlo import MonteCarlo


# In[ ]:


import numpy as np


# In[ ]:


self = MolecularDynamics(number_atoms=[300, 100],
                      epsilon=[1, 1],
                      sigma=[1, 5],
                      atom_mass= [1, 1],
                      Lx=30,
                      Ly=30,
                      Lz=30,
                      minimization_steps = 20,
                      maximum_steps=2000,
                      desired_temperature=300,
                      thermo = 100,
                      dump = 100,  
                      tau_temp = 100,
                      tau_press= 1000,
                      time_step=1,
                      seed=219817,
                      )
self.run()


# In[ ]:


x = MonteCarlo(number_atoms=[10, 2],
               epsilon=[1, 0.1],
               sigma=[1, 4],
               atom_mass= [1, 1],
               Lx=20,
               Ly=20,
               Lz=20,
               maximum_steps=1000,
               displace_mc = 0.5,
               seed=6987,
               dump = 10,
               thermo = 10,
               )
x.run()

