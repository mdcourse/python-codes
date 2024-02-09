#!/usr/bin/env python
# coding: utf-8

# In[1]:


from InitializeSimulation import InitializeSimulation
from Utilities import Utilities
from Outputs import Outputs
from MolecularDynamics import MolecularDynamics
from MonteCarlo import MonteCarlo


# In[2]:


import numpy as np


# In[3]:


self = MolecularDynamics(number_atoms=[40, 10],
                      epsilon=[1, 0.4],
                      sigma=[1, 3],
                      atom_mass= [1, 1],
                      Lx=20,
                      Ly=20,
                      Lz=20,
                      maximum_steps=2000,
                      desired_temperature=300,
                      thermo = 100,
                      dump = 100,  
                      tau_temp = 100,
                      time_step=1,
                      seed=519817,
                      )
self.run()


# In[ ]:





# In[ ]:


if 2 < 1:
    x = MonteCarlo(number_atoms=[10, 5],
                Lx=40,
                Ly=40,
                Lz=15,
                maximum_steps=10,
                displace_mc = 0.5,
                seed=6987,
                dump = 10,
                thermo = 10,
                )
    x.run()

