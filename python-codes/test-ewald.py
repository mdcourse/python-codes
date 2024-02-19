from itertools import product

from itertools import combinations
from scipy.special import erfc # complementary error function : ercf = 1 - erf
import numpy as np

# electrostatic potential due to a point charge qi surounded by a Gaussian with net charge -qi 
# pot_short-range = qi/r - qi/r erf( sqrt(alpha)r ) = qi/r erfc( sqrt(alpha)r )
# total contribution:
# U_short-range = 0.5 * sum_{i \ne j} q_i q_j erfc( sqrt(alpha r_ij) ) / r_ij
# tofix : careful with double count ?
pot = 0
kappa = 6
for position_i, charge_i in zip(self.atoms_positions, self.atoms_charge):
    positions_j = self.atoms_positions
    charges_j = self.atoms_charge
    rij_xyz = (np.remainder(position_i - positions_j
        + self.box_size/2., self.box_size) - self.box_size/2.)
    norm_rij = np.linalg.norm(rij_xyz, axis=1)
    vij = charge_i * charges_j[norm_rij>0] \
          * erfc ( kappa * norm_rij[norm_rij>0] ) / norm_rij[norm_rij>0] # Screened Coulomb term
    pot += np.sum(vij)
print(pot)

twopi = 2.0*np.pi
twopi_sq = twopi**2

# first call

kappa = 6
nk = 8
b = 1.0 / 4.0 / kappa / kappa
k_sq_max = nk**2 
kfac = np.zeros(k_sq_max+1,dtype=np.float_)

for kx,ky,kz in product(range(nk+1),repeat=3):
    k_sq = kx**2 + ky**2 + kz**2
    if k_sq <= k_sq_max and k_sq>0: # Test to ensure within range
        kr_sq      = twopi_sq * k_sq           # k**2 in real units
        kfac[k_sq] = twopi * np.exp ( -b * kr_sq ) / kr_sq # Stored expression for later use

# other call
eikx = np.zeros((self.total_number_atoms,nk+1),  dtype=np.complex_) # omits negative k indices
eiky = np.zeros((self.total_number_atoms,2*nk+1),dtype=np.complex_) # includes negative k indices
eikz = np.zeros((self.total_number_atoms,2*nk+1),dtype=np.complex_) # includes negative k indices

# Calculate kx, ky, kz = 0, 1 explicitly
eikx[:,   0] = 1.0 + 0.0j
eiky[:,nk+0] = 1.0 + 0.0j
eikz[:,nk+0] = 1.0 + 0.0j

eikx[:,   1] = np.cos(twopi*self.atoms_positions[:,0]) + np.sin(twopi*self.atoms_positions[:,0])*1j
eiky[:,nk+1] = np.cos(twopi*self.atoms_positions[:,1]) + np.sin(twopi*self.atoms_positions[:,1])*1j
eikz[:,nk+1] = np.cos(twopi*self.atoms_positions[:,2]) + np.sin(twopi*self.atoms_positions[:,2])*1j

# Calculate remaining positive kx, ky and kz by recurrence
for k in range(2,nk+1):
    eikx[:,   k] = eikx[:,   k-1] * eikx[:,   1]
    eiky[:,nk+k] = eiky[:,nk+k-1] * eiky[:,nk+1]
    eikz[:,nk+k] = eikz[:,nk+k-1] * eikz[:,nk+1]

# Negative k values are complex conjugates of positive ones
# We do not need negative values of kx
eiky[:,0:nk] = np.conj ( eiky[:,2*nk:nk:-1] )
eikz[:,0:nk] = np.conj ( eikz[:,2*nk:nk:-1] )


pot = 0.0

for kx in range(nk+1): # Outer loop over non-negative kx

    factor = 1.0 if kx==0 else 2.0 # Accounts for skipping negative kx

    for ky,kz in product(range(-nk,nk+1),repeat=2):  # Double loop over ky, kz vector components

        k_sq = kx**2 + ky**2 + kz**2

        if k_sq <= k_sq_max and k_sq > 0: # Test to ensure within range

            term = np.sum ( self.atoms_charge * eikx[:,kx] * eiky[:,nk+ky] * eikz[:,nk+kz] ) # Sum over all ions
            pot  = pot + factor * kfac[k_sq] * np.real ( np.conj(term)*term )

# Subtract self part of k-space sum
pot = pot - kappa * np.sum ( self.atoms_charge**2 ) / np.sqrt(np.pi)

print(pot)




# Calculate remaining positive kx, ky and kz by recurrence
for k in range(2,nk+1):
    eikx[:,   k] = eikx[:,   k-1] * eikx[:,   1]
    eiky[:,nk+k] = eiky[:,nk+k-1] * eiky[:,nk+1]
    eikz[:,nk+k] = eikz[:,nk+k-1] * eikz[:,nk+1]

# Negative k values are complex conjugates of positive ones
# We do not need negative values of kx
eiky[:,0:nk] = np.conj ( eiky[:,2*nk:nk:-1] )
eikz[:,0:nk] = np.conj ( eikz[:,2*nk:nk:-1] )

pot = 0.0

for kx in range(nk+1): # Outer loop over non-negative kx

    factor = 1.0 if kx==0 else 2.0 # Accounts for skipping negative kx

    for ky,kz in product(range(-nk,nk+1),repeat=2):  # Double loop over ky, kz vector components

        k_sq = kx**2 + ky**2 + kz**2

        if k_sq <= k_sq_max and k_sq > 0: # Test to ensure within range

            term = np.sum ( q[:] * eikx[:,kx] * eiky[:,nk+ky] * eikz[:,nk+kz] ) # Sum over all ions
            pot  = pot + factor * kfac[k_sq] * np.real ( np.conj(term)*term )

# Subtract self part of k-space sum
pot = pot - kappa * np.sum ( q**2 ) / np.sqrt(np.pi)