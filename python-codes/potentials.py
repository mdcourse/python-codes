from numba import njit

@njit
def potentials(epsilon, sigma, r, derivative=False):
    """ LJ potential for interaction between a pair of neutral atoms"""
    if derivative: # compute the derivative of the Lennard-Jones potential
        return 48 * epsilon * ((sigma / r) ** 12 - 0.5 * (sigma / r) ** 6) / r
    else: # calculate the Lennard-Jones potential itself
        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
