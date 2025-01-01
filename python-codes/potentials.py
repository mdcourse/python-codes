from numba import njit


@njit
def potentials(epsilon: float, sigma: float, r: float, derivative: bool = False) -> float:
    """Compute the Lennard-Jones (LJ) potential or its derivative for a pair of atoms."""
    if derivative:  # Compute the derivative of the Lennard-Jones potential
        return 48 * epsilon * ((sigma / r) ** 12 - 0.5 * (sigma / r) ** 6) / r
    else:  # Compute the Lennard-Jones potential itself
        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
