"""Utilities for computing Lennard-Jones potentials and total potential energy."""

from numba import njit


@njit
def compute_potentials(epsilon: float, sigma: float, r: float, potential_type: int) -> float:
    """Compute the potential for a pair of atoms."""    
    if potential_type == 0:  # Lennard-Jones
        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    elif potential_type == 1:  # Hard-sphere
        if r < sigma:
            return epsilon
        else:
            return 0.0
    else:
        return 0.0  # or raise if you really want

@njit
def compute_forces(epsilon: float, sigma: float, r: float, potential_type: int) -> float:
    """Compute the potential's derivative for a pair of atoms."""
    if potential_type == 0:  # Lennard-Jones
        return 48 * epsilon * ((sigma / r) ** 12 - 0.5 * (sigma / r) ** 6) / r
    elif potential_type == 1:  # Hard-sphere
        return 0.0
    else:
        return 0.0  # or raise if you want
