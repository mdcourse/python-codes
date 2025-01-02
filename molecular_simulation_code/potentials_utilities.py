"""Utilities for computing Lennard-Jones potentials and total potential energy."""

from numba import njit


@njit
def compute_potentials(epsilon: float, sigma: float, r: float, potential_type: str) -> float:
    """Compute the potential for a pair of atoms."""
    if potential_type == "LJ":  # Compute the Lennard-Jones potential
        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    elif potential_type == "HS":  # Compute the hard-sphere potential
        if r < sigma:
            return epsilon  # use epsilon instead of infinite
        else:
            return 0.0  # Zero potential otherwise
    else:
        raise ValueError(f"Invalid potential_type: {potential_type}")

@njit
def compute_forces(epsilon: float, sigma: float, r: float, potential_type: str) -> float:
    """Compute the potential' derivative for a pair of atoms."""
    if potential_type == "LJ":  # Compute the derivative of the Lennard-Jones potential
        return 48 * epsilon * ((sigma / r) ** 12 - 0.5 * (sigma / r) ** 6) / r
    elif potential_type == "HS":
        return 0  # ill define
