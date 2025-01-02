import numpy as np


def calculate_Lambda(temperature, mass):
    """Estimate the de Broglie wavelength."""
    return 1/np.sqrt(2*np.pi*mass*temperature)