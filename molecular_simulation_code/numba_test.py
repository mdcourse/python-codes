import numpy as np
from numba import njit

@njit
def numba_copy(array):
    # Create an empty array of the same shape and type
    new_array = np.empty_like(array)
    # Copy elements manually
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            new_array[i, j] = array[i, j]
    return new_array