"""Utilities for computing particle contact matrices and neighbor lists."""

from numba import njit, types
from numba.typed import List
import numpy as np

@njit
def _compute_neighbor_lists(positions: np.ndarray, cutoff: float, box: np.ndarray):
    """Compute neighbor lists as typed.List of numpy.ndarray[int64]."""
    N = positions.shape[0]
    cutoff_sq = cutoff ** 2

    # Temporary Python list to collect neighbors
    tmp_lists = [ [] for _ in range(N) ]

    for i in range(N - 1):
        for j in range(i + 1, N):
            diff = positions[i] - positions[j]
            if box[2] == 0:  # 2D
                for k in range(2):
                    diff[k] -= round(diff[k] / box[k]) * box[k]
            else:  # 3D
                for k in range(3):
                    diff[k] -= round(diff[k] / box[k]) * box[k]

            dist_sq = np.dot(diff, diff)
            if dist_sq < cutoff_sq:
                tmp_lists[i].append(j)
                tmp_lists[j].append(i)

    # Convert to typed.List of numpy arrays
    neighbor_lists = List()
    for lst in tmp_lists:
        arr = np.array(lst, dtype=np.int64)  # or int32 if you prefer
        neighbor_lists.append(arr)

    return neighbor_lists

def __compute_neighbor_lists(positions: np.ndarray, cutoff: float, box: np.ndarray):
    """Compute neighbor lists outside njit with Python lists, then convert."""
    N = positions.shape[0]
    cutoff_sq = cutoff ** 2

    tmp_lists = [[] for _ in range(N)]

    for i in range(N - 1):
        for j in range(i + 1, N):
            diff = positions[i] - positions[j]
            if box[2] == 0:
                for k in range(2):
                    diff[k] -= round(diff[k] / box[k]) * box[k]
            else:
                for k in range(3):
                    diff[k] -= round(diff[k] / box[k]) * box[k]

            dist_sq = np.dot(diff, diff)
            if dist_sq < cutoff_sq:
                tmp_lists[i].append(j)
                tmp_lists[j].append(i)

    neighbor_lists = List()
    for lst in tmp_lists:
        neighbor_lists.append(np.array(lst, dtype=np.int64))

    return neighbor_lists

@njit
def compute_neighbor_lists(positions, cutoff, box, max_neighbors):
    """
    Compute neighbor lists for all atoms with a fixed-size padded array.

    Parameters
    ----------
    positions : (N, D) float64
        Positions of atoms.
    cutoff : float64
        Cutoff distance.
    box : (3,) float64
        Box dimensions (z can be 0 for 2D).
    max_neighbors : int
        Maximum number of neighbors to store per atom.

    Returns
    -------
    neighbors : (N, max_neighbors) int64
        Neighbor indices for each atom, padded with -1.
    neighbor_counts : (N,) int64
        Number of neighbors for each atom.
    """
    N = positions.shape[0]
    cutoff_sq = cutoff * cutoff

    # Allocate arrays
    neighbors = -1 * np.ones((N, max_neighbors), dtype=np.int64)
    neighbor_counts = np.zeros(N, dtype=np.int64)

    for i in range(N - 1):
        for j in range(i + 1, N):
            diff = positions[i] - positions[j]

            # Apply minimum image convention
            if box[2] == 0.0:  # 2D
                for k in range(2):
                    diff[k] -= np.round(diff[k] / box[k]) * box[k]
            else:              # 3D
                for k in range(3):
                    diff[k] -= np.round(diff[k] / box[k]) * box[k]

            # Squared distance
            dist_sq = 0.0
            for d in range(diff.shape[0]):
                dist_sq += diff[d] * diff[d]

            if dist_sq < cutoff_sq:
                count_i = neighbor_counts[i]
                count_j = neighbor_counts[j]

                if count_i < max_neighbors:
                    neighbors[i, count_i] = j
                    neighbor_counts[i] += 1
                if count_j < max_neighbors:
                    neighbors[j, count_j] = i
                    neighbor_counts[j] += 1

    return neighbors, neighbor_counts