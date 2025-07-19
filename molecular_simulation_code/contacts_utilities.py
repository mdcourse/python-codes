"""Utilities for computing particle contact matrices and neighbor lists."""

import numpy as np
from numba import njit, typed


@njit
def compute_neighbor_lists(positions: np.ndarray, cutoff: float, box: np.ndarray):
    """Compute neighbor lists directly without building full matrix."""
    N = positions.shape[0]
    cutoff_sq = cutoff ** 2

    neighbor_lists = typed.List()
    for i in range(N):
        neighbor_lists.append(typed.List.empty_list(np.int32))

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
                neighbor_lists[i].append(j)
                neighbor_lists[j].append(i)

    return neighbor_lists

