"""
Translation of L2_distance.m (external/IsomapR1/L2_distance.m)

Original author: Roland Bunschoten, University of Amsterdam.
Computes pairwise Euclidean distances using the identity:
  ||A - B||^2 = ||A||^2 + ||B||^2 - 2 A^T B
"""

import numpy as np


def l2_distance(a: np.ndarray, b: np.ndarray, df: int = 0) -> np.ndarray:
    """Compute the pairwise Euclidean distance matrix between columns of a and b.

    Args:
        a: (D x M) array.
        b: (D x N) array.
        df: If 1, force the diagonal entries to zero. Default 0.

    Returns:
        d: (M x N) array of Euclidean distances.
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError("a and b must have the same number of rows (dimensionality).")

    # MATLAB treats 1-D vectors as (1 x N); pad a row of zeros so the
    # dot-product formula works correctly for the 1-D case.
    if a.shape[0] == 1:
        a = np.vstack([a, np.zeros((1, a.shape[1]))])
        b = np.vstack([b, np.zeros((1, b.shape[1]))])

    aa = np.sum(a * a, axis=0)   # (M,)
    bb = np.sum(b * b, axis=0)   # (N,)
    ab = a.T @ b                  # (M x N)

    # Broadcasting equivalent of MATLAB's repmat
    d = np.sqrt(np.maximum(aa[:, np.newaxis] + bb[np.newaxis, :] - 2 * ab, 0.0))

    # Clamp imaginary artefacts from floating-point noise
    d = np.real(d)

    if df == 1:
        # Zero out the diagonal (only makes sense for square matrices)
        d = d * (1 - np.eye(d.shape[0], d.shape[1]))

    return d
