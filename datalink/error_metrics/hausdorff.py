"""
Hausdorff distance — pure NumPy.

hausdorff(P, Q) = max( dhd(P,Q), dhd(Q,P) )
dhd(P, Q)       = max_p min_q ||p - q||
"""

from __future__ import annotations

import numpy as np


def hausdorff(P: np.ndarray, Q: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Compute the Hausdorff distance between point sets P and Q.

    Parameters
    ----------
    P : (m, d) array
    Q : (n, d) array

    Returns
    -------
    hd : float
        Hausdorff distance.
    D  : (m, n) array
        Pairwise distance matrix (empty array if either set is empty).
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))

    if P.shape[1] != Q.shape[1]:
        raise ValueError("P and Q must have the same number of columns (dimensions).")

    m, n = len(P), len(Q)
    if m == 0 or n == 0:
        return 0.0, np.empty((m, n))

    # Pairwise Euclidean distance matrix
    # D[i, j] = ||P[i] - Q[j]||
    diff = P[:, np.newaxis, :] - Q[np.newaxis, :, :]  # (m, n, d)
    D = np.sqrt(np.sum(diff ** 2, axis=-1))             # (m, n)

    # Directional Hausdorff distances
    vp = float(np.max(np.min(D, axis=1)))  # max over P of min-to-Q
    vq = float(np.max(np.min(D, axis=0)))  # max over Q of min-to-P

    hd = max(vp, vq)
    return hd, D


def directed_hausdorff(P: np.ndarray, Q: np.ndarray) -> float:
    """One-directional: max_p min_q ||p - q||."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    diff = P[:, np.newaxis, :] - Q[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff ** 2, axis=-1))
    return float(np.max(np.min(D, axis=1)))
