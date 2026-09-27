"""
Optimal Subpattern Assignment (OSPA) metric — pure NumPy/SciPy.

Reference: D. Schuhmacher, B.-T. Vo, B.-N. Vo, "A consistent metric for
performance evaluation of multi-object filters," IEEE T-SP 2008.

OSPA(X, Y; c, p) = ( 1/max(m,n) * [ min_pi sum_i d^c(x_i, y_pi(i))^p
                                      + c^p * |m - n| ] )^(1/p)

where d^c(a, b) = min(||a-b||, c) is the cut-off distance.

Components returned:
  ospa        — total OSPA distance
  loc         — localisation component (error for assigned pairs)
  card        — cardinality component  (penalty for unmatched objects)
  labeling    — labeling component (0 when track IDs not provided)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------------ #
# Core OSPA
# ------------------------------------------------------------------ #

def ospa(
    X: np.ndarray,
    Y: np.ndarray,
    c: float = 100.0,
    p: float = 2.0,
    x_ids: Optional[np.ndarray] = None,
    y_ids: Optional[np.ndarray] = None,
) -> tuple[float, float, float, float]:
    """
    Compute OSPA distance between two finite sets.

    Parameters
    ----------
    X      : (m, d) array  — estimated tracks, each row a state vector
    Y      : (n, d) array  — ground-truth objects, each row a state vector
    c      : cut-off distance (default 100.0 m for Cartesian)
    p      : order (default 2.0)
    x_ids  : (m,) integer track IDs for labeling component; optional
    y_ids  : (n,) integer truth IDs; optional

    Returns
    -------
    (ospa_total, loc_component, card_component, label_component)
    All values in the same units as X and Y (after cut-off).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    m, n = len(X), len(Y)

    # Edge cases: one or both sets empty
    if m == 0 and n == 0:
        return 0.0, 0.0, 0.0, 0.0
    if m == 0 or n == 0:
        total = (c ** p) ** (1.0 / p)
        return total, 0.0, total, 0.0

    # Build pairwise distance matrix (cut-off applied)
    # D[i, j] = min(||X[i] - Y[j]||, c)
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]  # (m, n, d)
    raw  = np.linalg.norm(diff, axis=-1)               # (m, n)
    D    = np.minimum(raw, c)

    # Solve linear assignment on the smaller set
    # scipy expects cost matrix of shape (min_m_n, max_m_n)
    if m <= n:
        row_ind, col_ind = linear_sum_assignment(D ** p)
        assign_cost = float(np.sum((D[row_ind, col_ind]) ** p))
    else:
        row_ind, col_ind = linear_sum_assignment((D ** p).T)
        assign_cost = float(np.sum((D[col_ind, row_ind]) ** p))

    max_mn   = max(m, n)
    card_val = c ** p * abs(m - n)

    # Labeling: count mis-labelled pairs (only when IDs provided)
    label_val = 0.0
    if x_ids is not None and y_ids is not None:
        x_ids = np.asarray(x_ids)
        y_ids = np.asarray(y_ids)
        if m <= n:
            for xi, yi in zip(row_ind, col_ind):
                if x_ids[xi] != y_ids[yi]:
                    label_val += c ** p
        else:
            for yi, xi in zip(row_ind, col_ind):
                if x_ids[xi] != y_ids[yi]:
                    label_val += c ** p

    total_p   = (assign_cost + card_val + label_val) / max_mn
    loc_p     = assign_cost / max_mn
    card_p    = card_val    / max_mn
    label_p   = label_val   / max_mn

    total = total_p ** (1.0 / p)
    loc   = loc_p   ** (1.0 / p)
    card  = card_p  ** (1.0 / p)
    label = label_p ** (1.0 / p)

    return total, loc, card, label


# ------------------------------------------------------------------ #
# Convenience wrappers for Track/Detection-object inputs
# ------------------------------------------------------------------ #

def ospa_cartesian(
    tracks: list,
    truths: list,
    c: float = 100.0,
    p: float = 2.0,
) -> tuple[float, float, float, float]:
    """
    OSPA for Cartesian-state tracks.

    tracks / truths : list of dicts or Track objects.
      - Each must expose 'state' (length-6 CV vector) or 'Position' (length-3).
    """
    def _pos(obj) -> np.ndarray:
        if hasattr(obj, 'state'):
            s = obj.state
            return np.array([s[0], s[2], s[4]])
        if isinstance(obj, dict):
            s = obj.get('state')
            if s is not None:
                return np.array([s[0], s[2], s[4]])
            p = obj.get('Position')
            if p is not None:
                return np.asarray(p, dtype=float).flatten()[:3]
        raise ValueError(f"Cannot extract position from {obj!r}")

    if not tracks or not truths:
        if not tracks and not truths:
            return 0.0, 0.0, 0.0, 0.0
        total = (c ** p) ** (1.0 / p)
        return total, 0.0, total, 0.0

    X = np.stack([_pos(t) for t in tracks])
    Y = np.stack([_pos(t) for t in truths])
    return ospa(X, Y, c=c, p=p)


def ospa_msc(
    tracks: list,
    truths: list,
    c: float = 0.1,
    p: float = 2.0,
) -> tuple[float, float, float, float]:
    """
    OSPA for MSC-state tracks (angular coordinates).

    Distance is computed on [az, el] in radians (indices 0 and 2 of MSC state).
    Default cut-off c=0.1 rad (~5.7 deg).
    """
    def _ang(obj) -> np.ndarray:
        if hasattr(obj, 'state'):
            s = obj.state
            return np.array([s[0], s[2]])
        if isinstance(obj, dict):
            s = obj.get('state')
            if s is not None:
                return np.array([s[0], s[2]])
            pos = obj.get('Position')
            if pos is not None:
                return np.asarray(pos, dtype=float)[:2]
        raise ValueError(f"Cannot extract MSC angles from {obj!r}")

    if not tracks or not truths:
        if not tracks and not truths:
            return 0.0, 0.0, 0.0, 0.0
        total = (c ** p) ** (1.0 / p)
        return total, 0.0, total, 0.0

    X = np.stack([_ang(t) for t in tracks])
    Y = np.stack([_ang(t) for t in truths])
    return ospa(X, Y, c=c, p=p)
