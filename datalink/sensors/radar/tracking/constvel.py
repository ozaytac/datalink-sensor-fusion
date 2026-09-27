"""
Constant-velocity Cartesian state transition.

Includes ownship-motion correction (subtracted observer maneuver) via a
module-level observer-acceleration variable set by the radar simulation.
"""

import numpy as np
from datalink.transformations.gradients import const_vel_jac


# Module-level ownship acceleration, set by the radar simulation each step
_observer_acc: np.ndarray = np.zeros(3)


def set_observer_acceleration(acc: np.ndarray) -> None:
    """Set the current observer (ownship) acceleration [ax, ay, az] m/s²."""
    global _observer_acc
    _observer_acc = np.asarray(acc, dtype=float).ravel()


def get_observer_acceleration() -> np.ndarray:
    return _observer_acc.copy()


def constvel(state: np.ndarray, dt: float = 1.0,
             noise: np.ndarray | None = None) -> np.ndarray:
    """
    Propagate state [x,vx,y,vy,z,vz] by *dt* seconds under constant velocity.

    Subtracts the observer acceleration maneuver (MPAR_ownship_acc correction).

    Parameters
    ----------
    state : ndarray, shape (6,) or (6, N)
    dt    : float, time step [s]
    noise : ndarray, shape (3,) or (3, N), optional — acceleration noise

    Returns
    -------
    state_new : ndarray, same shape as *state*
    """
    s = np.asarray(state, dtype=float)
    scalar = s.ndim == 1
    if scalar:
        s = s[:, np.newaxis]

    F, G = const_vel_jac(dt)
    n_cols = s.shape[1]

    if noise is None:
        w = np.zeros((3, n_cols))
    else:
        w = np.asarray(noise, dtype=float)
        if w.ndim == 1:
            w = np.tile(w[:, np.newaxis], (1, n_cols))

    s_new = F @ s + G @ w

    # Subtract observer maneuver (ownship acceleration correction)
    obs = _observer_acc  # (3,)
    B = np.array([[dt**2 / 2, 0, 0],
                  [dt,        0, 0],
                  [0, dt**2 / 2, 0],
                  [0, dt,        0],
                  [0, 0, dt**2 / 2],
                  [0, 0, dt       ]])
    obs_maneuver = B @ obs  # (6,)
    s_new -= obs_maneuver[:, np.newaxis]

    return s_new[:, 0] if scalar else s_new
