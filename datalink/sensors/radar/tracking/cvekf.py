"""
Constant-Velocity EKF in Cartesian coordinates.

State : [x, vx, y, vy, z, vz]
Meas  : [x, y, z]  (rectangular)
"""

from __future__ import annotations
import numpy as np
from datalink.transformations.gradients import const_vel_jac
from .constvel import constvel, _observer_acc


# Measurement matrix: extract position [x, y, z] from state
_H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0],
], dtype=float)

# Default process noise: unit acceleration std-dev × tuning constant
_A_MAX  = 9.0 * 9.81        # ~88 m/s²
_K_Q    = 20_000.0
_DEFAULT_Q = np.eye(3) * _K_Q * _A_MAX


class CVEKF:
    """
    Extended Kalman Filter for constant-velocity Cartesian tracking.

    Parameters
    ----------
    state             : (6,) initial state [x,vx,y,vy,z,vz]
    state_covariance  : (6,6) initial covariance
    process_noise     : (3,3) Q matrix (acceleration noise covariance)
    measurement_noise : (3,3) R matrix
    """

    def __init__(
        self,
        state: np.ndarray,
        state_covariance: np.ndarray,
        process_noise: np.ndarray | None = None,
        measurement_noise: np.ndarray | None = None,
    ):
        self.state             = np.asarray(state,            dtype=float).copy()
        self.state_covariance  = np.asarray(state_covariance, dtype=float).copy()
        self.process_noise     = np.asarray(process_noise,    dtype=float).copy() \
                                 if process_noise is not None else _DEFAULT_Q.copy()
        self.measurement_noise = np.asarray(measurement_noise, dtype=float).copy() \
                                 if measurement_noise is not None else np.eye(3)

    # ------------------------------------------------------------------ #
    def predict(self, dt: float) -> None:
        """Propagate state and covariance forward by *dt* seconds."""
        F, G = const_vel_jac(dt)

        # State prediction (with observer maneuver subtraction)
        self.state = constvel(self.state, dt)

        # Covariance prediction
        Q_full = G @ self.process_noise @ G.T
        self.state_covariance = F @ self.state_covariance @ F.T + Q_full

    # ------------------------------------------------------------------ #
    def update(self, measurement: np.ndarray,
               measurement_noise: np.ndarray | None = None) -> np.ndarray:
        """
        Correct state with a position measurement [x, y, z].

        Returns
        -------
        innovation : ndarray, shape (3,)
        """
        z = np.asarray(measurement, dtype=float)
        R = self.measurement_noise if measurement_noise is None \
            else np.asarray(measurement_noise, dtype=float)

        innov = z - _H @ self.state
        S     = _H @ self.state_covariance @ _H.T + R
        K     = self.state_covariance @ _H.T @ np.linalg.inv(S)

        self.state            = self.state + K @ innov
        self.state_covariance = (np.eye(6) - K @ _H) @ self.state_covariance
        return innov

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        pos = self.state[[0, 2, 4]]
        return f"CVEKF(pos={pos})"


# ------------------------------------------------------------------ #
# Factory — mirrors initCVEKF.m behaviour
# ------------------------------------------------------------------ #

def init_cv_ekf(detection) -> CVEKF:
    """
    Create a CVEKF from a Detection object (rectangular frame).

    Initialization convention:
      - state covariance: large position uncertainty, 100 m²/s² velocity
      - process noise:    20000 * 9*9.81 * I₃
    """
    meas = np.asarray(detection.measurement, dtype=float)   # [x, y, z]
    R    = np.asarray(detection.measurement_noise, dtype=float)

    # Initial state: put position into [x,vx,y,vy,z,vz], velocity = 0
    state = np.zeros(6)
    state[[0, 2, 4]] = meas[:3]

    # State covariance: measurement noise for position, 100 for velocity
    P = np.eye(6) * 1e6
    P[0, 0] = R[0, 0] if R.shape == (3, 3) else 1e6
    P[2, 2] = R[1, 1] if R.shape == (3, 3) else 1e6
    P[4, 4] = R[2, 2] if R.shape == (3, 3) else 1e6
    P[1, 1] = 100.0
    P[3, 3] = 100.0
    P[5, 5] = 100.0

    Q = _DEFAULT_Q.copy()
    return CVEKF(state, P, Q, R)
