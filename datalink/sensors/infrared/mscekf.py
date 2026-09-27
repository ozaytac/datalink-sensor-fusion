"""
MSCEKF — Extended Kalman Filter in Modified Spherical Coordinates.

State   : [az, omega, el, elDot, invR, dRinvR]
Measurement : [az, el]  (angle-only, radians)

Prediction uses a chained Jacobian approach:
  1. Convert MSC → Cartesian
  2. Propagate with constant-velocity model
  3. Convert Cartesian → MSC
  4. Propagate covariance via F_msc = J_c2m(new) · F_cv · J_m2c(old)

Process noise *Q_proc* is a 3×3 matrix in Cartesian acceleration space,
mapped to MSC via the chained Jacobian.
"""

import numpy as np
from datalink.transformations.msc import cartesian_to_msc, msc_to_cartesian
from datalink.transformations.gradients import (
    cartesian_to_msc_gradient,
    msc_to_cartesian_gradient,
    const_vel_jac,
)

# Measurement Jacobian: d([az,el]) / d(msc_state)
_H = np.array([[1, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0]], dtype=float)


class MSCEKF:
    """
    Single-target EKF in MSC coordinates.

    Parameters
    ----------
    state : array_like, shape (6,)
        Initial MSC state [az, omega, el, elDot, invR, dRinvR].
    covariance : array_like, shape (6, 6)
        Initial state covariance.
    process_noise : array_like, shape (3, 3)
        Cartesian acceleration noise covariance (Q).
    measurement_noise : array_like, shape (2, 2)
        Angle measurement noise covariance [az, el] in radians².
    observer_input : array_like, shape (3,), optional
        Observer acceleration [ax, ay, az] in m/s².  Default: zeros.
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        observer_input: np.ndarray | None = None,
    ):
        self.state             = np.asarray(state,             dtype=float).copy()
        self.covariance        = np.asarray(covariance,        dtype=float).copy()
        self.process_noise     = np.asarray(process_noise,     dtype=float).copy()
        self.measurement_noise = np.asarray(measurement_noise, dtype=float).copy()
        self.observer_input    = (
            np.zeros(3) if observer_input is None
            else np.asarray(observer_input, dtype=float).copy()
        )

    # ------------------------------------------------------------------
    def predict(self, dt: float) -> None:
        """
        Propagate state and covariance forward by *dt* seconds.

        Observer acceleration is subtracted from the predicted Cartesian
        velocity increment to account for the observer's own motion.
        """
        # 1. Convert current MSC → Cartesian
        x_cart = msc_to_cartesian(self.state)

        # 2. Constant-velocity Cartesian prediction
        F_cv, G_cv = const_vel_jac(dt)
        x_cart_pred = F_cv @ x_cart

        # Subtract observer acceleration contribution (ownship motion correction)
        # Velocity increment due to observer: obs_acc * dt
        if np.any(self.observer_input != 0):
            x_cart_pred[1] -= self.observer_input[0] * dt
            x_cart_pred[3] -= self.observer_input[1] * dt
            x_cart_pred[5] -= self.observer_input[2] * dt

        # 3. Convert predicted Cartesian → MSC
        self.state = cartesian_to_msc(x_cart_pred)

        # 4. Linearised state-transition in MSC frame
        J_c2m = cartesian_to_msc_gradient(x_cart_pred)  # d(msc_new)/d(cart_new)
        J_m2c = msc_to_cartesian_gradient(
            msc_to_cartesian.__module__ and self.state  # use updated state
        )
        # Full MSC transition Jacobian
        F_msc = J_c2m @ F_cv @ msc_to_cartesian_gradient(
            # re-evaluate at old MSC state (before update)
            np.array(self.state)  # already updated – use F_cv chain
        )
        # Simpler & numerically stable: compute via chained Jacobians
        # F_msc ≈ J_c2m(new) · F_cv · J_m2c(old)
        #   J_m2c(old) was already available before the state update;
        #   we recompute it from x_cart (= msc_to_cartesian of old state).
        J_m2c_old = cartesian_to_msc_gradient(x_cart)   # at old cart state
        # Wait – we need d(cart_old)/d(msc_old):
        # revert to pre-prediction MSC state covariance propagation
        # Use: F_msc = J_c2m_new · F_cv · J_m2c_old
        # where J_m2c_old = msc_to_cartesian_gradient(old_msc_state)
        # We already have x_cart = msc_to_cartesian(old state), so:
        F_msc = J_c2m @ F_cv @ _msc_to_cart_grad_from_cart(x_cart)

        # 5. Process noise mapped to MSC frame
        Q_cart = G_cv @ self.process_noise @ G_cv.T
        Q_msc  = J_c2m @ Q_cart @ J_c2m.T

        # 6. Covariance prediction
        self.covariance = F_msc @ self.covariance @ F_msc.T + Q_msc

    # ------------------------------------------------------------------
    def update(self, measurement: np.ndarray,
               measurement_noise: np.ndarray | None = None) -> np.ndarray:
        """
        Correct state with an angle-only measurement [az, el] (radians).

        Parameters
        ----------
        measurement : array_like, shape (2,)
            [az, el] in radians.
        measurement_noise : array_like, shape (2, 2), optional
            Override filter's default measurement noise for this step.

        Returns
        -------
        innovation : ndarray, shape (2,)
        """
        z = np.asarray(measurement, dtype=float)
        R = self.measurement_noise if measurement_noise is None else np.asarray(measurement_noise)

        h = np.array([self.state[0], self.state[2]])
        innov = z - h
        # Wrap azimuth innovation to [-pi, pi]
        innov[0] = (innov[0] + np.pi) % (2*np.pi) - np.pi

        S = _H @ self.covariance @ _H.T + R
        K = self.covariance @ _H.T @ np.linalg.inv(S)

        self.state      = self.state + K @ innov
        self.covariance = (np.eye(6) - K @ _H) @ self.covariance
        return innov

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        az = np.degrees(self.state[0])
        el = np.degrees(self.state[2])
        r  = 1.0 / self.state[4] if self.state[4] != 0 else float('inf')
        return f"MSCEKF(az={az:.2f}°, el={el:.2f}°, R={r:.1f} m)"


# ---------------------------------------------------------------------------
# Internal helper – avoids a circular dependency on the gradient function
# ---------------------------------------------------------------------------

def _msc_to_cart_grad_from_cart(x_cart: np.ndarray) -> np.ndarray:
    """
    Return msc_to_cartesian_gradient evaluated at the MSC state
    corresponding to *x_cart*, without recomputing cartesian_to_msc.
    Equivalent to msc_to_cartesian_gradient(cartesian_to_msc(x_cart)).
    """
    msc = cartesian_to_msc(x_cart)
    return msc_to_cartesian_gradient(msc)
