"""
Filter factory functions for the IR (infrared) tracking filters.

Detection dict schema
----------------------------------------------------------
{
    "measurement"       : [az_deg, el_deg],   # degrees
    "measurement_noise" : 2×2 array,          # noise in degrees²
    "orientation"       : 3×3 array,          # sensor→scenario rotation (default eye(3))
    "sensor_velocity"   : [vx, vy, vz],       # observer velocity  (default zeros)
}
"""

import numpy as np
from .mscekf import MSCEKF
from datalink.transformations.msc import cartesian_to_msc, msc_to_cartesian
from datalink.transformations.gradients import cartesian_to_msc_gradient


_DEFAULT_RANGE       = 3e4   # metres
_DEFAULT_RANGE_SIGMA = 2e3   # metres


def init_cv_msc_ekf(
    detection: dict,
    range_estimation: tuple | None = None,
) -> MSCEKF:
    """
    Initialise a constant-velocity MSC-EKF from an angle-only detection.

    Parameters
    ----------
    detection : dict
        Keys: 'measurement' ([az_deg, el_deg]),
              'measurement_noise' (2×2, degrees²),
              'orientation' (3×3, default eye(3)),
              'sensor_velocity' (3-vector, default zeros).
    range_estimation : (range_m, range_sigma_m) or None
        Initial range and its std-dev.  Default: (3e4, 2e3).

    Returns
    -------
    MSCEKF
    """
    meas   = np.asarray(detection["measurement"], dtype=float)
    R_meas = np.asarray(detection.get("measurement_noise", np.eye(2)), dtype=float)
    orient = np.asarray(detection.get("orientation", np.eye(3)), dtype=float)
    sensor_vel = np.asarray(detection.get("sensor_velocity", [0, 0, 0]), dtype=float)

    az_rad = np.deg2rad(meas[0])
    el_rad = np.deg2rad(meas[1]) if len(meas) > 1 else 0.0

    if range_estimation is not None:
        rng, sigma_rng = float(range_estimation[0]), float(range_estimation[1])
    else:
        rng, sigma_rng = _DEFAULT_RANGE, _DEFAULT_RANGE_SIGMA

    # ---- Build initial MSC state in sensor frame -------------------------
    msc_state = np.zeros(6)
    msc_state[0] = az_rad        # az
    msc_state[2] = el_rad        # el  (sign convention: stored positive here)
    msc_state[4] = 1.0 / rng     # invR

    # ---- Rotate MSC state to scenario frame ------------------------------
    # Simple rotation: rotate position part via orientation matrix
    # (full rotateMSCState is non-trivial; we apply a first-order rotation)
    msc_state = _rotate_msc_state(msc_state, orient)

    # ---- Build Cartesian state and assign observer-relative velocity ------
    cart_state = msc_to_cartesian(msc_state)
    cart_state[1] = -sensor_vel[0]
    cart_state[3] = -sensor_vel[1]
    cart_state[5] = -sensor_vel[2]
    msc_state = cartesian_to_msc(cart_state)

    # ---- Initial covariance in MSC frame ---------------------------------
    # Build Cartesian covariance (LOS frame), then transform to MSC
    rot = _ypr_to_rotmat(az_rad, -el_rad, 0.0).T   # pitch = -el

    az_sigma  = np.deg2rad(np.sqrt(R_meas[0, 0]))
    el_sigma  = np.deg2rad(np.sqrt(R_meas[1, 1])) if R_meas.shape[0] > 1 else np.deg2rad(52.0)

    los_pos_cov = np.diag([
        sigma_rng**2,
        (rng * np.cos(el_rad) * az_sigma)**2,
        (rng * el_sigma)**2,
    ])
    los_vel_cov = 100.0 * np.eye(3)

    cart_cov = np.zeros((6, 6))
    cart_cov[0::2, 0::2] = orient @ rot @ los_pos_cov @ rot.T @ orient.T
    cart_cov[1::2, 1::2] = orient @ rot @ los_vel_cov @ rot.T @ orient.T

    H = cartesian_to_msc_gradient(cart_state)
    msc_cov = H @ cart_cov @ H.T

    # ---- Process noise Q = I₃ (unit acceleration std-dev) ---------------
    Q = np.eye(3)

    # ---- Measurement noise in radians² -----------------------------------
    R_rad = np.deg2rad(1.0)**2 * np.eye(R_meas.shape[0])   # default 1°
    if R_meas.shape[0] == 2:
        R_rad[0, 0] = np.deg2rad(np.sqrt(R_meas[0, 0]))**2
        R_rad[1, 1] = np.deg2rad(np.sqrt(R_meas[1, 1]))**2

    return MSCEKF(
        state=msc_state,
        covariance=msc_cov,
        process_noise=Q,
        measurement_noise=R_rad,
    )


def init_msc_rp_ekf(
    detection: dict,
    r_min: float = 8e3,
    r_max: float = 8e4,
    num_filters: int = 3,
) -> list[MSCEKF]:
    """
    Range-parameterised bank of MSC-EKF filters (Gaussian sum filter style).

    Returns a list of *num_filters* MSCEKF instances covering [r_min, r_max]
    with logarithmically-spaced centre ranges.

    Parameters
    ----------
    detection : dict
        Same schema as init_cv_msc_ekf.
    r_min, r_max : float
        Range bracket in metres.
    num_filters : int
        Number of parallel filters.

    Returns
    -------
    filters : list[MSCEKF]
    """
    rho = (r_max / r_min) ** (1.0 / num_filters)
    Cr  = 2.0 * (rho - 1.0) / (rho + 1.0) / np.sqrt(12.0)

    filters = []
    for i in range(1, num_filters + 1):
        centre_range = r_min / 2.0 * (rho**i + rho**(i - 1))
        range_sigma  = Cr * centre_range
        ekf = init_cv_msc_ekf(detection, (centre_range, range_sigma))
        # Inflate velocity covariance to reflect the range-parameterisation prior
        ekf.covariance[1::2, 1::2] *= 400.0
        filters.append(ekf)

    return filters


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ypr_to_rotmat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """ZYX Euler rotation matrix (yaw-pitch-roll)."""
    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll),  np.sin(roll)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr           ],
    ])


def _rotate_msc_state(msc: np.ndarray, orient: np.ndarray) -> np.ndarray:
    """
    Apply orientation rotation to MSC state (first-order approximation).

    For identity orientation this is a no-op.  For non-trivial rotations
    we rotate the implied LOS unit vector and recompute az/el.
    """
    if np.allclose(orient, np.eye(3)):
        return msc.copy()

    az, el = msc[0], msc[2]
    los = np.array([np.cos(az)*np.cos(el),
                    np.sin(az)*np.cos(el),
                    np.sin(el)])
    los_rot = orient @ los
    az_new = np.arctan2(los_rot[1], los_rot[0])
    el_new = np.arcsin(np.clip(los_rot[2], -1.0, 1.0))

    msc_out      = msc.copy()
    msc_out[0]   = az_new
    msc_out[2]   = el_new
    return msc_out
