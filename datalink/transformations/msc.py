"""
Modified Spherical Coordinate (MSC) transformations.

State conventions
-----------------
Cartesian : [x, vx, y, vy, z, vz]   (interleaved position/velocity)
MSC       : [az, omega, el, elDot, invR, dRinvR]

  az     – azimuth angle (rad)
  omega  – azimuth rate projected onto the horizontal plane (rad/s)
  el     – elevation angle (rad)
  elDot  – elevation rate (rad/s)
  invR   – inverse range (1/m)
  dRinvR – range-rate / range  (s^-1),  equivalent to d(ln R)/dt
"""

import numpy as np


def cartesian_to_msc(x: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian state(s) to MSC state(s).

    Parameters
    ----------
    x : array_like, shape (6,) or (6, N)

    Returns
    -------
    msc : ndarray, same shape as *x*
    """
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 1
    if scalar:
        x = x[:, np.newaxis]

    X, Vx = x[0], x[1]
    Y, Vy = x[2], x[3]
    Z, Vz = x[4], x[5]

    r2_xy = X**2 + Y**2
    r2    = r2_xy + Z**2

    invR   = 1.0 / np.sqrt(r2)
    az     = np.arctan2(Y, X)
    el     = -np.arctan2(-Z, np.sqrt(r2_xy))
    dRinvR = (X*Vx + Y*Vy + Z*Vz) / r2
    omega  = (X*Vy - Y*Vx) / r2_xy * np.cos(np.arctan2(-Z, np.sqrt(r2_xy)))
    elDot  = -(Z*(X*Vx + Y*Vy) - Vz*r2_xy) / (np.sqrt(r2_xy) * r2)

    out = np.array([az, omega, el, elDot, invR, dRinvR])
    return out[:, 0] if scalar else out


def msc_to_cartesian(msc: np.ndarray) -> np.ndarray:
    """
    Convert MSC state(s) to Cartesian state(s).

    Parameters
    ----------
    msc : array_like, shape (6,) or (6, N)

    Returns
    -------
    cart : ndarray, same shape as *msc*
    """
    msc = np.asarray(msc, dtype=float)
    scalar = msc.ndim == 1
    if scalar:
        msc = msc[:, np.newaxis]

    az     =  msc[0]
    omega  =  msc[1]
    el     = -msc[2]   # elevation sign is flipped relative to raw MSC storage
    elDot  = -msc[3]
    invR   =  msc[4]
    dRinvR =  msc[5]

    x  =  np.cos(az)*np.cos(el) / invR
    y  =  np.sin(az)*np.cos(el) / invR
    z  = -np.sin(el) / invR
    vx = (dRinvR*np.cos(az)*np.cos(el) - omega*np.sin(az) - elDot*np.cos(az)*np.sin(el)) / invR
    vy = (dRinvR*np.sin(az)*np.cos(el) + omega*np.cos(az) - elDot*np.sin(az)*np.sin(el)) / invR
    vz = (-dRinvR*np.sin(el) - elDot*np.cos(el)) / invR

    out = np.array([x, vx, y, vy, z, vz])
    return out[:, 0] if scalar else out
