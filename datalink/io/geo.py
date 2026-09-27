"""
Geodetic / coordinate conversion utilities.

All angles in **degrees**.
All distances in **metres**.

WGS-84 ellipsoid parameters:
    a = 6 378 137 m  (semi-major axis)
    e²= 6.694 379 990 14 × 10⁻³  (first eccentricity squared)
"""

from __future__ import annotations

import math

import numpy as np


# ------------------------------------------------------------------ #
# WGS-84 constants
# ------------------------------------------------------------------ #

_WGS84_A    = 6_378_137.0
_WGS84_E_SQ = 6.69437999014e-3
_WGS84_E    = math.sqrt(_WGS84_E_SQ)
_WGS84_F    = 1.0 / 298.257223563
_WGS84_B    = _WGS84_A * (1.0 - _WGS84_F)
_WGS84_B_SQ = _WGS84_B ** 2
_WGS84_A_SQ = _WGS84_A ** 2
_WGS84_EP_SQ = (_WGS84_A_SQ - _WGS84_B_SQ) / _WGS84_B_SQ


# ------------------------------------------------------------------ #
# LLA → ECEF
# ------------------------------------------------------------------ #

def lla_to_ecef(
    lat: float | np.ndarray,
    lon: float | np.ndarray,
    alt: float | np.ndarray,
) -> tuple:
    """
    Convert geodetic (lat, lon, alt) to ECEF (x, y, z).

    Parameters
    ----------
    lat, lon : degrees
    alt      : metres above ellipsoid

    Returns
    -------
    (x, y, z) in metres
    """
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)

    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E_SQ * np.sin(lat_r) ** 2)
    x = (N + alt) * np.cos(lat_r) * np.cos(lon_r)
    y = (N + alt) * np.cos(lat_r) * np.sin(lon_r)
    z = (_WGS84_B_SQ / _WGS84_A_SQ * N + alt) * np.sin(lat_r)
    return x, y, z


# ------------------------------------------------------------------ #
# ECEF → NED (differential, requires reference origin)
# ------------------------------------------------------------------ #

def ecef_to_ned(
    u: float | np.ndarray,
    v: float | np.ndarray,
    w: float | np.ndarray,
    lat0: float,
    lon0: float,
) -> tuple:
    """
    Rotate an ECEF difference vector (u, v, w) to NED at (lat0, lon0).

    Parameters
    ----------
    u, v, w  : ECEF vector components [m]
    lat0, lon0 : reference origin [degrees]

    Returns
    -------
    (N, E, D) in metres
    """
    cos_phi    = math.cos(math.radians(lat0))
    sin_phi    = math.sin(math.radians(lat0))
    cos_lambda = math.cos(math.radians(lon0))
    sin_lambda = math.sin(math.radians(lon0))

    t = cos_lambda * u + sin_lambda * v
    E = -sin_lambda * u + cos_lambda * v
    D = -(cos_phi * t + sin_phi * w)
    N = -sin_phi * t + cos_phi * w
    return N, E, D


# ------------------------------------------------------------------ #
# NED (ENU) → ECEF
# Note: the ENU-style argument order is (xEast, yNorth, zUp)
# ------------------------------------------------------------------ #

def ned_to_ecef(
    x_east:  float | np.ndarray,
    y_north: float | np.ndarray,
    z_up:    float | np.ndarray,
    lat0: float,
    lon0: float,
    h0:   float,
) -> tuple:
    """
    Convert local ENU offset to ECEF absolute position.

    Parameters
    ----------
    x_east, y_north, z_up : local ENU offsets [m]
    lat0, lon0 : reference origin [degrees]
    h0         : reference altitude [m]

    Returns
    -------
    (x, y, z) ECEF [m]
    """
    x0, y0, z0 = lla_to_ecef(lat0, lon0, h0)

    cos_phi    = math.cos(math.radians(lat0))
    sin_phi    = math.sin(math.radians(lat0))
    cos_lambda = math.cos(math.radians(lon0))
    sin_lambda = math.sin(math.radians(lon0))

    t  = cos_phi * z_up - sin_phi * y_north
    dz = sin_phi * z_up + cos_phi * y_north
    dx = cos_lambda * t - sin_lambda * x_east
    dy = sin_lambda * t + cos_lambda * x_east

    return x0 + dx, y0 + dy, z0 + dz


# ------------------------------------------------------------------ #
# ECEF → LLA (iterative Bowring / closed-form Vermeille)
# ------------------------------------------------------------------ #

def ecef_to_lla(
    x: float | np.ndarray,
    y: float | np.ndarray,
    z: float | np.ndarray,
) -> tuple:
    """
    Convert ECEF (x, y, z) to geodetic (lat_deg, lon_deg, alt_m).

    Iterative Bowring method — numerically stable at all latitudes.
    Converges to machine precision in 2–3 iterations.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    a    = _WGS84_A
    e_sq = _WGS84_E_SQ

    lon = np.arctan2(y, x)
    r   = np.sqrt(x**2 + y**2)

    # Initial latitude estimate (Bowring)
    lat = np.arctan2(z, r * (1.0 - e_sq))

    for _ in range(10):
        sin_lat = np.sin(lat)
        N       = a / np.sqrt(1.0 - e_sq * sin_lat**2)
        lat_new = np.arctan2(z + e_sq * N * sin_lat, r)
        if np.all(np.abs(lat_new - lat) < 1e-12):
            lat = lat_new
            break
        lat = lat_new

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    N = a / np.sqrt(1.0 - e_sq * sin_lat**2)

    # Altitude: use r/cos or z/sin depending on which is more stable
    near_pole = np.abs(cos_lat) < 1e-10
    alt = np.where(
        near_pole,
        np.abs(z) / np.where(np.abs(sin_lat) > 1e-30, np.abs(sin_lat), 1.0)
            - N * (1.0 - e_sq),
        r / np.where(np.abs(cos_lat) > 1e-30, np.abs(cos_lat), 1.0) - N,
    )

    return np.degrees(lat), np.degrees(lon), alt


# ------------------------------------------------------------------ #
# Geodetic → NED (convenience combining the above)
# ------------------------------------------------------------------ #

def geodetic_to_ned(
    lat: float,
    lon: float,
    alt: float,
    lat0: float,
    lon0: float,
    alt0: float,
) -> tuple:
    """
    Convert geodetic target (lat, lon, alt) to NED relative to origin.

    Returns
    -------
    (N, E, D) in metres
    """
    x,  y,  z  = lla_to_ecef(lat,  lon,  alt)
    x0, y0, z0 = lla_to_ecef(lat0, lon0, alt0)
    return ecef_to_ned(x - x0, y - y0, z - z0, lat0, lon0)


# ------------------------------------------------------------------ #
# NED state (Cartesian) → geodetic absolute position
# ------------------------------------------------------------------ #

def ned_state_to_lla(
    ned_state: np.ndarray,
    lat0: float,
    lon0: float,
    alt0: float,
) -> tuple[float, float, float]:
    """
    Convert a NED Cartesian state vector [x, vx, y, vy, z, vz]
    (indices 0,2,4 are position) to absolute (lat, lon, alt).

    The NED convention used by FusionManager: state = [x, vx, y, vy, z, vz].
    Mapping to ENU for ned_to_ecef: xEast=state[2], yNorth=state[0], zUp=state[4].
    Note: state[4] (Z_ned, i.e. Down) is passed directly as zUp without
    sign-flipping — this is the defined calling convention, not a bug.
    """
    x_east  = float(ned_state[2])   # Y_ned → East
    y_north = float(ned_state[0])   # X_ned → North
    z_up    = float(ned_state[4])   # Z_ned is Down; passed directly as "zUp" (no sign flip)
    ex, ey, ez = ned_to_ecef(x_east, y_north, z_up, lat0, lon0, alt0)
    lat, lon, alt = ecef_to_lla(ex, ey, ez)
    return float(lat), float(lon), float(alt)
