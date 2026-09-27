"""
Ground clutter model.
"""

from __future__ import annotations
import math
import numpy as np
from scipy.stats import gamma as gamma_dist


_C = 3e8  # speed of light [m/s]


class Clutter:
    """
    Surface clutter generator and SCR calculator.

    Parameters
    ----------
    params : dict
        resolution_in_surface  : float — surface bin resolution [m]
        range_limit            : float — max range for clutter [m]
        scatterer_distribution : str   — 'auto'|'gaussian'|'chisq_dof2'|'chisq_dof4'
        sigma_0                : float — clutter coefficient [dB m²/m²]
        tau                    : float — pulse width [s]
    """

    def __init__(self, params: dict):
        self.resolution_in_surface = params['resolution_in_surface']
        self.range_limit           = params['range_limit']
        self.scatterer_distribution = params['scatterer_distribution']
        self.sigma_0_db            = params['sigma_0']
        self.sigma_0_linear        = 10.0 ** (params['sigma_0'] / 10.0)
        self.tau                   = params['tau']

        # Set by set_one_step_params each simulation step
        self.altitude      = 0.0
        self.dcm_ned2body  = np.eye(3)
        self.dcm_body2ned  = np.eye(3)

    # ------------------------------------------------------------------ #
    def set_one_step_params(self, altitude: float, dcm_ned2body: np.ndarray) -> None:
        self.altitude     = altitude
        self.dcm_ned2body = np.asarray(dcm_ned2body, dtype=float)
        self.dcm_body2ned = self.dcm_ned2body.T

    # ------------------------------------------------------------------ #
    def create_surface_clutter(self,
                               theta_3db: float, phi_3db: float,
                               range_res: float,
                               theta_body: float, phi_body: float,
                               range_body: float) -> np.ndarray:
        """
        Generate clutter scatterer positions in body-frame spherical coords.

        Returns
        -------
        clutters : ndarray, shape (N, 3) — [az, el, r] in radians/metres
                   Empty array if no clutter.
        """
        # Convert body-spherical → NED-Cartesian → NED-spherical
        x, y, z = _sph2cart(theta_body, phi_body, range_body)
        local_cart = self.dcm_body2ned @ np.array([x, y, z])
        theta_l, phi_l, range_l = _cart2sph(*local_cart)

        # Only create clutter for targets below the horizon (phi_local < 0)
        if phi_l >= 0:
            return np.empty((0, 3))

        psi_g = -phi_l   # grazing angle
        range2surface = self.altitude / math.sin(psi_g)

        if range2surface > self.range_limit:
            return np.empty((0, 3))

        # Number of scatterer bins
        n_bins = round(100 * (1 + 30 * range_l / self.range_limit))

        dist = self.scatterer_distribution
        if dist == 'auto':
            dist = 'chisq_dof4' if range_body <= self.range_limit / 2 else 'gaussian'

        azs_, els_, ranges_ = _sample_scatterers(
            n_bins, dist, theta_l, phi_l, range_l,
            theta_3db, phi_3db, range_res)

        # Convert NED-spherical → NED-Cartesian → body-Cartesian → body-spherical
        ned_cart = np.column_stack([
            *_sph2cart_batch(azs_, els_, ranges_)
        ])                                           # (N, 3)
        body_cart = (self.dcm_ned2body @ ned_cart.T).T   # (N, 3)
        az, el, r = _cart2sph_batch(
            body_cart[:, 0], body_cart[:, 1], body_cart[:, 2])
        return np.column_stack([az, el, r])

    # ------------------------------------------------------------------ #
    def create_chaff(self,
                     target_velocity_global: np.ndarray,
                     theta_3db: float, phi_3db: float,
                     theta_body: float, phi_body: float,
                     range_body: float) -> np.ndarray:
        """Generate chaff cloud around target position (body-frame spherical)."""
        x, y, z = _sph2cart(theta_body, phi_body, range_body)
        local_cart = self.dcm_body2ned @ np.array([x, y, z])

        n = 200
        indexes = np.clip(np.random.randn(n, 3), -2.5, 2.5) / 2.4
        diss = indexes * 500.0
        pts_ned = local_cart[np.newaxis, :] + diss   # (N, 3)

        body_cart = (self.dcm_ned2body @ pts_ned.T).T   # (N, 3)
        az, el, r = _cart2sph_batch(
            body_cart[:, 0], body_cart[:, 1], body_cart[:, 2])
        return np.column_stack([az, el, r])

    # ------------------------------------------------------------------ #
    def calculate_scr_db(self,
                         target_rcs: float,
                         theta_3db: float, phi_3db: float,
                         theta_body: float, phi_body: float,
                         range_body: float) -> float:
        """Signal-to-clutter ratio [dB]. Returns 0 if target above horizon."""
        x, y, z = _sph2cart(theta_body, phi_body, range_body)
        local_cart = self.dcm_body2ned @ np.array([x, y, z])
        _, phi_l, _ = _cart2sph(*local_cart)

        if phi_l >= 0:
            return 0.0
        psi_g = -phi_l
        range2surface = self.altitude / math.sin(psi_g)
        if range2surface > self.range_limit:
            return 0.0

        scr_lin = (2.0 * target_rcs * math.cos(psi_g)) / \
                  (self.sigma_0_linear * theta_3db * range_body * _C * self.tau)
        return 10.0 * math.log10(scr_lin)


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _sph2cart(az: float, el: float, r: float):
    x = r * math.cos(el) * math.cos(az)
    y = r * math.cos(el) * math.sin(az)
    z = r * math.sin(el)
    return x, y, z


def _cart2sph(x: float, y: float, z: float):
    r  = math.sqrt(x**2 + y**2 + z**2)
    az = math.atan2(y, x)
    el = math.atan2(z, math.sqrt(x**2 + y**2))
    return az, el, r


def _sph2cart_batch(az, el, r):
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


def _cart2sph_batch(x, y, z):
    r  = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(x**2 + y**2))
    return az, el, r


def _sample_scatterers(n: int, dist: str,
                       theta_l, phi_l, range_l,
                       theta_3db, phi_3db, range_res):
    if dist == 'gaussian':
        idx = np.clip(np.random.randn(n, 3), -2.5, 2.5) / 2.4
        azs    = theta_l + idx[:, 0] * theta_3db / 2
        els    = phi_l   + idx[:, 1] * phi_3db   / 2
        ranges = range_l + idx[:, 2] * range_res  / 2

    elif dist == 'chisq_dof2':
        az_idx  = _centered_gamma(n, 1, theta_3db, theta_3db)
        el_idx  = _centered_gamma(n, 1, phi_3db,   phi_3db)
        rng_idx = _centered_gamma(n, 1, range_res,  range_res)
        azs    = theta_l + az_idx
        els    = phi_l   + el_idx
        ranges = range_l + rng_idx

    elif dist == 'chisq_dof4':
        az_idx  = _centered_gamma(n, 2, theta_3db / 2, theta_3db)
        el_idx  = _centered_gamma(n, 2, phi_3db   / 2, phi_3db)
        rng_idx = _centered_gamma(n, 2, range_res  / 2, range_res)
        azs    = theta_l + az_idx
        els    = phi_l   + el_idx
        ranges = range_l + rng_idx

    else:
        raise ValueError(f"Unknown scatterer distribution: {dist!r}")

    return azs, els, ranges


def _centered_gamma(n: int, a: float, scale: float, clip: float) -> np.ndarray:
    samples = gamma_dist.rvs(a=a, scale=scale, size=n)
    samples = np.clip(samples, 0, clip)
    return samples - samples.mean()
