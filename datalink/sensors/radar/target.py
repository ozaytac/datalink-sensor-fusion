"""
Radar target model.
"""

import numpy as np
from .fluctuation_models import FluctuationModel


class Target:
    """
    Single radar target with position and RCS model.

    Parameters
    ----------
    target_id       : int
    pose_in_ownship : array_like (3,)  — body-frame Cartesian position [m]
    pose_in_local   : array_like (3,)  — NED-frame Cartesian position [m]
    rcs_avg         : float            — mean RCS [m²]
    swerling_type   : str              — 'Swerling0'..'Swerling5'
    chaff           : int              — 1 if chaff is deployed, 0 otherwise
    """

    def __init__(self, target_id: int,
                 pose_in_ownship, pose_in_local,
                 rcs_avg: float, swerling_type: str, chaff: int):
        self.ID = target_id
        self.chaff = chaff

        self.target_local_cartesian = np.asarray(pose_in_local, dtype=float)
        self.target_body_cartesian  = np.asarray(pose_in_ownship, dtype=float)

        self.target_local_spherical = _cart2sph(self.target_local_cartesian)
        self.target_body_spherical  = _cart2sph(self.target_body_cartesian)

        self.RCS_avg       = rcs_avg
        self.swerling_type = swerling_type
        self.fluc_model    = FluctuationModel(swerling_type, rcs_avg)

    # ------------------------------------------------------------------ #
    def sample_rcs(self) -> float:
        """Draw a random RCS sample [m²]."""
        return self.fluc_model.sample()

    # ---- Getters ---- #
    def get_local_cartesian(self):
        x, y, z = self.target_local_cartesian
        return x, y, z

    def get_local_spherical(self):
        az, el, r = self.target_local_spherical
        return az, el, r

    def get_body_cartesian(self):
        x, y, z = self.target_body_cartesian
        return x, y, z

    def get_body_spherical(self):
        az, el, r = self.target_body_spherical
        return az, el, r

    def set_body_spherical(self, az: float, el: float, r: float) -> None:
        self.target_body_spherical = np.array([az, el, r])

    # ---- Velocity (optional, set externally) ---- #
    @property
    def target_velocity_global(self) -> np.ndarray:
        return getattr(self, '_velocity_global', np.zeros(3))

    @target_velocity_global.setter
    def target_velocity_global(self, v):
        self._velocity_global = np.asarray(v, dtype=float)


# ------------------------------------------------------------------ #
def _cart2sph(xyz) -> np.ndarray:
    """Convert Cartesian [x,y,z] → [az, el, r] (radians)."""
    x, y, z = xyz
    r  = float(np.sqrt(x**2 + y**2 + z**2))
    az = float(np.arctan2(y, x))
    el = float(np.arctan2(z, np.sqrt(x**2 + y**2)))
    return np.array([az, el, r])
