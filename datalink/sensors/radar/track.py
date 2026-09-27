"""
Track dataclass — a confirmed track's state, covariance, and metadata.

State convention: [x, vx, y, vy, z, vz]
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Track:
    """
    Single target track.

    Attributes
    ----------
    track_id          : int
    state             : ndarray, shape (6,)  — [x,vx,y,vy,z,vz]
    state_covariance  : ndarray, shape (6,6)
    update_time       : float
    age               : int     — number of steps since birth
    is_confirmed      : bool    — tentative (False) or confirmed (True)
    object_attributes : dict    — SNR, TargetIndex, …
    """
    track_id: int
    state: np.ndarray
    state_covariance: np.ndarray
    update_time: float = 0.0
    age: int = 1
    is_confirmed: bool = False
    object_attributes: dict = field(default_factory=dict)

    # How many consecutive steps without a detection
    missed_count: int = 0
    # How many consecutive detections (for confirmation)
    hit_count: int = 1

    def __post_init__(self):
        self.state = np.asarray(self.state, dtype=float)
        self.state_covariance = np.asarray(self.state_covariance, dtype=float)

    @property
    def position(self) -> np.ndarray:
        """Return [x, y, z] position."""
        return self.state[[0, 2, 4]]

    @property
    def velocity(self) -> np.ndarray:
        """Return [vx, vy, vz] velocity."""
        return self.state[[1, 3, 5]]
