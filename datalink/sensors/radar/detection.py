"""
Detection dataclass — a single sensor measurement plus metadata.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Detection:
    """
    Single sensor detection.

    Attributes
    ----------
    time              : float     — simulation time [s]
    measurement       : ndarray   — position [x,y,z] m or [az_deg,el_deg,r] m
    sensor_index      : int
    measurement_noise : ndarray   — covariance matrix (n×n)
    measurement_parameters : dict — frame, orientation, etc.
    object_attributes : dict      — SNR, TargetIndex, …
    """
    time: float
    measurement: np.ndarray
    sensor_index: int = 0
    measurement_noise: np.ndarray = field(default_factory=lambda: np.eye(3))
    measurement_parameters: dict = field(default_factory=dict)
    object_attributes: dict = field(default_factory=dict)

    def __post_init__(self):
        self.measurement = np.asarray(self.measurement, dtype=float)
        self.measurement_noise = np.asarray(self.measurement_noise, dtype=float)
        # Ensure dicts are independent copies so callers can't mutate internals
        self.measurement_parameters = dict(self.measurement_parameters)
        self.object_attributes = dict(self.object_attributes)
