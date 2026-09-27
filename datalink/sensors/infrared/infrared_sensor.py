"""
Infrared sensor stub.

This module provides a minimal, interface-compatible stub that:
  - Accepts the same constructor parameters
  - Exposes the key public properties referenced by Ownship/simulation code
  - Returns empty detections — to be replaced with a real IR model later
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class IRDetection:
    """Minimal IR detection (azimuth, elevation in degrees)."""
    time: float
    azimuth: float          # deg
    elevation: float        # deg
    snr_db: float
    target_index: int
    sensor_index: int


class InfraredSensor:
    """
    Stub IR sensor — angular-only measurements (azimuth, elevation).

    Parameters
    ----------
    sensor_index : int
    params       : dict (full params dict; uses 'infraredSensor' sub-key if present)
    """

    def __init__(self, sensor_index: int, params: Optional[dict] = None) -> None:
        self.sensor_index = sensor_index
        self.params = params or {}
        p = self.params.get('infraredSensor', {})

        # Optics
        self.lens_diameter  = float(p.get('lens_diameter',  8e-2))   # m
        self.focal_length   = float(p.get('focal_length',   800.0))  # pixels
        self.num_detectors  = list(p.get('num_detectors',   [1000, 1000]))  # [rows, cols]
        self.cutoff_freq    = float(p.get('cutoff_frequency', 20e3))
        self.detector_area  = float(p.get('detector_area',  1.44e-6))
        self.detectivity    = float(p.get('detectivity',    1.2e10))
        self.neq_bandwidth  = float(p.get('neq_bandwidth',  30.0))

        # Detection parameters
        self.false_alarm_rate       = float(p.get('false_alarm_rate', 1e-6))
        self.has_elevation          = bool(p.get('has_elevation', True))
        self.has_noise              = bool(p.get('has_noise', True))
        self.has_false_alarms       = bool(p.get('has_false_alarms', True))
        self.az_bias_fraction       = float(p.get('az_bias_fraction', 0.1))
        self.el_bias_fraction       = float(p.get('el_bias_fraction', 0.1))
        self.scan_mode              = str(p.get('scan_mode', 'Mechanical'))
        self.mounting_location      = np.zeros(3)
        self.mounting_angles        = np.zeros(3)

        # Derived — azimuth / elevation resolutions (deg)
        fov_az = 2.0 * math.degrees(math.atan(self.num_detectors[1] / (2.0 * self.focal_length)))
        fov_el = 2.0 * math.degrees(math.atan(self.num_detectors[0] / (2.0 * self.focal_length)))
        self.field_of_view = np.array([fov_az, fov_el])
        self.az_resolution = fov_az / self.num_detectors[1]
        self.el_resolution = fov_el / self.num_detectors[0]

        # Sensor Fusion Toolbox scan-mode state (stub)
        self._is_locked = False

    # ------------------------------------------------------------------ #
    # Interface methods
    # ------------------------------------------------------------------ #

    def step(
        self,
        targets: List[Any],
        ins: Any,
        time: float,
    ) -> Tuple[List[IRDetection], int, Dict]:
        """
        Generate IR detections.

        Parameters
        ----------
        targets : list of target structs / objects
        ins     : INS struct (Position, Velocity, Orientation)
        time    : float [s]

        Returns
        -------
        (detections, num_dets, config)
        """
        self._is_locked = True
        detections: List[IRDetection] = []

        dcm = np.asarray(ins.Orientation, dtype=float)   # NED→body rotation

        for tgt in targets:
            pos_ned  = np.asarray(tgt.Position, dtype=float)
            pos_body = dcm @ pos_ned
            az_rad  = math.atan2(pos_body[1], pos_body[0])
            el_rad  = math.atan2(-pos_body[2],
                                  math.hypot(pos_body[0], pos_body[1]))
            rng = float(np.linalg.norm(pos_body))

            snr_db = self._compute_snr_db(tgt, rng)
            if snr_db < 0:
                continue

            az_deg = math.degrees(az_rad)
            el_deg = math.degrees(el_rad)

            if self.has_noise:
                snr_lin = 10.0 ** (snr_db / 10.0)
                sigma_az = self._az_sigma(snr_lin)
                sigma_el = self._el_sigma(snr_lin) if self.has_elevation else 0.0
                az_deg += np.random.normal(0, sigma_az)
                el_deg += np.random.normal(0, sigma_el)

            detections.append(IRDetection(
                time=time,
                azimuth=az_deg,
                elevation=el_deg if self.has_elevation else 0.0,
                snr_db=snr_db,
                target_index=getattr(tgt, 'PlatformID', -1),
                sensor_index=self.sensor_index,
            ))

        if self.has_false_alarms:
            detections.extend(self._false_alarms(time))

        config = {
            'SensorIndex':  self.sensor_index,
            'FieldOfView':  self.field_of_view,
            'IsValidTime':  True,
        }
        return detections, len(detections), config

    def is_locked(self) -> bool:
        return self._is_locked

    def release(self) -> None:
        self._is_locked = False

    # ------------------------------------------------------------------ #
    # Private
    # ------------------------------------------------------------------ #

    def _compute_snr_db(self, tgt: Any, rng: float) -> float:
        """Simplified IR SNR: SNR = A_eff * P_IR / (NEP * pi * R^2)."""
        ir_power = getattr(tgt, 'IRPower', 1e3)   # W·sr⁻¹ fallback
        nep = (math.sqrt(self.detector_area * self.neq_bandwidth)
               / (self.detectivity / 1e2))
        A_eff = (self.lens_diameter ** 2) / 4.0
        snr_lin = A_eff * ir_power / (nep * math.pi * max(rng, 1.0) ** 2)
        return 10.0 * math.log10(max(snr_lin, 1e-30))

    def _az_sigma(self, snr_lin: float) -> float:
        if self.scan_mode.lower() == 'no scanning':
            fn = 1.0 / (1.6 * math.sqrt(2.0 * max(snr_lin, 1e-9)))
        else:
            fc = self.cutoff_freq
            scale = 3.704 / (2.0 * math.pi * fc)
            fn = scale / (math.sqrt(max(snr_lin, 1e-9)) * self.az_resolution)
        return self.az_resolution * math.sqrt(fn**2 + self.az_bias_fraction**2)

    def _el_sigma(self, snr_lin: float) -> float:
        if self.scan_mode.lower() == 'no scanning':
            fn = 1.0 / (1.6 * math.sqrt(2.0 * max(snr_lin, 1e-9)))
        else:
            fc = self.cutoff_freq
            scale = 3.704 / (2.0 * math.pi * fc)
            fn = scale / (math.sqrt(max(snr_lin, 1e-9)) * self.el_resolution)
        return self.el_resolution * math.sqrt(fn**2 + self.el_bias_fraction**2)

    def _false_alarms(self, time: float) -> List[IRDetection]:
        fa = []
        fov_az, fov_el = self.field_of_view
        p_fa = self.false_alarm_rate * (fov_az / self.az_resolution)
        if self.has_elevation:
            p_fa *= fov_el / self.el_resolution
        if np.random.rand() < p_fa:
            fa.append(IRDetection(
                time=time,
                azimuth=np.random.uniform(-fov_az / 2, fov_az / 2),
                elevation=np.random.uniform(-fov_el / 2, fov_el / 2)
                          if self.has_elevation else 0.0,
                snr_db=0.0,
                target_index=-1,
                sensor_index=self.sensor_index,
            ))
        return fa
