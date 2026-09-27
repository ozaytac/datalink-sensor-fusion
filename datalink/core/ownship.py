"""
Ownship — top-level platform that owns all sensors, trackers,
fusion manager, and simulation summary.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from datalink.sensors.radar.radar_sensor import RadarSensor
from datalink.sensors.infrared.infrared_sensor import InfraredSensor
from datalink.fusion.fusion_manager import FusionManager
from datalink.core.simulation_summary import SimulationSummary


def _angle2dcm(yaw: float, pitch: float, roll: float) -> np.ndarray:
    return Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()


class Ownship:
    """
    Ownship platform — orchestrates sensor updates, tracking, and fusion.

    Parameters
    ----------
    platform_id   : int
    trajectory    : trajectory object (with .Position, .Velocity, .Acceleration)
    params        : full params dict
    blue1_idx     : index of this ownship in the trajectory array
    initial_pos   : (3,) NED initial position [m]
    initial_lla   : (3,) [lat, lon, alt]
    num_ownship   : total number of ownships in scenario
    num_target    : total number of targets
    """

    def __init__(
        self,
        platform_id:  int,
        trajectory:   Any,
        params:       dict,
        blue1_idx:    int,
        initial_pos:  np.ndarray,
        initial_lla:  np.ndarray,
        num_ownship:  int,
        num_target:   int,
    ) -> None:
        self.ID           = platform_id
        self.platform     = trajectory
        self.params       = params
        self.num_ownship  = num_ownship
        self.num_target   = num_target
        self.initial_position = np.asarray(initial_pos, dtype=float)
        self.lla           = np.asarray(initial_lla, dtype=float)

        self.total_n_cells_in_ir: int = 10

        # Sensor objects
        self.radar_sensor:    Optional[RadarSensor]    = None
        self.ir_sensor:       Optional[InfraredSensor] = None

        # Fusion manager — factory via params
        fm_class_name = params['scenarioParams'].get('FusionManager', 'FusionManager')
        self.fusion_manager = FusionManager(trajectory, params)

        # Sim summary
        self.sim_summary = SimulationSummary()

        # Track outputs
        self.confirmed_radar_tracks: list = []
        self.confirmed_ir_tracks:    list = []
        self.confirmed_ew_tracks:    list = []
        self.all_ir_tracks:          list = []
        self.old_ir_tracks:          list = []
        self.ew_track_history:       list = [[], []]

        # Detection histories
        self.radar_detection_history: list = []
        self.ir_detection_history:    list = []
        self.ew_detection_history:    list = []

        # Cached per-step pose/target data
        self.ownship_data:    Optional[Any] = None
        self.targets_local:   Optional[list] = None
        self.pose_in_local:   list = []
        self.pose_in_ownship: list = []

        # Previous pose for ownship-motion compensation
        self._prev_pose: Optional[np.ndarray] = None
        self._prev_vel:  Optional[np.ndarray] = None

        self._initialize_sensors(params, blue1_idx)

    # ------------------------------------------------------------------ #
    def _initialize_sensors(self, params: dict, blue1_idx: int) -> None:
        sp = params['scenarioParams']
        if sp.get('HasRadar', False):
            self.radar_sensor = RadarSensor(
                blue1_index=blue1_idx,
                sensor_index=self.ID * 3 - 2,
                params=params,
            )
        if sp.get('HasInfrared', False):
            ir_p = params.get('IRParams', {})
            self.ir_sensor = InfraredSensor(
                sensor_index=self.ID * 3 - 1,
                params={'infraredSensor': {
                    'lens_diameter':       ir_p.get('LensDiameter', 8e-2),
                    'focal_length':        ir_p.get('FocalLength', 800),
                    'num_detectors':       ir_p.get('NumDetectors', [1000, 1000]),
                    'cutoff_frequency':    ir_p.get('CutoffFrequency', 20e3),
                    'detector_area':       ir_p.get('DetectorArea', 1.44e-6),
                    'detectivity':         ir_p.get('Detectivity', 1.2e10),
                    'neq_bandwidth':       ir_p.get('NoiseEquivalentBandwidth', 30),
                    'false_alarm_rate':    ir_p.get('FalseAlarmRate', 1e-6),
                    'has_elevation':       ir_p.get('HasElevation', True),
                    'has_noise':           ir_p.get('HasNoise', True),
                    'has_false_alarms':    ir_p.get('HasFalseAlarms', True),
                    'az_bias_fraction':    ir_p.get('AzimuthBiasFraction', 0.1),
                    'el_bias_fraction':    ir_p.get('ElevationBiasFraction', 0.1),
                    'scan_mode':           ir_p.get('ScanMode', 'Mechanical'),
                }},
            )

    # ------------------------------------------------------------------ #
    def produce_sensors_data(self, sim_time: float, traj: Any) -> None:
        """
        Pre-compute per-step ownship + target trajectory data.
        Must be called before stepRadar / stepInfrared.
        """
        self.ownship_data, self.targets_local = \
            self._produce_target_relative_positions(sim_time, traj)

    # ------------------------------------------------------------------ #
    def step_radar(
        self,
        sim_time:   float,
        delta_time: float,
    ) -> Tuple[list, list]:
        """
        Run radar sensor for one step.

        Returns
        -------
        (radar_detections, confirmed_radar_tracks)
        """
        if self.radar_sensor is None:
            return [], []

        body_dets, tracker_dets, tracks, cov_config = \
            self.radar_sensor.get_detections_and_tracks(
                sim_time, delta_time,
                self.ownship_data, self.targets_local, self.platform)

        self.confirmed_radar_tracks = tracks
        self.radar_detection_history.append(body_dets)
        return body_dets, tracks

    # ------------------------------------------------------------------ #
    def step_infrared(
        self,
        sim_time:   float,
        delta_time: float,
        ir_tracker: Any,
    ) -> Tuple[list, list]:
        """
        Run IR sensor and tracker for one step.

        Returns
        -------
        (ir_detections_all, confirmed_ir_tracks)
        """
        if self.ir_sensor is None:
            return [], []

        all_ir_dets: list = []
        all_tracks:  list = []
        ir_p = self.params.get('IRParams', {})
        fov_az1 = math.radians(ir_p.get('AzimuthScanLimits', [-60, 60])[0])
        fov_az2 = math.radians(ir_p.get('AzimuthScanLimits', [-60, 60])[1])
        fov_azs = np.linspace(fov_az1, fov_az2, self.total_n_cells_in_ir + 1)

        for i in range(self.total_n_cells_in_ir - 1, -1, -1):
            sensor_time = (sim_time
                           - i / (self.params['scenarioParams']['updateRate']
                                  * self.total_n_cells_in_ir))

            # Build per-cell target subset
            targets_cell, dcm = self._get_targets_in_cell(
                fov_azs[i], fov_azs[i + 1], sensor_time)

            # Get IR detections
            ir_dets = self._get_ir_detections(targets_cell, dcm, sensor_time)
            all_ir_dets.extend(ir_dets)

        self.ir_detection_history.append(all_ir_dets)
        # Simplified: return all detections, no tracker step here
        # Full IR tracker integration (GNN + MSC-EKF) can be wired in later
        confirmed = []
        return all_ir_dets, confirmed

    # ------------------------------------------------------------------ #
    def update_fusion(
        self,
        sim_time: float,
        confirmed_radar_tracks: list,
        confirmed_ir_tracks:    list,
    ) -> list:
        """
        Run one fusion manager update step.

        Returns the list of fused track dicts.
        """
        self.fusion_manager.update_manager(
            confirmed_radar_tracks, confirmed_ir_tracks, sim_time)
        return self.fusion_manager.get_fused_tracks()

    # ------------------------------------------------------------------ #
    def update_fusion_datalink(self, datalink_tracks: list) -> None:
        """Incorporate DataLink tracks into fusion manager."""
        self.fusion_manager.update_manager_with_datalink(datalink_tracks)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _produce_target_relative_positions(
        self, sim_time: float, traj: Any
    ) -> Tuple[Any, list]:
        """
        Build ownship_data and targets_local for the current timestep.

        Collects time-interpolated LLA + RCS / Swerling / Chaff /
        JammerParams arrays for each target relative to this ownship.
        """
        import types
        update_rate = self.params['scenarioParams']['updateRate']
        prf = self.params['trackingRadar']['PRF']
        start_time = sim_time - 1.0 / update_rate
        end_time   = sim_time

        # Coarse time samples (at traj file rate)
        traj_rate = self.params['scenarioParams'].get('trajFileUpdateRate', 50.0)
        coarse_times = np.arange(start_time, end_time + 1e-9, 1.0 / traj_rate)

        # ownship data (ID is 1-based; list index is 0-based)
        ow_id = self.ID
        if isinstance(traj, list):
            idx = ow_id - 1
            ow_traj = traj[idx] if 0 <= idx < len(traj) else traj[0]
        elif isinstance(traj, dict):
            ow_traj = traj.get(ow_id, traj.get(0, next(iter(traj.values()))))
        else:
            ow_traj = traj

        def _interp_from_traj(platform, attr_name, t_dst):
            """Interpolate a trajectory attribute onto t_dst using the platform's time axis."""
            t_src = np.atleast_1d(getattr(platform, 'time', coarse_times)).astype(float)
            vals  = np.atleast_1d(getattr(platform, attr_name, np.zeros(len(t_src)))).astype(float)
            if len(vals) != len(t_src):
                vals = np.zeros(len(t_src))
            return np.interp(t_dst, t_src, vals)

        ownship_data = types.SimpleNamespace(
            time  = coarse_times,
            yaw   = _interp_from_traj(ow_traj, 'heading',   coarse_times),
            pitch = _interp_from_traj(ow_traj, 'pitch',     coarse_times),
            roll  = _interp_from_traj(ow_traj, 'roll',      coarse_times),
            lat   = _interp_from_traj(ow_traj, 'latitude',  coarse_times),
            lon   = _interp_from_traj(ow_traj, 'longitude', coarse_times),
            alt   = _interp_from_traj(ow_traj, 'altitude',  coarse_times),
        )

        # Fine interpolation at PRF rate
        fine_times = np.arange(start_time, end_time + 1e-9, 1.0 / prf)
        for attr in ('yaw', 'pitch', 'roll', 'lat', 'lon', 'alt'):
            setattr(ownship_data, attr,
                    np.interp(fine_times, coarse_times, getattr(ownship_data, attr)))
        ownship_data.time = fine_times

        # targets_local — minimal structure
        targets_local = []
        tgt_trajs = [t for t in (traj if isinstance(traj, list) else [])
                     if getattr(t, 'id', None) != ow_id]

        for tgt_t in tgt_trajs:
            tl = types.SimpleNamespace(
                PlatformID = getattr(tgt_t, 'id', -1),
                lat        = _interp_from_traj(tgt_t, 'latitude',  fine_times),
                lon        = _interp_from_traj(tgt_t, 'longitude', fine_times),
                alt        = _interp_from_traj(tgt_t, 'altitude',  fine_times),
                RCS        = np.full(len(fine_times), getattr(tgt_t, 'RCS', 1.0)),
                Swerling   = np.full(len(fine_times), int(getattr(tgt_t, 'Swerling', 0))),
                Chaff      = np.zeros(len(fine_times), dtype=int),
                JammerParams = [_default_jammer_params(tgt_t, t)
                                for t in fine_times],
            )
            targets_local.append(tl)

        return ownship_data, targets_local

    def _get_targets_in_cell(
        self, az_min: float, az_max: float, sensor_time: float
    ) -> Tuple[list, np.ndarray]:
        """Filter targets_local to those whose azimuth falls in [az_min, az_max]."""
        import types
        if not self.ownship_data or not self.targets_local:
            return [], np.eye(3)

        times = np.asarray(self.ownship_data.time)
        idx = int(np.argmin(np.abs(times - sensor_time)))

        yaw   = float(self.ownship_data.yaw[idx])
        pitch = float(self.ownship_data.pitch[idx])
        roll  = float(self.ownship_data.roll[idx])
        dcm   = _angle2dcm(yaw, pitch, roll)

        lat0 = float(self.ownship_data.lat[idx])
        lon0 = float(self.ownship_data.lon[idx])
        alt0 = float(self.ownship_data.alt[idx])

        from datalink.sensors.radar.radar_sensor import _geodetic_to_ned
        cell_targets = []
        for tgt_l in self.targets_local:
            lat = float(tgt_l.lat[idx])
            lon = float(tgt_l.lon[idx])
            alt = float(tgt_l.alt[idx])
            n, e, d = _geodetic_to_ned(lat, lon, alt, lat0, lon0, alt0)
            body_pos = dcm @ np.array([n, e, -d])
            az = math.atan2(body_pos[1], body_pos[0])
            if az_min <= az <= az_max:
                tgt_cell = types.SimpleNamespace(**vars(tgt_l))
                tgt_cell.Position = body_pos
                tgt_cell.IRPower  = getattr(tgt_l, 'IRPower', 1e3)
                cell_targets.append(tgt_cell)

        return cell_targets, dcm

    def _get_ir_detections(
        self, targets_cell: list, dcm: np.ndarray, time: float
    ) -> list:
        """Call InfraredSensor.step for the given cell."""
        if self.ir_sensor is None or not targets_cell:
            return []

        import types
        ins = types.SimpleNamespace(
            Position    = np.asarray(getattr(self.platform, 'Position', np.zeros(3))),
            Velocity    = np.asarray(getattr(self.platform, 'Trajectory',
                                              types.SimpleNamespace(
                                                  Velocity=np.zeros(3))).Velocity),
            Orientation = dcm,
        )
        dets, _, _ = self.ir_sensor.step(targets_cell, ins, time)
        return dets


# ------------------------------------------------------------------ #

def _default_jammer_params(tgt_traj: Any, time: float) -> Any:
    """Build a minimal jammer params namespace for a target at a given time."""
    import types
    jp = types.SimpleNamespace(is_jamming=0)
    # If trajectory has jammer data, populate here
    if hasattr(tgt_traj, 'Jammer'):
        times = getattr(tgt_traj, 'time', [0.0])
        idx   = int(np.argmin(np.abs(np.asarray(times) - time)))
        if tgt_traj.Jammer[idx]:
            jp.is_jamming = 1
            jp_arr = tgt_traj.JammerParams[:, idx]
            jp.transmitter_power = float(jp_arr[0])
            jp.antenna_gain      = float(jp_arr[1])
            jp.jammer_losses     = float(jp_arr[2])
            jp.bandwidth         = float(jp_arr[3])
    return jp
