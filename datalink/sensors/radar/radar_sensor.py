"""
Radar sensor — top-level interface used by simulation.

Wraps MPAR and handles multi-cluster FOV sweeping, track deduplication,
and trajectory data access.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .mpar import MPAR
from .detection import Detection
from .track import Track


class RadarSensor:
    """
    Top-level radar sensor wrapper.

    Parameters
    ----------
    blue1_index  : int   — index of ownship in trajectory data
    sensor_index : int   — unique sensor identifier
    params       : dict  — full params dict (from params.json)
    """

    def __init__(
        self,
        blue1_index: int,
        sensor_index: int,
        params: dict,
    ) -> None:
        self.blue1_index  = blue1_index
        self.sensor_index = sensor_index
        self.params = params

        self.traj_data: Optional[Any] = None
        self.traj_data_sample_rate: float = 1.0
        self.ownship_index_in_traj: int = blue1_index

        self.n_clusters: int = 5
        theta_fov = params['trackingRadar']['theta_FOV']
        self.fov_theta_s = np.linspace(
            -theta_fov, theta_fov, self.n_clusters + 1)

        # Build MPAR
        self.mpar = MPAR(sensor_index, params)

    # ------------------------------------------------------------------ #
    def get_detections_and_tracks(
        self,
        start_time: float,
        delta_time: float,
        ownship_data: Any,
        targets_local: Any,
        blue1: Any,
    ) -> Tuple[List[Detection], List[Detection], List[Track], Dict]:
        """
        Run all FOV cluster sub-steps and return deduplicated tracks.

        Parameters
        ----------
        start_time    : float — current sim time [s]
        delta_time    : float — sim step duration [s]
        ownship_data  : object with .time, .yaw, .pitch, .roll, .lat, .lon, .alt
        targets_local : list of target scenario objects
        blue1         : ownship platform with .Trajectory.{Position,Velocity,Acceleration}

        Returns
        -------
        (body_dets, tracker_dets, unique_confirmed_tracks, coverage_config)
        """
        phi_fov = self.params['trackingRadar']['phi_FOV']
        all_body_dets:   List[Detection] = []
        all_tracker_dets: List[Detection] = []
        all_tracks: List[Track] = []

        coverage_config: Dict = {}

        for i_c in range(self.n_clusters - 1, -1, -1):
            intr_time = start_time - delta_time * i_c / self.n_clusters
            sub_dt    = delta_time / self.n_clusters

            (jammers, targets, ownship_pos, ownship_vel,
             ownship_acc, ownship_ori, cov_cfg) = self._create_one_step_vars(
                ownship_data, targets_local, intr_time, blue1)

            coverage_config = cov_cfg

            # Set FOV for this cluster slice
            self.mpar.tracking_radar.set_FOV(
                math.degrees(self.fov_theta_s[i_c]),
                math.degrees(self.fov_theta_s[i_c + 1]),
                -phi_fov, phi_fov,
            )

            body_dets, tracker_dets, tracks = self.mpar.update_sensor(
                ownship_pos, ownship_ori, ownship_vel, ownship_acc,
                targets, jammers,
                intr_time, sub_dt,
            )

            all_body_dets.extend(body_dets)
            all_tracker_dets.extend(tracker_dets)
            all_tracks.extend(tracks)

        # Deduplicate tracks — keep most recent by track_id
        seen_ids: set = set()
        unique_tracks: List[Track] = []
        for trk in reversed(all_tracks):
            if trk.track_id not in seen_ids:
                seen_ids.add(trk.track_id)
                unique_tracks.append(trk)

        return all_body_dets, all_tracker_dets, unique_tracks, coverage_config

    # ------------------------------------------------------------------ #
    def load_traj_data(self, traj_data: Any, sample_rate: float) -> None:
        """Attach pre-loaded trajectory data."""
        self.traj_data = traj_data
        self.traj_data_sample_rate = sample_rate

    # ------------------------------------------------------------------ #
    def _create_one_step_vars(
        self,
        ownship_data: Any,
        targets_local: Any,
        intr_time: float,
        blue1: Any,
    ) -> tuple:
        """
        Build per-step inputs from scenario objects.

        Returns
        -------
        (jammers, targets, ownship_pos, ownship_vel, ownship_acc,
         ownship_ori, coverage_config)
        """
        from .target import Target
        from .jammers.jammer import Jammer

        # Find matching time index (nearest)
        times = np.asarray(ownship_data.time)
        data_index = int(np.argmin(np.abs(times - intr_time)))

        ownship_pos = np.asarray(blue1.Trajectory.Position, dtype=float)
        ownship_vel = np.asarray(blue1.Trajectory.Velocity, dtype=float)
        ownship_acc = np.asarray(blue1.Trajectory.Acceleration, dtype=float)

        yaw   = float(ownship_data.yaw[data_index])
        pitch = float(ownship_data.pitch[data_index])
        roll  = float(ownship_data.roll[data_index])
        ownship_ori = np.array([yaw, pitch, roll])

        from scipy.spatial.transform import Rotation
        dcm = Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()

        theta_fov = self.params['trackingRadar']['theta_FOV']
        phi_fov   = self.params['trackingRadar']['phi_FOV']
        range_max = self.mpar.tracking_radar.range_max

        coverage_config = {
            'Index':       self.sensor_index,
            'LookAngle':   0,
            'FieldOfView': [theta_fov * 2, phi_fov * 2],
            'ScanLimits':  [0, 0],
            'Range':       range_max,
            'Position':    ownship_pos,
            'Orientation': dcm,
        }

        lat0 = float(ownship_data.lat[data_index])
        lon0 = float(ownship_data.lon[data_index])
        alt0 = float(ownship_data.alt[data_index])

        targets = []
        jammers = []

        for tgt_obj in targets_local:
            pid         = int(tgt_obj.PlatformID)
            rcs_avg     = float(tgt_obj.RCS[data_index])
            swerling    = int(tgt_obj.Swerling[data_index])
            chaff       = int(tgt_obj.Chaff[data_index])

            lat = float(tgt_obj.lat[data_index])
            lon = float(tgt_obj.lon[data_index])
            alt = float(tgt_obj.alt[data_index])

            x_ned, y_ned, z_ned = _geodetic_to_ned(lat, lon, alt, lat0, lon0, alt0)
            pose_local   = np.array([x_ned, y_ned, -z_ned])
            pose_ownship = (dcm @ np.array([x_ned, y_ned, -z_ned]))

            trgt = Target(
                target_id=pid,
                body_position=pose_ownship,
                local_ned_position=pose_local,
                rcs_avg=rcs_avg,
                swerling_type=swerling,
                chaff=chaff,
            )
            targets.append(trgt)

            jammer_params = tgt_obj.JammerParams[data_index]
            if getattr(jammer_params, 'is_jamming', 0) == 1:
                az, el, r = trgt.get_body_spherical()
                jp = dict(vars(jammer_params))
                jp['azimuth_to_body']   = az
                jp['elevation_to_body'] = el
                jp['range_to_body']     = r
                jammers.append(Jammer(jp))

        return (jammers, targets, ownship_pos, ownship_vel,
                ownship_acc, ownship_ori, coverage_config)


# ------------------------------------------------------------------ #
def _geodetic_to_ned(
    lat: float, lon: float, alt: float,
    lat0: float, lon0: float, alt0: float,
) -> Tuple[float, float, float]:
    """
    Approximate geodetic → NED conversion using flat-Earth assumption.
    Replace with pymap3d.geodetic2ned for higher accuracy if available.
    """
    try:
        import pymap3d
        n, e, d = pymap3d.geodetic2ned(lat, lon, alt, lat0, lon0, alt0)
        return float(n), float(e), float(d)
    except ImportError:
        pass

    _R_EARTH = 6_371_000.0
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    x_n = dlat * _R_EARTH
    x_e = dlon * _R_EARTH * math.cos(math.radians(lat0))
    x_d = -(alt - alt0)
    return x_n, x_e, x_d
