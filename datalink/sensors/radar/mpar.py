"""
Multifunction Phased Array Radar (MPAR).

Orchestrates TrackingRadar + GNNTracker, converts body-spherical detections
to local-NED Cartesian, and manages orthogonal-track ghost removal.
"""

from __future__ import annotations
import math
import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Optional, Tuple

from .tracking_radar import TrackingRadar
from .tracking.gnn_tracker import GNNTracker
from .tracking.dbscan_filter import apply_dbscan
from .tracking.cost import calculate_assignment_cost
from .tracking.constvel import set_observer_acceleration
from .detection import Detection
from .track import Track


def _angle2dcm(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """ZYX Euler angles → DCM (NED-to-body).  Angles in radians."""
    return Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()


class MPAR:
    """
    Multifunction Phased Array Radar.

    Parameters
    ----------
    sensor_index : int
    params       : dict  (full params.json contents)
    """

    _ORTH_EPSILON_RAD = math.radians(1.0)

    def __init__(self, sensor_index: int, params: dict) -> None:
        self.sensor_index = sensor_index
        self.params = params

        tp = params['trackerParams']

        self.tracking_radar = TrackingRadar(params)
        self.tracker = GNNTracker(
            max_tracks=tp.get('max_tracks', 100),
            gate=tp.get('gate', 1e4),
            confirm_threshold=tuple(tp.get('confirm_threshold', [2, 3])),
            delete_threshold=tuple(tp.get('delete_threshold', [3, 4])),
            has_cost_matrix_input=tp.get('has_cost_matrix', False),
        )
        self.tracking_gate = tp.get('gate', 1e4)
        self.has_cost_matrix = tp.get('has_cost_matrix', False)
        self.do_dbscan = params['scenarioParams'].get('doDBSCAN', False)
        self.dbscan_epsilon = params['clutterParams'].get('dbscan_epsilon', 500.0)

        self.tracks:     List[Track] = []
        self.all_tracks: List[Track] = []

        # Orthogonal ghost removal (last 3 steps)
        self._orth_history: list = []
        self._step_n: int = 0

        # Ownship state
        self._ownship_position    = np.zeros(3)
        self._ownship_orientation = np.zeros(3)   # [yaw, pitch, roll] rad
        self._ownship_velocity    = np.zeros(3)

        # Bias accumulator (for spherical → cartesian conversion logging)
        self.mu_s: list = []

    # ------------------------------------------------------------------ #
    def update_sensor(
        self,
        ownship_pos: np.ndarray,
        ownship_ori: np.ndarray,
        ownship_vel: np.ndarray,
        ownship_acc: np.ndarray,
        targets: list,
        jammers: list,
        sim_time: float,
        delta_time: float,
    ) -> Tuple[List[Detection], List[Detection], List[Track]]:
        """
        Run one sensor update step.

        Parameters
        ----------
        ownship_pos : (3,) NED position [m]
        ownship_ori : (3,) [yaw, pitch, roll] in radians
        ownship_vel : (3,) NED velocity [m/s]
        ownship_acc : (3,) NED acceleration [m/s²]
        targets     : list of Target
        jammers     : list of Jammer
        sim_time    : float [s]
        delta_time  : float [s]

        Returns
        -------
        (body_spherical_detections, tracker_detections, confirmed_tracks)
        """
        # Update module-level observer acceleration used by constvel
        set_observer_acceleration(np.asarray(ownship_acc, dtype=float))

        self._ownship_position    = np.asarray(ownship_pos, dtype=float).ravel()
        self._ownship_orientation = np.asarray(ownship_ori, dtype=float).ravel()
        self._ownship_velocity    = np.asarray(ownship_vel, dtype=float).ravel()

        # Generate raw (body-spherical) detections from radar
        local_ned_dets, body_sph_dets, _ = self._get_detections(
            targets, jammers, sim_time, delta_time)

        # Optionally cluster with DBSCAN
        if self.do_dbscan:
            global_dets, tracker_dets = apply_dbscan(
                local_ned_dets, body_sph_dets, self.dbscan_epsilon)
        else:
            global_dets   = local_ned_dets
            tracker_dets  = body_sph_dets

        # Update GNN tracker
        if self.tracker.is_locked() or global_dets:
            detectable_ids = self._get_detectable_ids(sim_time)

            if self.has_cost_matrix and self.all_tracks:
                cost_matrix = calculate_assignment_cost(
                    global_dets, self.all_tracks, self.tracking_gate)
                confirmed, all_trks = self.tracker.step(
                    tracker_dets, sim_time,
                    cost_matrix=cost_matrix,
                    detectable_ids=detectable_ids,
                    dt=delta_time,
                )
            else:
                confirmed, all_trks = self.tracker.step(
                    tracker_dets, sim_time,
                    detectable_ids=detectable_ids,
                    dt=delta_time,
                )

            self.all_tracks = all_trks
            self.tracks = confirmed
        else:
            self.tracks = []

        # Filter confirmed tracks to those in detectable set
        detectable_ids = self._get_detectable_ids(sim_time)
        visible_tracks = [t for t in self.tracks if t.track_id in detectable_ids]

        # Adaptive process-noise update
        self._update_process_noise()

        # Orthogonal ghost removal
        if self._step_n > 0:
            self._calculate_orthogonal_tracks()
            self._delete_orthogonal_tracks()

        self._step_n += 1
        return body_sph_dets, tracker_dets, visible_tracks

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _get_detections(
        self,
        targets: list,
        jammers: list,
        sim_time: float,
        delta_time: float,
    ) -> Tuple[List[Detection], List[Detection], float]:
        """Call TrackingRadar and convert raw tuples to Detection objects."""
        altitude = float(self._ownship_position[2])
        yaw, pitch, roll = self._ownship_orientation
        dcm_ned2body = _angle2dcm(yaw, pitch, roll)

        raw_dets, time_stamp = self.tracking_radar.calculate_pseudo_measurements(
            altitude, dcm_ned2body, targets, jammers, sim_time, delta_time)

        local_ned_dets  = self._body_sph_to_ned_cartesian(raw_dets, dcm_ned2body)
        body_sph_dets   = self._body_sph_to_body_sph_detection(raw_dets, dcm_ned2body)
        return local_ned_dets, body_sph_dets, time_stamp

    # ------------------------------------------------------------------
    def _body_sph_to_ned_cartesian(
        self,
        raw_dets: list,
        dcm_ned2body: np.ndarray,
    ) -> List[Detection]:
        """Convert raw body-spherical tuples → NED-cartesian Detection objects."""
        inv_dcm = dcm_ned2body.T
        out = []
        for tup in raw_dets:
            t, theta, phi, rho, snr, noise_mat, tid = tup
            body_sph  = np.array([theta, phi, rho])
            body_cart, cart_noise = self._sph_to_cart_with_noise(
                body_sph, noise_mat)
            ned_cart  = inv_dcm @ body_cart
            ned_noise = inv_dcm @ cart_noise @ inv_dcm.T

            out.append(Detection(
                time=t,
                measurement=ned_cart,
                sensor_index=self.sensor_index,
                measurement_noise=ned_noise,
                measurement_parameters={
                    'Frame': 'rectangular',
                    'OriginPosition': np.zeros(3),
                    'OriginVelocity': np.zeros(3),
                    'Orientation': np.eye(3),
                    'IsParentToChild': True,
                },
                object_attributes={'SNR': snr, 'TargetIndex': tid},
            ))
        return out

    def _body_sph_to_body_sph_detection(
        self,
        raw_dets: list,
        dcm_ned2body: np.ndarray,
    ) -> List[Detection]:
        """Convert raw body-spherical tuples → body-spherical Detection objects (degrees)."""
        out = []
        for tup in raw_dets:
            t, theta, phi, rho, snr, noise_mat, tid = tup
            # Noise matrix diagonal is in radians; convert to degrees, then square
            noise_deg = noise_mat.copy().astype(float)
            noise_deg[0, 0] = math.degrees(noise_mat[0, 0])
            noise_deg[1, 1] = math.degrees(noise_mat[1, 1])
            noise_deg = noise_deg ** 2

            out.append(Detection(
                time=t,
                measurement=np.array([math.degrees(theta),
                                       math.degrees(phi),
                                       rho]),
                sensor_index=self.sensor_index,
                measurement_noise=noise_deg,
                measurement_parameters={
                    'Frame': 'spherical',
                    'OriginPosition': np.zeros(3),
                    'Orientation': dcm_ned2body,
                    'HasVelocity': False,
                    'IsParentToChild': True,
                },
                object_attributes={'SNR': snr, 'TargetIndex': tid},
            ))
        return out

    # ------------------------------------------------------------------
    def _sph_to_cart_with_noise(
        self,
        sph: np.ndarray,
        noise_sph: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert body-spherical measurement + noise to Cartesian.

        Uses the unbiased estimator with debiasing factors λ_az, λ_el.
        """
        az, el, r = sph
        sigma_az = noise_sph[0, 0]
        sigma_el = noise_sph[1, 1]
        sigma_r  = noise_sph[2, 2]

        lam_az  = math.exp(-(sigma_az**2) / 2.0)
        lam_el  = math.exp(-(sigma_el**2) / 2.0)
        lam_az2 = math.exp(-2.0 * sigma_az**2)
        lam_el2 = math.exp(-2.0 * sigma_el**2)

        x = (1.0 / lam_az) * (1.0 / lam_el) * r * math.cos(az) * math.cos(el)
        y = (1.0 / lam_az) * (1.0 / lam_el) * r * math.sin(az) * math.cos(el)
        z = (1.0 / lam_el) * r * math.sin(el)
        cart = np.array([x, y, z])

        # Log bias
        mu1 = ((1.0/(lam_az*lam_el)) - lam_az*lam_el) * r * math.cos(az) * math.cos(el)
        mu2 = ((1.0/(lam_az*lam_el)) - lam_az*lam_el) * r * math.sin(az) * math.cos(el)
        mu3 = ((1.0/lam_el) - lam_el) * r * math.sin(el)
        self.mu_s.append([mu1, mu2, mu3])

        # Covariance
        r2  = r**2
        sr2 = sigma_r**2
        R11 = -(lam_az*lam_el*r*math.cos(az)*math.cos(el))**2 + \
              0.25*(r2+sr2)*(1+lam_az2*math.cos(2*az))*(1+lam_el2*math.cos(2*el))
        R22 = -(lam_az*lam_el*r*math.sin(az)*math.cos(el))**2 + \
              0.25*(r2+sr2)*(1-lam_az2*math.cos(2*az))*(1+lam_el2*math.cos(2*el))
        R33 = -(lam_el*r*math.sin(el))**2 + \
              0.5*(r2+sr2)*(1-lam_el2*math.cos(2*el))
        R12 = -((lam_az*lam_el*r*math.cos(el))**2)*math.sin(az)*math.cos(az) + \
              0.25*(r2+sr2)*lam_az2*math.sin(2*az)*(1+lam_el2*math.cos(2*el))
        R13 = -((lam_el*r)**2)*lam_az*math.cos(az)*math.sin(el)*math.cos(el) + \
              0.5*(r2+sr2)*lam_az*lam_el2*math.cos(az)*math.sin(2*el)
        R23 = -((lam_el*r)**2)*lam_az*math.sin(az)*math.sin(el)*math.cos(el) + \
              0.5*(r2+sr2)*lam_az*lam_el2*math.sin(az)*math.sin(2*el)

        cov = np.array([[R11, R12, R13],
                        [R12, R22, R23],
                        [R13, R23, R33]])
        return cart, cov

    # ------------------------------------------------------------------
    def _get_detectable_ids(self, sim_time: float) -> List[int]:
        """Return track IDs whose predicted position falls within FOV."""
        yaw, pitch, roll = self._ownship_orientation
        dcm = _angle2dcm(yaw, pitch, roll)

        az_min = self.tracking_radar.theta_FOV_min * (1 - 0.1 * math.copysign(1, self.tracking_radar.theta_FOV_min))
        az_max = self.tracking_radar.theta_FOV_max * (1 + 0.1 * math.copysign(1, self.tracking_radar.theta_FOV_max))

        detectable = []
        for trk in self.all_tracks:
            pos = trk.state[[0, 2, 4]]
            body_pos = dcm @ pos
            az = math.atan2(body_pos[1], body_pos[0])
            if az_min <= az <= az_max:
                detectable.append(trk.track_id)
        return detectable

    # ------------------------------------------------------------------
    def _update_process_noise(self) -> None:
        """Adaptive Q update based on velocity change between steps."""
        # Requires old_all_tracks storage — keep lightweight version
        if not hasattr(self, '_prev_all_tracks'):
            self._prev_all_tracks: List[Track] = []

        prev_map = {t.track_id: t for t in self._prev_all_tracks}
        for trk in self.all_tracks:
            tid = trk.track_id
            if tid not in prev_map:
                continue
            old = prev_map[tid]
            dt = trk.update_time - old.update_time
            if dt <= 0:
                continue
            dot_state = (trk.state - old.state) / dt
            age = trk.age
            if age < 5:
                k_Q = 20000
            elif age < 8:
                k_Q = 2000
            elif age < 15:
                k_Q = 500
            else:
                k_Q = 200
            Q = np.diag([
                abs(dot_state[1]) * k_Q,
                abs(dot_state[3]) * k_Q,
                abs(dot_state[5]) * k_Q,
            ])
            self.tracker.set_process_noise(tid, Q)

        self._prev_all_tracks = list(self.all_tracks)

    # ------------------------------------------------------------------
    def _calculate_orthogonal_tracks(self) -> None:
        """Detect tracks whose velocity is orthogonal to ownship velocity."""
        ov = self._ownship_velocity
        norm_ov = np.linalg.norm(ov)
        if norm_ov < 1e-6:
            self._orth_history.append(set())
            return

        ortho = set()
        for trk in self.tracks:
            tv = trk.velocity
            norm_tv = np.linalg.norm(tv)
            if norm_tv < 1e-6:
                continue
            cos_a = np.dot(ov, tv) / (norm_ov * norm_tv)
            cos_a = max(-1.0, min(1.0, cos_a))
            angle = math.acos(cos_a)
            alpha = abs(math.pi / 2 - abs(angle))
            if alpha <= self._ORTH_EPSILON_RAD:
                ortho.add(trk.track_id)

        self._orth_history.append(ortho)
        if len(self._orth_history) > 3:
            self._orth_history = self._orth_history[-3:]

    def _delete_orthogonal_tracks(self) -> None:
        """Delete tracks that appeared orthogonal in 3 consecutive steps."""
        if len(self._orth_history) < 3:
            return
        first, second, third = self._orth_history[-3], self._orth_history[-2], self._orth_history[-1]
        if not (first and second and third):
            return
        to_delete = first & second & third
        for tid in to_delete:
            self.tracker.delete_track(tid)
        self.all_tracks = [t for t in self.all_tracks if t.track_id not in to_delete]
