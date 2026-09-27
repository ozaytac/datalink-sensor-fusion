"""
Global Nearest Neighbour (GNN) multi-target tracker.

Combines Hungarian-algorithm measurement-to-track assignment with a bank
of constant-velocity EKFs (one per track).
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Optional
from datalink.sensors.radar.detection import Detection
from datalink.sensors.radar.track import Track
from .cvekf import CVEKF, init_cv_ekf


class GNNTracker:
    """
    Global Nearest Neighbour tracker.

    Manages track lifecycle (birth → tentative → confirmed → deletion)
    using a per-track CVEKF and Munkres (Hungarian) assignment.

    Parameters
    ----------
    max_tracks            : int   — maximum number of simultaneous tracks
    gate                  : float — distance gate [m]; > gate → no assignment
    confirm_threshold     : (int, int) — (n_hits, n_steps) to confirm
    delete_threshold      : (int, int) — (n_misses, n_steps) to delete
    has_cost_matrix_input : bool  — if True, accept external cost matrix
    """

    def __init__(
        self,
        max_tracks: int = 100,
        gate: float = 1e4,
        confirm_threshold: tuple[int, int] = (2, 3),
        delete_threshold:  tuple[int, int] = (3, 4),
        has_cost_matrix_input: bool = False,
    ):
        self.max_tracks            = max_tracks
        self.gate                  = gate
        self.confirm_threshold     = confirm_threshold   # (M, N) → M hits in N steps
        self.delete_threshold      = delete_threshold    # (M, N) → M misses in N steps
        self.has_cost_matrix_input = has_cost_matrix_input

        self._next_id: int = 1
        self._filters: dict[int, CVEKF] = {}       # track_id → CVEKF
        self._tracks:  dict[int, Track] = {}        # track_id → Track
        self._recent_hits:   dict[int, list[int]] = {}   # hit/miss history
        self._is_locked: bool = False

    # ------------------------------------------------------------------ #
    def is_locked(self) -> bool:
        return self._is_locked

    # ------------------------------------------------------------------ #
    def step(
        self,
        detections: List[Detection],
        sim_time: float,
        cost_matrix: Optional[np.ndarray] = None,
        detectable_ids: Optional[List[int]] = None,
        dt: float = 0.2,
    ) -> tuple[List[Track], List[Track]]:
        """
        Process one tracker step.

        Parameters
        ----------
        detections     : List[Detection]
        sim_time       : float — current simulation time [s]
        cost_matrix    : ndarray (n_tracks, n_dets), optional
        detectable_ids : list of track IDs in sensor FOV, optional
        dt             : float — time step for prediction [s]

        Returns
        -------
        (confirmed_tracks, all_tracks)
        """
        if not self._is_locked and not detections:
            return [], list(self._tracks.values())
        self._is_locked = True

        all_track_ids = list(self._tracks.keys())

        # --- 1. Predict all existing tracks ---
        for tid, ekf in self._filters.items():
            ekf.predict(dt)
            trk = self._tracks[tid]
            trk.state            = ekf.state.copy()
            trk.state_covariance = ekf.state_covariance.copy()
            trk.update_time      = sim_time

        # --- 2. Build cost matrix ---
        tracks_list = [self._tracks[tid] for tid in all_track_ids]
        if cost_matrix is None and tracks_list and detections:
            from .cost import calculate_assignment_cost
            cost_matrix = calculate_assignment_cost(
                detections, tracks_list, self.gate)

        # --- 3. Hungarian assignment ---
        assigned_tracks = set()
        assigned_dets   = set()
        if cost_matrix is not None and cost_matrix.size > 0:
            # Only assign within gate (finite cost)
            C = cost_matrix.copy()
            C[C == np.inf] = 1e18
            row_ind, col_ind = linear_sum_assignment(C)
            for r, c in zip(row_ind, col_ind):
                if r < len(tracks_list) and c < len(detections):
                    if cost_matrix[r, c] < self.gate:
                        tid = all_track_ids[r]
                        self._update_track(tid, detections[c], sim_time)
                        self._record_hit(tid, hit=True)
                        assigned_tracks.add(tid)
                        assigned_dets.add(c)

        # --- 4. Record misses ---
        for tid in all_track_ids:
            if tid not in assigned_tracks:
                self._record_hit(tid, hit=False)

        # --- 5. Birth new tracks for unassigned detections ---
        for c, det in enumerate(detections):
            if c not in assigned_dets and len(self._tracks) < self.max_tracks:
                self._birth_track(det, sim_time)

        # --- 6. Manage confirmation and deletion ---
        to_delete = []
        for tid in list(self._tracks.keys()):
            trk = self._tracks[tid]
            history = self._recent_hits[tid]
            hits   = sum(history)
            misses = len(history) - hits

            # Confirm
            M_conf, N_conf = self.confirm_threshold
            if not trk.is_confirmed and sum(history[-N_conf:]) >= M_conf:
                trk.is_confirmed = True

            # Delete
            M_del, N_del = self.delete_threshold
            if sum(1 - np.array(history[-N_del:])) >= M_del:
                to_delete.append(tid)

        for tid in to_delete:
            self._delete_track(tid)

        # --- 7. Collect outputs ---
        all_tracks = list(self._tracks.values())
        confirmed  = [t for t in all_tracks if t.is_confirmed]
        return confirmed, all_tracks

    # ------------------------------------------------------------------ #
    def delete_track(self, track_id: int) -> None:
        """Externally force-delete a track (e.g., orthogonal ghost removal)."""
        self._delete_track(track_id)

    def set_process_noise(self, track_id: int, Q: np.ndarray) -> None:
        """Override process noise for a specific track."""
        if track_id in self._filters:
            self._filters[track_id].process_noise = np.asarray(Q, dtype=float)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _birth_track(self, det: Detection, sim_time: float) -> None:
        tid  = self._next_id
        self._next_id += 1
        ekf  = init_cv_ekf(det)
        trk  = Track(
            track_id=tid,
            state=ekf.state.copy(),
            state_covariance=ekf.state_covariance.copy(),
            update_time=sim_time,
            age=1,
            is_confirmed=False,
            object_attributes=det.object_attributes.copy(),
        )
        self._filters[tid]     = ekf
        self._tracks[tid]      = trk
        self._recent_hits[tid] = [1]

    def _update_track(self, tid: int, det: Detection, sim_time: float) -> None:
        ekf = self._filters[tid]
        ekf.update(det.measurement[:3], det.measurement_noise)
        trk = self._tracks[tid]
        trk.state            = ekf.state.copy()
        trk.state_covariance = ekf.state_covariance.copy()
        trk.update_time      = sim_time
        trk.age             += 1
        trk.object_attributes = det.object_attributes.copy()

    def _record_hit(self, tid: int, hit: bool) -> None:
        self._recent_hits[tid].append(1 if hit else 0)
        # Keep only the last max(N_conf, N_del) entries
        N = max(self.confirm_threshold[1], self.delete_threshold[1])
        if len(self._recent_hits[tid]) > N:
            self._recent_hits[tid] = self._recent_hits[tid][-N:]

    def _delete_track(self, tid: int) -> None:
        self._filters.pop(tid, None)
        self._tracks.pop(tid,  None)
        self._recent_hits.pop(tid, None)
