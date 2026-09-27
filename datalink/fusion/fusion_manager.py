"""
Track-to-track fusion manager (Radar + IR + DataLink).

Implements heterogeneous sensor fusion using:
  - t2tf_msc  : Radar (Cartesian) ← → IR (MSC) via LMMSE
  - t2tf_lmmse: Radar (Cartesian) ← → IR (spherical, 4-D) via LMMSE
  - linear_kalman : homogeneous Kalman update (e.g. DataLink)
  - constvel_model: CV prediction for time alignment
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from datalink.transformations import (
    cartesian_to_msc,
    cartesian_to_msc_gradient,
    cart2sph_gradient,
)


# ------------------------------------------------------------------ #
# Small track struct used internally
# ------------------------------------------------------------------ #

@dataclass
class FusionEntry:
    """One entry in the FusionManager table."""
    fusion_id: int
    radar_id:  int = -1
    ir_id:     int = -1
    dl_id:     int = -1

    radar_state: Optional[np.ndarray] = None
    radar_cov:   Optional[np.ndarray] = None
    radar_update_time: float = -1.0
    radar_target_index: int = -1

    ir_state: Optional[np.ndarray] = None
    ir_cov:   Optional[np.ndarray] = None
    ir_update_time: float = -1.0
    ir_target_index: int = -1

    dl_state: Optional[np.ndarray] = None
    dl_cov:   Optional[np.ndarray] = None
    dl_update_time: float = -1.0
    dl_target_index: int = -1

    fused_state: Optional[np.ndarray] = None
    fused_cov:   Optional[np.ndarray] = None
    fusion_update_time: float = -1.0

    fused_state_ir_only: Optional[np.ndarray] = None
    fused_cov_ir_only:   Optional[np.ndarray] = None
    fusion_update_time_ir_only: float = -1.0

    cross_cov: Optional[np.ndarray] = None
    target_index: int = -1

    # Update flags: each is a list of 0/1 over history
    radar_updates:  List[int] = field(default_factory=lambda: [0])
    ir_updates:     List[int] = field(default_factory=lambda: [0])
    fusion_updates: List[int] = field(default_factory=lambda: [0])
    dl_updates:     List[int] = field(default_factory=lambda: [0])
    virtual_updates: List[int] = field(default_factory=lambda: [0])

    radar_ids_history: List[int] = field(default_factory=lambda: [-1])
    ir_ids_history:    List[int] = field(default_factory=lambda: [-1])
    dl_ids_history:    List[int] = field(default_factory=lambda: [-1])

    radar_state_history: List[Optional[np.ndarray]] = field(default_factory=lambda: [None])
    ir_state_history:    List[Optional[np.ndarray]] = field(default_factory=lambda: [None])

    _HISTORY = 50   # max entries to keep


_HISTORY = 50


def _new_entry(fusion_id: int) -> FusionEntry:
    return FusionEntry(
        fusion_id=fusion_id,
        radar_cov=np.eye(6) * 1e3,
        dl_cov=np.eye(6) * 1e3,
    )


# ------------------------------------------------------------------ #

class FusionManager:
    """
    Multi-sensor track-to-track fusion manager.

    Manages a table of FusionEntry objects, each combining information
    from radar (Cartesian), IR (MSC), and DataLink (Cartesian) sensors.

    Parameters
    ----------
    traj   : trajectory data (passed through, stored for reference)
    params : full params dict
    """

    # Deletion threshold: (hits, window)
    _DELETE_EPSILON = (0, 10)

    def __init__(self, traj: Any, params: dict) -> None:
        self.traj   = traj
        self.params = params.get('scenarioParams', params)

        self._entries: List[FusionEntry] = []
        self._next_fusion_id: int = 1

        # Alias properties (used by SimulationSummary)
        self._sim_time: float = 0.0

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    @property
    def fusion_ids(self):       return [e.fusion_id for e in self._entries]
    @property
    def radar_ids(self):        return [e.radar_id  for e in self._entries]
    @property
    def ir_ids(self):           return [e.ir_id     for e in self._entries]
    @property
    def datalink_ids(self):     return [e.dl_id     for e in self._entries]
    @property
    def fused_states(self):     return [e.fused_state for e in self._entries]
    @property
    def fused_covs(self):       return [e.fused_cov   for e in self._entries]
    @property
    def fusion_update_times(self): return [e.fusion_update_time for e in self._entries]

    # ------------------------------------------------------------------ #
    def update_manager(
        self,
        confirmed_radar_tracks: List[Any],
        confirmed_ir_tracks:    List[Any],
        time: float,
    ) -> None:
        """
        Fuse radar and IR confirmed tracks.

        Parameters
        ----------
        confirmed_radar_tracks : list of Track (Cartesian, 6-D state)
        confirmed_ir_tracks    : list of Track (MSC, 6-D state)
        time                   : current simulation time [s]
        """
        self._sim_time = time
        self._init_new_timestep_variables()

        radar_tracks = confirmed_radar_tracks
        ir_tracks    = confirmed_ir_tracks

        assigned_ids: list = []

        # --- Radar → IR assignment ---
        if radar_tracks and self._entries:
            fusion_ir_ids = [e.ir_id for e in self._entries if e.ir_id != -1]
            if fusion_ir_ids:
                fusion_ir_tracks = self._build_fusion_ir_tracks()
                if fusion_ir_tracks:
                    asgn = self._get_assignments(radar_tracks, fusion_ir_tracks)
                    assigned_ids.extend(asgn)

        # --- IR → Radar assignment ---
        if ir_tracks and self._entries:
            fusion_radar_ids = [e.radar_id for e in self._entries if e.radar_id != -1]
            if fusion_radar_ids:
                fusion_radar_tracks = self._build_fusion_radar_tracks()
                if fusion_radar_tracks:
                    asgn = self._get_assignments(fusion_radar_tracks, ir_tracks)
                    assigned_ids.extend(asgn)

        # Unpack (radar_id, ir_id) pairs
        if assigned_ids:
            arr = np.array(assigned_ids, dtype=int).reshape(-1, 2)
            arr = np.unique(arr, axis=0)
        else:
            arr = np.empty((0, 2), dtype=int)

        radar_ids_all = set(t.track_id for t in radar_tracks)
        ir_ids_all    = set(t.track_id for t in ir_tracks)
        assigned_radar = set(arr[:, 0].tolist()) if len(arr) else set()
        assigned_ir    = set(arr[:, 1].tolist()) if len(arr) else set()
        unassigned_radar = radar_ids_all - assigned_radar
        unassigned_ir    = ir_ids_all    - assigned_ir

        # Update assigned pairs
        for radar_id, ir_id in arr.tolist():
            fid = self._get_or_create_fusion_id(radar_id, ir_id)
            self._append_track_to_manager(fid, radar_id, ir_id,
                                          radar_tracks, ir_tracks)

        # Update unassigned radar tracks
        for rid in unassigned_radar:
            fid = self._get_or_create_fusion_id(rid, -1)
            self._append_track_to_manager(fid, rid, -1, radar_tracks, ir_tracks)

        # Update unassigned IR tracks
        for iid in unassigned_ir:
            fid = self._get_or_create_fusion_id(-1, iid)
            self._append_track_to_manager(fid, -1, iid, radar_tracks, ir_tracks)

        # Handle entries not updated this step
        updated_fids = set(e.fusion_id for e in self._entries
                           if e.radar_updates[-1] or e.ir_updates[-1])
        for e in self._entries:
            if e.fusion_id not in updated_fids:
                self._append_track_to_manager(e.fusion_id, -1, -1, [], [])

        self._remove_tracks_from_manager()

    # ------------------------------------------------------------------ #
    def update_manager_with_datalink(self, datalink_tracks: List[Any]) -> None:
        """Fuse DataLink tracks into existing fusion entries."""
        if not datalink_tracks:
            return

        for dl_trk in datalink_tracks:
            # Find best matching fusion entry by position distance
            best_idx = self._find_nearest_entry(dl_trk)
            if best_idx is None:
                # Create new entry for unmatched DataLink track
                fid = self._next_fusion_id
                self._next_fusion_id += 1
                entry = _new_entry(fid)
                self._entries.append(entry)
                best_idx = len(self._entries) - 1

            e = self._entries[best_idx]
            dl_state = np.asarray(dl_trk.state, dtype=float)
            dl_cov   = np.asarray(dl_trk.state_covariance, dtype=float)
            dl_time  = float(dl_trk.update_time)

            if e.fused_state is not None:
                # Predict DataLink track to fusion time, then linear Kalman
                dt = e.fusion_update_time - dl_time
                pred_state, pred_cov = self._constvel_model(dl_state, dl_cov, dt)
                fused_s, fused_c = self._linear_kalman(
                    e.fused_state, e.fused_cov, pred_state, pred_cov)
                e.fused_state = fused_s
                e.fused_cov   = fused_c

            e.dl_id           = getattr(dl_trk, 'track_id', -1)
            e.dl_state        = dl_state
            e.dl_cov          = dl_cov
            e.dl_update_time  = dl_time
            e.dl_target_index = getattr(dl_trk, 'object_attributes', {}).get(
                'TargetIndex', -1)
            if e.dl_updates:
                e.dl_updates[-1] = 1
            e.dl_ids_history[-1] = e.dl_id

    # ------------------------------------------------------------------ #
    def get_fused_tracks(self) -> List[Dict]:
        """Return all entries that have a valid fused state."""
        result = []
        for e in self._entries:
            if e.fused_state is not None:
                result.append({
                    'fusion_id':          e.fusion_id,
                    'state':              e.fused_state.copy(),
                    'state_covariance':   e.fused_cov.copy()
                                          if e.fused_cov is not None
                                          else np.eye(6),
                    'update_time':        e.fusion_update_time,
                    'radar_id':           e.radar_id,
                    'ir_id':              e.ir_id,
                    'dl_id':              e.dl_id,
                    'target_index':       e.target_index,
                    'is_fused':           e.radar_id != -1 and e.ir_id != -1,
                })
            elif e.fused_state_ir_only is not None:
                result.append({
                    'fusion_id':        e.fusion_id,
                    'state':            e.fused_state_ir_only.copy(),
                    'state_covariance': e.fused_cov_ir_only.copy()
                                        if e.fused_cov_ir_only is not None
                                        else np.eye(6),
                    'update_time':      e.fusion_update_time_ir_only,
                    'radar_id':         e.radar_id,
                    'ir_id':            e.ir_id,
                    'dl_id':            e.dl_id,
                    'target_index':     e.target_index,
                    'is_fused':         False,
                    'is_ir_only':       True,
                })
        return result

    # ------------------------------------------------------------------ #
    # Private — assignment
    # ------------------------------------------------------------------ #

    def _get_assignments(
        self, tracks_a: List[Any], tracks_b: List[Any]
    ) -> List[Tuple[int, int]]:
        """
        Heterogeneous assignment using Hungarian algorithm on an
        approximate Mahalanobis/angular distance cost matrix.

        tracks_a : Cartesian-frame tracks (radar)
        tracks_b : MSC-frame tracks (IR) or vice-versa
        """
        if not tracks_a or not tracks_b:
            return []

        na, nb = len(tracks_a), len(tracks_b)
        D = np.full((na, nb), np.inf)
        EPSILON = 200.0

        for i, ta in enumerate(tracks_a):
            for j, tb in enumerate(tracks_b):
                D[i, j] = self._assignment_cost(ta, tb)

        # Replace inf with large value for Hungarian solver
        D_solve = np.where(np.isinf(D), 1e18, D)
        rows, cols = linear_sum_assignment(D_solve)

        pairs = []
        for r, c in zip(rows, cols):
            if D[r, c] <= EPSILON:
                pairs.append((tracks_a[r].track_id, tracks_b[c].track_id))
        return pairs

    def _assignment_cost(self, trk_a: Any, trk_b: Any) -> float:
        """
        Angular distance between radar Cartesian track and IR MSC track.
        Predicts the earlier track to the later update time.
        """
        t_a = getattr(trk_a, 'update_time', 0.0)
        t_b = getattr(trk_b, 'update_time', 0.0)

        state_a = np.asarray(trk_a.state, dtype=float)
        state_b = np.asarray(trk_b.state, dtype=float)
        cov_a   = np.asarray(trk_a.state_covariance, dtype=float)
        cov_b   = np.asarray(trk_b.state_covariance, dtype=float)

        if t_a < t_b:
            state_a, cov_a = self._constvel_model(state_a, cov_a, t_b - t_a)
        elif t_b < t_a:
            state_b, cov_b = self._constvel_model(state_b, cov_b, t_a - t_b)

        # Convert Cartesian state_a to spherical for comparison with MSC state_b
        x, vx, y, vy, z, vz = state_a
        r = math.sqrt(x*x + y*y + z*z)
        if r < 1e-9:
            return np.inf
        az_a = math.atan2(y, x)
        el_a = math.atan2(-z, math.sqrt(x*x + y*y))

        az_b = float(state_b[0])
        el_b = -float(state_b[2])   # MSC sign convention

        d_az = abs(az_a - az_b)
        d_el = abs(el_a - el_b)
        return math.sqrt(d_az**2 + d_el**2)

    # ------------------------------------------------------------------ #
    # Private — track table management
    # ------------------------------------------------------------------ #

    def _init_new_timestep_variables(self) -> None:
        for e in self._entries:
            e.radar_updates.append(0)
            e.ir_updates.append(0)
            e.fusion_updates.append(0)
            e.dl_updates.append(0)
            e.virtual_updates.append(0)
            e.radar_ids_history.append(-1)
            e.ir_ids_history.append(-1)
            e.dl_ids_history.append(-1)
            e.radar_state_history.append(None)
            e.ir_state_history.append(None)

    def _get_or_create_fusion_id(self, radar_id: int, ir_id: int) -> int:
        """Find existing fusion entry or allocate a new one."""
        for e in self._entries:
            if radar_id != -1 and e.radar_id == radar_id:
                return e.fusion_id
            if ir_id != -1 and e.ir_id == ir_id:
                return e.fusion_id
        fid = self._next_fusion_id
        self._next_fusion_id += 1
        return fid

    def _find_entry(self, fusion_id: int) -> Optional[FusionEntry]:
        for e in self._entries:
            if e.fusion_id == fusion_id:
                return e
        return None

    def _find_nearest_entry(self, dl_trk: Any) -> Optional[int]:
        """Return index of fusion entry whose fused state is closest to dl_trk."""
        dl_pos = np.asarray(dl_trk.state, dtype=float)[[0, 2, 4]]
        best_dist = 50e3   # 50 km gate
        best_idx  = None
        for i, e in enumerate(self._entries):
            if e.fused_state is not None:
                pos = e.fused_state[[0, 2, 4]]
                d = float(np.linalg.norm(pos - dl_pos))
                if d < best_dist:
                    best_dist = d
                    best_idx  = i
        return best_idx

    def _build_fusion_ir_tracks(self) -> List[Any]:
        """Build track-like objects from IR states stored in manager."""
        from datalink.sensors.radar.track import Track
        result = []
        for e in self._entries:
            if e.ir_id != -1 and e.ir_state is not None:
                if len(e.ir_updates) > 1 and e.ir_updates[-2]:
                    result.append(Track(
                        track_id=e.ir_id,
                        state=e.ir_state.copy(),
                        state_covariance=e.ir_cov.copy(),
                        update_time=e.ir_update_time,
                        object_attributes={'TargetIndex': e.ir_target_index},
                    ))
        return result

    def _build_fusion_radar_tracks(self) -> List[Any]:
        from datalink.sensors.radar.track import Track
        result = []
        for e in self._entries:
            if e.radar_id != -1 and e.radar_state is not None:
                if len(e.radar_updates) > 1 and e.radar_updates[-2]:
                    result.append(Track(
                        track_id=e.radar_id,
                        state=e.radar_state.copy(),
                        state_covariance=e.radar_cov.copy(),
                        update_time=e.radar_update_time,
                        object_attributes={'TargetIndex': e.radar_target_index},
                    ))
        return result

    def _append_track_to_manager(
        self,
        fusion_id: int,
        radar_id:  int,
        ir_id:     int,
        radar_tracks: List[Any],
        ir_tracks:    List[Any],
    ) -> None:
        entry = self._find_entry(fusion_id)
        if entry is None:
            entry = _new_entry(fusion_id)
            self._entries.append(entry)
            entry.radar_updates   = [0, 0]
            entry.ir_updates      = [0, 0]
            entry.fusion_updates  = [0, 0]
            entry.dl_updates      = [0, 0]
            entry.virtual_updates = [0, 0]
            entry.radar_ids_history = [-1, -1]
            entry.ir_ids_history    = [-1, -1]
            entry.dl_ids_history    = [-1, -1]
            entry.radar_state_history = [None, None]
            entry.ir_state_history    = [None, None]

        e = entry
        radar_update = 0
        ir_update    = 0
        virtual_update = 0

        # ── Radar ──
        if radar_id == -1:
            e.radar_state = None
            e.radar_cov   = None
            e.radar_update_time = -1.0
        else:
            rtrk = next((t for t in radar_tracks if t.track_id == radar_id), None)
            if rtrk is not None:
                radar_update = 1
                e.radar_state = np.asarray(rtrk.state, dtype=float).copy()
                e.radar_cov   = np.asarray(rtrk.state_covariance, dtype=float).copy()
                e.radar_update_time = float(rtrk.update_time)
                e.radar_target_index = rtrk.object_attributes.get('TargetIndex', -1)
                e.target_index = e.radar_target_index
                if e.radar_state_history:
                    e.radar_state_history[-1] = e.radar_state.copy()

        # ── IR ──
        if ir_id == -1:
            e.ir_state = None
            e.ir_cov   = None
            e.ir_update_time = -1.0
        else:
            itrk = next((t for t in ir_tracks if t.track_id == ir_id), None)
            if itrk is not None:
                ir_update = 1
                e.ir_state = np.asarray(itrk.state, dtype=float).copy()
                e.ir_cov   = np.asarray(itrk.state_covariance, dtype=float).copy()
                e.ir_update_time = float(itrk.update_time)
                e.ir_target_index = itrk.object_attributes.get('TargetIndex', -1)
                e.target_index = e.ir_target_index
                if e.ir_state_history:
                    e.ir_state_history[-1] = e.ir_state.copy()

        e.fusion_id = fusion_id
        e.radar_id  = radar_id
        e.ir_id     = ir_id

        # ── Fusion logic ──
        fusion_update = 0

        if radar_id != -1 and ir_id != -1 and e.radar_state is not None and e.ir_state is not None:
            # Heterogeneous T2TF (Radar-Cartesian ↔ IR-MSC)
            radar_state, radar_cov = e.radar_state, e.radar_cov
            ir_state,    ir_cov    = e.ir_state,    e.ir_cov

            if e.radar_update_time > e.ir_update_time:
                virtual_update = 1
                ir_state, ir_cov = self._predict_ir(
                    ir_state, ir_cov, e.radar_update_time - e.ir_update_time)
                fuse_time = e.radar_update_time
            elif e.radar_update_time < e.ir_update_time:
                virtual_update = 1
                radar_state, radar_cov = self._constvel_model(
                    radar_state, radar_cov, e.ir_update_time - e.radar_update_time)
                fuse_time = e.ir_update_time
            else:
                fuse_time = e.ir_update_time

            fused_s, fused_c, cross = self._t2tf_msc(
                radar_state, radar_cov, ir_state, ir_cov)
            fusion_update = 1
            e.fused_state        = fused_s
            e.fused_cov          = fused_c
            e.fusion_update_time = fuse_time
            e.cross_cov          = cross

        elif radar_id != -1 and ir_id == -1 and e.radar_state is not None:
            fusion_update = 1
            if e.fused_state is not None and e.dl_state is not None and sum(e.dl_updates) > 0:
                virtual_update = 1
                dt = e.radar_update_time - e.dl_update_time
                virt_s, virt_c = self._constvel_model(e.dl_state, e.dl_cov, dt)
                fused_s, fused_c = self._linear_kalman(
                    e.radar_state, e.radar_cov, virt_s, virt_c)
            else:
                fused_s, fused_c = e.radar_state.copy(), e.radar_cov.copy()
            e.fused_state        = fused_s
            e.fused_cov          = fused_c
            e.fusion_update_time = e.radar_update_time

        elif ir_id != -1 and radar_id == -1 and e.ir_state is not None:
            if e.fused_state is not None and (sum(e.radar_updates) > 0 or
                                               sum(e.dl_updates) > 0):
                virtual_update = 1
                ir_s4 = e.ir_state[:4]
                ir_c4 = e.ir_cov[:4, :4]
                ref_s = (e.dl_state if (e.dl_update_time > e.radar_update_time
                                        and e.dl_state is not None)
                         else e.fused_state)
                ref_c = (e.dl_cov if (e.dl_update_time > e.radar_update_time
                                      and e.dl_cov is not None)
                         else e.fused_cov)
                fused_s, fused_c, _ = self._t2tf_lmmse(ref_s, ref_c, ir_s4, ir_c4)
                fusion_update = 1
                e.fused_state        = fused_s
                e.fused_cov          = fused_c
                e.fusion_update_time = e.ir_update_time
            else:
                e.fused_state_ir_only        = e.ir_state.copy()
                e.fused_cov_ir_only          = e.ir_cov.copy()
                e.fusion_update_time_ir_only = e.ir_update_time
                e.fused_state        = None
                e.fused_cov          = None
                e.fusion_update_time = -1.0

        elif radar_id == -1 and ir_id == -1 and e.fused_state is not None:
            # Predict existing fused track forward
            virtual_update = 1
            fusion_update  = 1
            if (e.fused_state_ir_only is not None and
                    e.fusion_update_time_ir_only > e.fusion_update_time):
                ir_s, ir_c = self._predict_ir(
                    e.fused_state_ir_only, e.fused_cov_ir_only or np.eye(6),
                    self._sim_time - e.fusion_update_time_ir_only)
                e.fused_state_ir_only        = ir_s
                e.fused_cov_ir_only          = ir_c
                e.fusion_update_time_ir_only = self._sim_time
            else:
                pred_s, pred_c = self._constvel_model(
                    e.fused_state, e.fused_cov,
                    self._sim_time - e.fusion_update_time)
                e.fused_state        = pred_s
                e.fused_cov          = pred_c
                e.fusion_update_time = self._sim_time

        # Record update flags
        if not e.radar_updates[-1]:
            e.radar_updates[-1]   = radar_update
        if not e.ir_updates[-1]:
            e.ir_updates[-1]      = ir_update
        e.fusion_updates[-1]      = fusion_update
        e.virtual_updates[-1]     = virtual_update
        e.radar_ids_history[-1]   = radar_id
        e.ir_ids_history[-1]      = ir_id

        # Trim history
        for lst_name in ('radar_updates', 'ir_updates', 'fusion_updates',
                         'dl_updates', 'virtual_updates',
                         'radar_ids_history', 'ir_ids_history', 'dl_ids_history',
                         'radar_state_history', 'ir_state_history'):
            lst = getattr(e, lst_name)
            if len(lst) > _HISTORY:
                setattr(e, lst_name, lst[-_HISTORY:])

    def _remove_tracks_from_manager(self) -> None:
        n_del, n_win = self._DELETE_EPSILON
        keep = []
        for e in self._entries:
            if len(e.radar_updates) <= 20:
                keep.append(e)
                continue
            window = min(n_win, len(e.radar_updates)) - 1
            dl_win = min(n_win * 2, len(e.dl_updates)) - 1
            total_hits = (sum(e.radar_updates[-window - 1:]) +
                          sum(e.ir_updates[-window - 1:]) +
                          sum(e.dl_updates[-dl_win - 1:]))
            if total_hits > n_del:
                keep.append(e)
        self._entries = keep

    # ------------------------------------------------------------------ #
    # Fusion math
    # ------------------------------------------------------------------ #

    @staticmethod
    def _constvel_model(
        state: np.ndarray, cov: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CV prediction: state [x,vx,y,vy,z,vz]."""
        A_2 = np.array([[1, dt], [0, 1]])
        B_2 = np.array([[dt**2 / 2], [dt]])
        A = np.kron(np.eye(3), A_2)
        B = np.kron(np.eye(3), B_2)
        sigma = 1.0 + dt * 10.0
        Q = (B @ B.T) * sigma
        new_state = A @ state
        new_cov   = A.T @ cov @ A + Q
        return new_state, new_cov

    @staticmethod
    def _predict_ir(
        ir_state: np.ndarray, ir_cov: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple CV prediction for IR (MSC) state."""
        A_2 = np.array([[1, dt], [0, 1]])
        A = np.kron(np.eye(3), A_2)
        sigma = 1.0 + dt * 10.0
        B_2 = np.array([[dt**2 / 2], [dt]])
        B = np.kron(np.eye(3), B_2)
        Q = (B @ B.T) * sigma
        new_state = A @ ir_state
        new_cov   = A.T @ ir_cov @ A + Q
        return new_state, new_cov

    @staticmethod
    def _linear_kalman(
        state1: np.ndarray, cov1: np.ndarray,
        state2: np.ndarray, cov2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standard linear Kalman: treat state2 as a noisy measurement of state1."""
        C = np.eye(len(state1))
        L = cov1 @ C.T @ np.linalg.inv(C @ cov1 @ C.T + cov2)
        fused_s = state1 + L @ (state2 - C @ state1)
        fused_c = cov1 - L @ C @ cov1
        return fused_s, fused_c

    @staticmethod
    def _t2tf_msc(
        state_cart: np.ndarray, cov_cart: np.ndarray,
        state_msc:  np.ndarray, cov_msc:  np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Radar (Cartesian) + IR (MSC) LMMSE fusion.

        Converts the Cartesian state to MSC via the Jacobian, then
        applies the LMMSE equations.
        """
        G   = cartesian_to_msc_gradient(state_cart)    # (6,6)
        Pij = np.zeros((6, 6))                          # uncorrelated assumption
        Pji = Pij.T

        Pxz = cov_cart @ G.T - Pij                     # (6,6)
        Pzz = cov_msc - G @ Pij - Pji @ G.T + G @ cov_cart @ G.T   # (6,6)

        g_xi_hat = cartesian_to_msc(state_cart)         # (6,)

        try:
            K = Pxz @ np.linalg.inv(Pzz)
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(Pzz)

        fused_s = state_cart + K @ (state_msc - g_xi_hat)
        fused_c = cov_cart  - K @ Pxz.T

        return fused_s, fused_c, Pij

    @staticmethod
    def _t2tf_lmmse(
        state_i: np.ndarray, cov_i: np.ndarray,
        state_j: np.ndarray, cov_j: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Radar (Cartesian 6-D) + IR spherical (4-D) LMMSE fusion.

        state_j is [az, az_dot, el, el_dot] (MSC first 4 components).
        """
        G   = cart2sph_gradient(state_i)               # (4,6)
        Pij = np.zeros((6, 4))                          # uncorrelated assumption
        Pji = Pij.T

        Pxz = cov_i @ G.T - Pij                        # (6,4)
        Pzz = cov_j - G @ Pij - Pji @ G.T + G @ cov_i @ G.T   # (4,4)

        x, vx, y, vy, z, vz = state_i
        r2_xy = x*x + y*y
        r     = math.sqrt(r2_xy + z*z)
        az    = math.atan2(y, x)
        el    = math.atan2(-z, math.sqrt(r2_xy))
        if r2_xy > 1e-12:
            omega = (x*vy - y*vx) / r2_xy
            el_dot = -(z*(x*vx + y*vy) - vz*r2_xy) / (math.sqrt(r2_xy) * r*r)
        else:
            omega = 0.0
            el_dot = 0.0

        g_xi_hat = np.array([az, omega, el, el_dot])

        try:
            K = Pxz @ np.linalg.inv(Pzz)
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(Pzz)

        fused_s = state_i + K @ (state_j - g_xi_hat)
        fused_c = cov_i  - K @ Pxz.T

        return fused_s, fused_c, Pij
