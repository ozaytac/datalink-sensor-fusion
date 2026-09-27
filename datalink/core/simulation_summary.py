"""
Simulation summary — per-step history accumulator.
"""

from __future__ import annotations
import math
import os
import pickle
from typing import Any, Dict, List, Optional
import numpy as np


class SimulationSummary:
    """
    Records per-timestep simulation data for later analysis or export.
    """

    def __init__(self) -> None:
        self.time:             List[float] = []
        self.radar_detections: List[Any]   = []
        self.ir_detections:    List[Any]   = []
        self.ew_detections:    List[Any]   = []
        self.radar_tracks:     List[Any]   = []
        self.ir_tracks:        List[Any]   = []
        self.ew_tracks:        List[Any]   = []
        self.fusion_history:   List[Any]   = []
        self.ground_truth:     List[Any]   = []

    # ------------------------------------------------------------------ #
    def update_sim_history(
        self,
        time: float,
        detections: List[Any],      # [radar_dets, ir_dets, ew_dets]
        tracks: List[Any],          # [radar_tracks, ir_tracks, ew_tracks]
        ownship: Any,               # Ownship instance
    ) -> None:
        self.time.append(time)

        # Detections
        self.radar_detections.append(detections[0] if len(detections) > 0 else [])
        self.ir_detections.append(   detections[1] if len(detections) > 1 else [])
        self.ew_detections.append(   detections[2] if len(detections) > 2 else [])

        # Tracks
        self.radar_tracks.append(tracks[0] if len(tracks) > 0 else [])
        self.ir_tracks.append(   self._configure_ir_tracks(
                                     tracks[1] if len(tracks) > 1 else []))
        self.ew_tracks.append(   tracks[2] if len(tracks) > 2 else [])

        # Fusion history snapshot
        fm = ownship.fusion_manager
        self.fusion_history.append({
            'FusionID':         list(fm.fusion_ids),
            'RadarID':          list(fm.radar_ids),
            'InfraredID':       list(fm.ir_ids),
            'DataLinkID':       list(fm.datalink_ids),
            'FusedState':       list(fm.fused_states),
            'FusedCov':         list(fm.fused_covs),
            'FusionUpdateTime': list(fm.fusion_update_times),
        })

        # Ground truth
        self._configure_ground_truth(
            ownship.pose_in_ownship, ownship.pose_in_local)

    # ------------------------------------------------------------------ #
    def save(
        self,
        ownship_id: int,
        params: dict,
        monte_carlo_idx: Optional[int] = None,
    ) -> None:
        """Save history as a pickle file."""
        sfx = f"_{monte_carlo_idx}" if monte_carlo_idx is not None else ""
        scene  = params['scenarioParams'].get('trajFile', 'unknown')
        noise  = params['scenarioParams'].get('sensorNoise', 'unknown')
        dl_tag = "w-DataLink_" if params['scenarioParams'].get('DataLink') else ""
        scene_name = os.path.splitext(os.path.basename(str(scene)))[0]

        os.makedirs('outputs', exist_ok=True)
        fname = f"outputs/summary_BLUE1{ownship_id}_{noise}_{dl_tag}{sfx}{scene_name}.pkl"
        with open(fname, 'wb') as f:
            pickle.dump(self.__dict__, f)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _configure_ir_tracks(ir_tracks: List[Any]) -> List[Any]:
        """Convert MSC angular states from radians to degrees (first 4 components)."""
        result = []
        for trk in ir_tracks:
            s = trk.state.copy()
            s[0] = math.degrees(s[0])
            s[1] = math.degrees(s[1])
            s[2] = math.degrees(s[2])
            s[3] = math.degrees(s[3])
            result.append({'track_id': trk.track_id, 'state': s,
                           'state_covariance': trk.state_covariance.copy()})
        return result

    def _configure_ground_truth(
        self,
        truths_body: List[Any],
        truths_local: List[Any],
    ) -> None:
        step = []
        for body, local in zip(truths_body, truths_local):
            ned_pos = np.asarray(getattr(local, 'Position',
                                         getattr(local, 'position', np.zeros(3))),
                                 dtype=float)
            az_ned  = math.atan2(ned_pos[1], ned_pos[0])
            el_ned  = math.atan2(-ned_pos[2],
                                  math.hypot(ned_pos[0], ned_pos[1]))
            rng_ned = float(np.linalg.norm(ned_pos))
            entry = dict(
                SphericalPositionNED=np.array([math.degrees(az_ned),
                                               math.degrees(el_ned),
                                               rng_ned]),
                CartesianPositionNED=ned_pos.copy(),
            )
            if hasattr(body, '__dict__'):
                entry.update(body.__dict__)
            step.append(entry)
        self.ground_truth.append(step)
