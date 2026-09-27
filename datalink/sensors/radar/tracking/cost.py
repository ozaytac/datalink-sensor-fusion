"""
Assignment cost matrix for GNN tracker.
"""

import numpy as np
from typing import List
from datalink.sensors.radar.detection import Detection
from datalink.sensors.radar.track import Track


def calculate_assignment_cost(
    detections: List[Detection],
    tracks: List[Track],
    gate_upper: float,
) -> np.ndarray:
    """
    Compute the cost matrix for detection-to-track assignment.

    Parameters
    ----------
    detections : list of Detection  — position measurements [x,y,z]
    tracks     : list of Track
    gate_upper : float              — maximum allowed distance [m]; beyond → Inf

    Returns
    -------
    cost_matrix : ndarray, shape (n_tracks, n_dets)
        Entry [i,j] = Euclidean distance between track i and detection j,
        or Inf if distance > gate_upper.
    """
    n_dets   = len(detections)
    n_tracks = len(tracks)
    C = np.full((n_tracks, n_dets), np.inf)

    for j, det in enumerate(detections):
        det_pos = np.asarray(det.measurement[:3], dtype=float)
        for i, trk in enumerate(tracks):
            trk_pos = trk.state[[0, 2, 4]]          # [x, y, z]
            dist = float(np.linalg.norm(trk_pos - det_pos))
            C[i, j] = dist if dist <= gate_upper else np.inf

    C = np.where(np.isnan(C), np.inf, C)
    return C
