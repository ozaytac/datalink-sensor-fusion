"""
DBSCAN-based detection clustering.

Groups nearby detections and replaces each cluster with its centroid.
"""

from __future__ import annotations
import numpy as np
from typing import List
from sklearn.cluster import DBSCAN
from datalink.sensors.radar.detection import Detection


def apply_dbscan(
    ned_detections: List[Detection],
    body_detections: List[Detection],
    epsilon: float,
    min_samples: int = 1,
) -> tuple[List[Detection], List[Detection]]:
    """
    Cluster detections by position and merge each cluster to its centroid.

    Parameters
    ----------
    ned_detections  : detections in NED frame (rectangular)  → for global tracker
    body_detections : detections in body frame (spherical)    → for sensor tracker
    epsilon         : DBSCAN neighbourhood radius [m]
    min_samples     : minimum points to form a core (default 1)

    Returns
    -------
    (ned_out, body_out) : filtered detection lists, one per cluster
    """
    if not ned_detections:
        return ned_detections, body_detections

    X = np.vstack([d.measurement[:3] for d in ned_detections])

    labels = DBSCAN(eps=epsilon, min_samples=min_samples).fit_predict(X)

    ned_out  = _merge_clusters(ned_detections,  labels)
    body_out = _merge_clusters(body_detections, labels)
    return ned_out, body_out


def _merge_clusters(
    detections: List[Detection],
    labels: np.ndarray,
) -> List[Detection]:
    """
    For each cluster label, average all member detections into one.
    Noise points (label=-1) are each kept as a singleton cluster.
    """
    if not detections:
        return []

    cluster_ids = sorted(set(labels))
    merged: List[Detection] = []

    for cid in cluster_ids:
        idxs = np.where(labels == cid)[0]
        members = [detections[i] for i in idxs]

        mean_meas = np.mean([m.measurement for m in members], axis=0)
        mean_time = float(np.mean([m.time for m in members]))
        mean_snr  = float(np.mean([
            m.object_attributes.get('SNR', 0) for m in members]))

        ref = members[0]
        obj_attr = dict(ref.object_attributes)
        obj_attr['SNR'] = mean_snr

        merged.append(Detection(
            time=mean_time,
            measurement=mean_meas,
            sensor_index=ref.sensor_index,
            measurement_noise=ref.measurement_noise.copy(),
            measurement_parameters=ref.measurement_parameters.copy(),
            object_attributes=obj_attr,
        ))

    return merged
