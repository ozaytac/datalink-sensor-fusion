"""
Per-timestep fusion track error computation.

Given a SimulationSummary history dict and ownship trajectory data,
computes per-target position and angular errors at every timestep.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #

def calculate_optimization_errors(
    hist: Dict[str, Any],
    ownship_data: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute fusion position and angular errors vs ground truth.

    Parameters
    ----------
    hist : dict with keys:
        'time'            — list of N timestep floats
        'ground_truth'    — list[N] of list[T] of dicts, each with
                             'CartesianPositionNED' and 'SphericalPositionNED'
                             and optionally 'PlatformID'
        'fusion_history'  — list[N] of dicts, each with keys:
                             'FusionID', 'RadarID', 'InfraredID', 'DataLinkID',
                             'FusedState', 'FusedCov', 'FusionUpdateTime'
    ownship_data : SimpleNamespace with .time, .latitude, .longitude,
                   .altitude, .heading, .pitch, .roll  (each a 1-D array)

    Returns
    -------
    fusion_timestep  : (T, N) float — update time of the assigned fusion track
    fusion_pos_error : (T, N) float — Cartesian position error [m], NaN if unmatched
    fusion_az_error  : (T, N) float — azimuth error [deg], NaN if unmatched
    fusion_el_error  : (T, N) float — elevation error [deg], NaN if unmatched
    """
    times  = hist.get('time', [])
    gt_all = hist.get('ground_truth', [])
    fh_all = hist.get('fusion_history', [])

    N = len(times)
    if N == 0 or not gt_all:
        return (np.empty((0, 0)),) * 4

    T = len(gt_all[0]) if gt_all else 0
    if T == 0:
        return (np.full((1, N), np.nan),) * 4

    fusion_timestep  = np.full((T, N), np.nan)
    fusion_pos_error = np.full((T, N), np.nan)
    fusion_az_error  = np.full((T, N), np.nan)
    fusion_el_error  = np.full((T, N), np.nan)

    ow_time = np.asarray(getattr(ownship_data, 'time', [0.0]))
    ow_lat  = np.asarray(getattr(ownship_data, 'latitude',  np.zeros_like(ow_time)))
    ow_lon  = np.asarray(getattr(ownship_data, 'longitude', np.zeros_like(ow_time)))
    ow_alt  = np.asarray(getattr(ownship_data, 'altitude',  np.zeros_like(ow_time)))
    ow_yaw  = np.asarray(getattr(ownship_data, 'heading',   np.zeros_like(ow_time)))
    ow_pit  = np.asarray(getattr(ownship_data, 'pitch',     np.zeros_like(ow_time)))
    ow_rol  = np.asarray(getattr(ownship_data, 'roll',      np.zeros_like(ow_time)))

    for i, t in enumerate(times):
        gt_list = gt_all[i] if i < len(gt_all) else []
        fh      = fh_all[i] if i < len(fh_all) else {}

        fusion_ids   = list(fh.get('FusionID', []))
        radar_ids    = list(fh.get('RadarID', []))
        ir_ids       = list(fh.get('InfraredID', []))
        fused_states = list(fh.get('FusedState', []))
        update_times = list(fh.get('FusionUpdateTime', []))

        for j, gt in enumerate(gt_list):
            if j >= T:
                break

            gt_id  = gt.get('PlatformID', j)
            gt_pos = np.asarray(gt.get('CartesianPositionNED', np.zeros(3)), dtype=float)

            # Find matching fusion track
            ft_state = None
            ft_time  = np.nan
            for k, (rid, iid) in enumerate(zip(radar_ids, ir_ids)):
                if rid == gt_id or iid == gt_id:
                    ft_state = fused_states[k] if k < len(fused_states) else None
                    ft_time  = update_times[k] if k < len(update_times) else np.nan
                    break

            if ft_state is None or len(ft_state) < 6:
                continue

            ft_pos = np.array([ft_state[0], ft_state[2], ft_state[4]])
            fusion_timestep[j, i]  = ft_time
            fusion_pos_error[j, i] = float(np.linalg.norm(ft_pos - gt_pos))

            # Angular errors — project into body frame at ft_time
            if not np.isnan(ft_time) and len(ow_time) > 1:
                yaw   = float(np.interp(ft_time, ow_time, ow_yaw))
                pitch = float(np.interp(ft_time, ow_time, ow_pit))
                roll  = float(np.interp(ft_time, ow_time, ow_rol))

                dcm = Rotation.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()

                # Fusion body angles
                body_f = dcm @ ft_pos
                az_f = math.degrees(math.atan2(body_f[1], body_f[0]))
                el_f = math.degrees(math.atan2(-body_f[2],
                                               math.hypot(body_f[0], body_f[1])))

                # Ground-truth body angles
                body_g = dcm @ gt_pos
                az_g = math.degrees(math.atan2(body_g[1], body_g[0]))
                el_g = math.degrees(math.atan2(-body_g[2],
                                               math.hypot(body_g[0], body_g[1])))

                fusion_az_error[j, i] = abs(az_f - az_g)
                fusion_el_error[j, i] = abs(el_f - el_g)

    return fusion_timestep, fusion_pos_error, fusion_az_error, fusion_el_error


# ------------------------------------------------------------------ #
# OSPA wrappers with a fixed, stable calling convention
# ------------------------------------------------------------------ #

def calculate_radar_ospa(
    confirmed_radar_tracks: list,
    true_tracks: list,
    c: float = 100.0,
    p: float = 2.0,
) -> Tuple[float, float, float, float]:
    """
    OSPA for Cartesian-state radar tracks vs ground truth.

    Returns (ospa, loc, card, labeling).
    """
    from datalink.error_metrics.ospa import ospa_cartesian
    return ospa_cartesian(confirmed_radar_tracks, true_tracks, c=c, p=p)


def calculate_ir_ospa(
    confirmed_ir_tracks: list,
    true_tracks: list,
    c: float = 0.1,
    p: float = 2.0,
) -> Tuple[float, float, float, float]:
    """
    OSPA for MSC-state IR tracks vs ground truth (angular cut-off).

    Returns (ospa, loc, card, labeling).
    """
    from datalink.error_metrics.ospa import ospa_msc
    return ospa_msc(confirmed_ir_tracks, true_tracks, c=c, p=p)


def calculate_fusion_ospa(
    fused_tracks: list,
    true_tracks: list,
    c: float = 100.0,
    p: float = 2.0,
) -> Tuple[float, float, float, float]:
    """
    OSPA for fused (Cartesian) tracks vs ground truth.

    Returns (ospa, loc, card, labeling).
    """
    from datalink.error_metrics.ospa import ospa_cartesian
    return ospa_cartesian(fused_tracks, true_tracks, c=c, p=p)
