"""
Simulation results → Excel / CSV export.

Loads a SimulationSummary pickle produced by SimulationSummary.save(),
converts fused-track NED states to geodetic (lat/lon/alt), derives
ground speed / heading / heading-rate / altitude-rate, and writes an
Excel (or CSV) file with one row per fused-track timestep.

Usage
-----
    from datalink.io.export import export_to_excel
    export_to_excel(
        summary_path="outputs/summary_BLUE11_...pkl",
        ownship_data=my_ownship_traj,   # SimpleNamespace with .time/.latitude/...
        output_path="SenaryoCSV/scene.xlsx",
        min_track_points=10,
    )

Or from the command line:
    python -m datalink.io.export <summary.pkl> <output.xlsx>
"""

from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from datalink.io.geo import lla_to_ecef, ned_to_ecef, ecef_to_lla


# ------------------------------------------------------------------ #
# Bearing / heading-rate helpers
# ------------------------------------------------------------------ #

def _get_bearing(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Compute geodetic forward bearing (degrees, 0–360) between consecutive
    (lat, lon) pairs.  Returns array of same length (last value repeated).
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if len(lats) < 2:
        return np.zeros(len(lats))

    d_lon = np.radians(np.diff(lons))
    lat1  = np.radians(lats[:-1])
    lat2  = np.radians(lats[1:])

    x = np.cos(lat2) * np.sin(d_lon)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)

    bearing = np.degrees(np.arctan2(x, y)) % 360.0
    return np.append(bearing, bearing[-1])


def _get_heading_rate(heading: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    Compute heading rate [deg/s], wrapping discontinuities at ±180°.
    Returns array of same length (last value = 0).
    """
    heading = np.asarray(heading, dtype=float)
    time    = np.asarray(time,    dtype=float)
    if len(heading) < 2:
        return np.zeros(len(heading))

    dt   = np.diff(time)
    dt[dt == 0] = 1e-9            # guard divide-by-zero
    rate = np.diff(heading) / dt

    # Wrap to (−180, +180]
    rate = np.where(rate > 180,   rate - 360.0, rate)
    rate = np.where(rate < -180,  rate + 360.0, rate)

    return np.append(rate, 0.0)


# ------------------------------------------------------------------ #
# NED state → absolute LLA
# ------------------------------------------------------------------ #

def _ned_state_to_lla(
    fused_state: np.ndarray,
    lat0: float, lon0: float, alt0: float,
) -> tuple[float, float, float]:
    """
    Maps (y_ned, x_ned, z_ned) = (fused_state[2], fused_state[0], fused_state[4])
    as (xEast, yNorth, zUp) for the ENU-based conversion.
    """
    x_east  = float(fused_state[2])
    y_north = float(fused_state[0])
    z_up    = float(fused_state[4])
    ex, ey, ez = ned_to_ecef(x_east, y_north, z_up, lat0, lon0, alt0)
    lat, lon, alt = ecef_to_lla(ex, ey, ez)
    return float(lat), float(lon), float(alt)


# ------------------------------------------------------------------ #
# Main export function
# ------------------------------------------------------------------ #

def export_to_excel(
    summary_path:     str,
    ownship_data:     Any,
    output_path:      str = "SenaryoCSV/output.xlsx",
    min_track_points: int = 10,
) -> None:
    """
    Export fusion-track history to Excel (.xlsx) or CSV (.csv).

    Parameters
    ----------
    summary_path     : path to pickled SimulationSummary.__dict__
    ownship_data     : SimpleNamespace with arrays:
                         .time, .latitude, .longitude, .altitude,
                         .heading, .xSpeed, .ySpeed, .zSpeed  (all 1-D)
    output_path      : output file; extension determines format
                       (.xlsx → openpyxl/xlsxwriter, .csv → plain CSV)
    min_track_points : discard tracks with fewer rows than this
    """
    # Load summary
    with open(summary_path, "rb") as fh:
        hist = pickle.load(fh)

    if isinstance(hist, dict):
        fusion_history = hist.get("fusion_history", [])
        times          = hist.get("time", [])
    else:
        fusion_history = getattr(hist, "fusion_history", [])
        times          = getattr(hist, "time", [])

    ow_time = np.asarray(getattr(ownship_data, "time",      [0.0]))
    ow_lat  = np.asarray(getattr(ownship_data, "latitude",  np.zeros_like(ow_time)))
    ow_lon  = np.asarray(getattr(ownship_data, "longitude", np.zeros_like(ow_time)))
    ow_alt  = np.asarray(getattr(ownship_data, "altitude",  np.zeros_like(ow_time)))
    ow_yaw  = np.asarray(getattr(ownship_data, "heading",   np.zeros_like(ow_time)))
    ow_vx   = np.asarray(getattr(ownship_data, "xSpeed",    np.zeros_like(ow_time)))
    ow_vy   = np.asarray(getattr(ownship_data, "ySpeed",    np.zeros_like(ow_time)))
    ow_vz   = np.asarray(getattr(ownship_data, "zSpeed",    np.zeros_like(ow_time)))

    # Collect all fusion IDs across time
    all_fusion_ids: set = set()
    for fh_step in fusion_history:
        if isinstance(fh_step, dict):
            for fid in fh_step.get("FusionID", []):
                all_fusion_ids.add(int(fid))

    rows: list[dict] = []

    for fid in sorted(all_fusion_ids):
        track_rows: list[dict] = []

        for t_idx, fh_step in enumerate(fusion_history):
            if not isinstance(fh_step, dict):
                continue

            fusion_ids   = list(fh_step.get("FusionID",          []))
            fused_states = list(fh_step.get("FusedState",         []))
            update_times = list(fh_step.get("FusionUpdateTime",   []))
            target_idxs  = list(fh_step.get("TargetIndex",        [None] * len(fusion_ids)))
            types_list   = list(fh_step.get("Type",               [""  ] * len(fusion_ids)))

            try:
                k = fusion_ids.index(fid)
            except ValueError:
                continue

            fused_state = fused_states[k] if k < len(fused_states) else None
            update_time = float(update_times[k]) if k < len(update_times) else float("nan")

            if fused_state is None or len(fused_state) < 6:
                continue

            fused_state = np.asarray(fused_state, dtype=float)

            # Ownship position/velocity at update_time
            if len(ow_time) > 1 and not math.isnan(update_time):
                lat0 = float(np.interp(update_time, ow_time, ow_lat))
                lon0 = float(np.interp(update_time, ow_time, ow_lon))
                alt0 = float(np.interp(update_time, ow_time, ow_alt))
                vx0  = float(np.interp(update_time, ow_time, ow_vx))
                vy0  = float(np.interp(update_time, ow_time, ow_vy))
                vz0  = float(np.interp(update_time, ow_time, ow_vz))
            else:
                lat0 = lon0 = alt0 = 0.0
                vx0 = vy0 = vz0 = 0.0

            tgt_lat, tgt_lon, tgt_alt = _ned_state_to_lla(
                fused_state, lat0, lon0, alt0)

            # Ground speed and altitude rate (absolute = relative + ownship)
            gs       = math.hypot(fused_state[1] + vx0, fused_state[3] + vy0)
            alt_rate = float(fused_state[5]) + vz0

            # Heading from velocity vector
            heading_vel = math.degrees(
                math.atan2(fused_state[3] + vy0, fused_state[1] + vx0))

            target_id = target_idxs[k] if k < len(target_idxs) else -1
            type_str  = types_list[k]  if k < len(types_list)  else ""

            track_rows.append({
                "flight_id":    fid,
                "targetID":     target_id,
                "time":         round(update_time, 2),
                "latitude":     tgt_lat,
                "longitude":    tgt_lon,
                "altitude":     tgt_alt,
                "track":        heading_vel,
                "ground_speed": gs,
                "altitudeRate": alt_rate,
                "type":         type_str,
            })

        if len(track_rows) < min_track_points:
            continue

        # Compute heading rate column
        headings = np.array([r["track"] for r in track_rows])
        times_v  = np.array([r["time"]  for r in track_rows])
        h_rates  = _get_heading_rate(headings, times_v)

        for i, row in enumerate(track_rows):
            row["headingRate"] = float(h_rates[i])

        rows.extend(track_rows)

    # Write output
    _write_output(rows, output_path)


# ------------------------------------------------------------------ #
# Output writers
# ------------------------------------------------------------------ #

def _write_output(rows: list[dict], output_path: str) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        # Write empty file so callers don't crash
        if p.suffix.lower() == ".csv":
            p.write_text("")
        else:
            _write_xlsx_empty(p)
        return

    columns = [
        "flight_id", "targetID", "time",
        "latitude", "longitude", "altitude",
        "track", "ground_speed", "altitudeRate",
        "headingRate", "type",
    ]

    if p.suffix.lower() == ".csv":
        import csv
        with open(p, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    else:
        try:
            import pandas as pd
            df = pd.DataFrame(rows, columns=columns)
            df.to_excel(str(p), index=False)
        except ImportError:
            # Fallback: write CSV with .xlsx extension
            import csv
            with open(p, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)


def _write_xlsx_empty(p: Path) -> None:
    try:
        import pandas as pd
        pd.DataFrame().to_excel(str(p), index=False)
    except ImportError:
        p.write_text("")


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import types

    if len(sys.argv) < 3:
        print("Usage: python -m datalink.io.export <summary.pkl> <output.xlsx>")
        sys.exit(1)

    summary_path = sys.argv[1]
    output_path  = sys.argv[2]

    # Minimal ownship_data stub (no trajectory data available at CLI)
    ow = types.SimpleNamespace(
        time=np.array([0.0, 1.0]),
        latitude=np.zeros(2), longitude=np.zeros(2), altitude=np.zeros(2),
        heading=np.zeros(2),  xSpeed=np.zeros(2), ySpeed=np.zeros(2),
        zSpeed=np.zeros(2),
    )
    export_to_excel(summary_path, ow, output_path)
    print(f"Exported to {output_path}")
