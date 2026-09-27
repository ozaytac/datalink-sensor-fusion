"""
UDP telemetry functions for DataLink_v2.0.

Ports:
  5001 — radar tracks
  5002 — IR tracks
  5003 — fusion tracks
"""

from __future__ import annotations

import math
import socket
from typing import Any, List, Optional


# ------------------------------------------------------------------ #
# UDPSender
# ------------------------------------------------------------------ #

class UDPSender:
    """
    Thin wrapper around a UDP socket targeting localhost:<port>.
    Call send(msg) to transmit a UTF-8 string.
    """

    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        self.host = host
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, msg: str) -> None:
        data = msg.encode("utf-8")
        self._sock.sendto(data, (self.host, self.port))

    def close(self) -> None:
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ------------------------------------------------------------------ #
# Telemetry message formatters
# ------------------------------------------------------------------ #

def get_telemetry_tracks(
    ownship_id: int,
    radar_tracks: list,
    ir_tracks: list,
    time: float,
) -> tuple[str, str]:
    """
    Format radar and IR tracks as UDP telemetry strings.

    Radar format (semicolon-separated entries):
        ownship_id,R,FoF,trackId,time,X,Y,Z

    IR format (semicolon-separated entries):
        ownship_id,I,FoF,trackId,time,az_deg,el_deg

    Returns
    -------
    (radar_msg, ir_msg) — each is a semicolon-delimited string
    """
    radar_parts: list[str] = []
    for trk in radar_tracks:
        tid  = getattr(trk, 'track_id', 0)
        fof  = int(trk.object_attributes.get('FoF', 0) if hasattr(trk, 'object_attributes') else 0)
        s    = trk.state
        x, y, z = float(s[0]), float(s[2]), float(s[4])
        entry = f"{ownship_id},R,{fof},{tid},{time:.6f},{x:.4f},{y:.4f},{z:.4f}"
        radar_parts.append(entry)

    ir_parts: list[str] = []
    for trk in ir_tracks:
        tid   = getattr(trk, 'track_id', 0)
        fof   = int(trk.object_attributes.get('FoF', 0) if hasattr(trk, 'object_attributes') else 0)
        s     = trk.state
        az_d  = math.degrees(float(s[0]))
        el_d  = math.degrees(float(s[2]))
        entry = f"{ownship_id},I,{fof},{tid},{time:.6f},{az_d:.6f},{el_d:.6f}"
        ir_parts.append(entry)

    return ";".join(radar_parts), ";".join(ir_parts)


def get_telemetry_fusion_tracks(
    ownship_id: int,
    fusion_tracks: list[dict],
    time: float,
) -> str:
    """
    Format fused tracks as a UDP telemetry string.

    Format per entry (semicolon-separated):
        ownship_id, F, FoF, trackId, time,
        is_xyz, tX, tY, tZ, tAz, tEl,
        is_radar_exist, radar_track_id, rX, rY, rZ,
        is_ir_exist, ir_track_id, irAz, irEl

    Returns a semicolon-delimited string.
    """
    parts: list[str] = []
    for ft in fusion_tracks:
        fusion_id = ft.get('fusion_id', 0)
        fof       = ft.get('FoF', 0)
        state     = ft.get('state')

        # Fused state — prefer Cartesian if available
        if state is not None and len(state) >= 6:
            tX, tY, tZ = float(state[0]), float(state[2]), float(state[4])
            r_xy  = math.hypot(tX, tY)
            tAz   = math.degrees(math.atan2(tY, tX))
            tEl   = math.degrees(math.atan2(-tZ, r_xy))
            is_xyz = 1
        else:
            tX = tY = tZ = tAz = tEl = 0.0
            is_xyz = 0

        # Radar sub-track
        r_state = ft.get('radar_state')
        if r_state is not None:
            is_radar = 1
            rid  = ft.get('radar_id', 0)
            rX, rY, rZ = float(r_state[0]), float(r_state[2]), float(r_state[4])
        else:
            is_radar = 0
            rid = rX = rY = rZ = 0

        # IR sub-track
        ir_state = ft.get('ir_state')
        if ir_state is not None:
            is_ir   = 1
            ir_id   = ft.get('ir_id', 0)
            irAz    = math.degrees(float(ir_state[0]))
            irEl    = math.degrees(float(ir_state[2]))
        else:
            is_ir = 0
            ir_id = irAz = irEl = 0

        entry = (
            f"{ownship_id},F,{fof},{fusion_id},{time:.6f},"
            f"{is_xyz},{tX:.4f},{tY:.4f},{tZ:.4f},{tAz:.6f},{tEl:.6f},"
            f"{is_radar},{rid},{rX:.4f},{rY:.4f},{rZ:.4f},"
            f"{is_ir},{ir_id},{irAz:.6f},{irEl:.6f}"
        )
        parts.append(entry)

    return ";".join(parts)


# ------------------------------------------------------------------ #
# Convenience: create all four senders as in Sender.m
# ------------------------------------------------------------------ #

def make_senders(host: str = "127.0.0.1") -> dict[int, UDPSender]:
    """Return {port: UDPSender} for ports 5000–5003."""
    return {p: UDPSender(p, host) for p in range(5000, 5004)}
