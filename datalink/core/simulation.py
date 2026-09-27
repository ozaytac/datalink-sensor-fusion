"""
Main simulation loop.

Usage
-----
    from datalink.core.simulation import Simulation
    sim = Simulation("params.json")
    sim.run()

Or from the command line:
    python -m datalink.core.simulation params.json
"""

from __future__ import annotations

import json
import math
import pickle
import types
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from datalink.core.ownship import Ownship
from datalink.core.simulation_summary import SimulationSummary
from datalink.core.udp_functions import (
    UDPSender,
    get_telemetry_tracks,
    get_telemetry_fusion_tracks,
)


# ------------------------------------------------------------------ #
# Trajectory helpers
# ------------------------------------------------------------------ #

def _load_trajectory(traj_path: str) -> Any:
    """
    Load trajectory data from a .mat or .npy/.npz file.

    Supports:
      - scipy.io.loadmat for .mat files (v5/v7.3 format)
      - numpy .npy/.npz for pre-converted data
      - pickle .pkl files

    Returns a list of SimpleNamespace objects, one per platform,
    each with at least: id, time, latitude, longitude, altitude,
    heading, pitch, roll.
    """
    p = Path(traj_path)
    if not p.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    suffix = p.suffix.lower()

    if suffix == ".mat":
        try:
            import scipy.io as sio
            mat = sio.loadmat(str(p), squeeze_me=True, struct_as_record=False)
            return _mat_to_namespaces(mat)
        except Exception:
            # v7.3 MAT (HDF5)
            import h5py
            with h5py.File(str(p), 'r') as f:
                return _hdf5_to_namespaces(f)

    elif suffix in (".npy", ".npz"):
        data = np.load(str(p), allow_pickle=True)
        if suffix == ".npz":
            # Expect platforms stored under keys p0, p1, ...
            platforms = []
            for k in sorted(data.files):
                platforms.append(data[k].item())
            return platforms
        return [data.item()]

    elif suffix in (".pkl", ".pickle"):
        with open(str(p), "rb") as fh:
            return pickle.load(fh)

    else:
        raise ValueError(f"Unsupported trajectory format: {suffix}")


def _mat_to_namespaces(mat: dict) -> list:
    """Convert scipy.io.loadmat result to list of SimpleNamespace."""
    trajs = []
    # Heuristic: look for a key that is a struct array of platforms
    for key in mat:
        if key.startswith("_"):
            continue
        val = mat[key]
        if hasattr(val, "__len__") and not isinstance(val, (str, np.ndarray)):
            for i, item in enumerate(val):
                ns = _struct_to_ns(item, platform_id=i)
                trajs.append(ns)
            return trajs
        # Single trajectory object
        if hasattr(val, '_fieldnames'):
            trajs.append(_struct_to_ns(val, platform_id=0))
            return trajs
    return trajs


def _struct_to_ns(struct: Any, platform_id: int = 0) -> Any:
    ns = types.SimpleNamespace()
    ns.id = platform_id
    field_map = {
        'latitude':  ('lat', 'Latitude',  'latitude'),
        'longitude': ('lon', 'Longitude', 'longitude'),
        'altitude':  ('alt', 'Altitude',  'altitude'),
        'heading':   ('yaw', 'Heading',   'heading', 'psi'),
        'pitch':     ('theta', 'Pitch',   'pitch'),
        'roll':      ('phi',   'Roll',    'roll'),
        'time':      ('t', 'Time', 'time'),
    }
    for canonical, aliases in field_map.items():
        for alias in aliases:
            if hasattr(struct, alias):
                setattr(ns, canonical, np.atleast_1d(getattr(struct, alias)).astype(float))
                break
        else:
            setattr(ns, canonical, np.zeros(1))
    return ns


def _hdf5_to_namespaces(f) -> list:  # type: ignore[valid-type]
    trajs = []
    for i, key in enumerate(sorted(f.keys())):
        ns = types.SimpleNamespace(id=i)
        grp = f[key]
        for attr in ('latitude', 'longitude', 'altitude', 'heading', 'pitch', 'roll', 'time'):
            if attr in grp:
                setattr(ns, attr, np.array(grp[attr]).flatten())
            else:
                setattr(ns, attr, np.zeros(1))
        trajs.append(ns)
    return trajs


def _make_dummy_trajectory(n_steps: int = 100, dt: float = 0.2) -> list:
    """Minimal stationary platform trajectory for smoke tests."""
    t = np.linspace(0, n_steps * dt, n_steps)
    ns = types.SimpleNamespace(
        id=0,
        time=t,
        latitude=np.zeros(n_steps),
        longitude=np.zeros(n_steps),
        altitude=np.zeros(n_steps),
        heading=np.zeros(n_steps),
        pitch=np.zeros(n_steps),
        roll=np.zeros(n_steps),
        Position=np.zeros(3),
        Trajectory=types.SimpleNamespace(
            Velocity=np.zeros(3),
            Acceleration=np.zeros(3),
        ),
    )
    return [ns]


# ------------------------------------------------------------------ #
# Simulation class
# ------------------------------------------------------------------ #

class Simulation:
    """
    Main simulation orchestrator — replaces main_tracking.m.

    Parameters
    ----------
    params_path : str or Path
        Path to params.json (JSON-encoded scenario parameters).
    traj_path   : str or Path or None
        Path to trajectory file. If None, looks for 'traj_path' in params.
    enable_udp  : bool
        If True, open UDP sockets and transmit telemetry. Default False.
    """

    def __init__(
        self,
        params_path: str,
        traj_path:   Optional[str] = None,
        enable_udp:  bool = False,
    ) -> None:
        self.params = self._load_params(params_path)
        self.enable_udp = enable_udp

        # Trajectory
        tp = traj_path or self.params.get('traj_path')
        if tp:
            self.trajectories = _load_trajectory(tp)
        else:
            sp = self.params.get('scenarioParams', {})
            n_steps = int(sp.get('num_steps', 100))
            self.trajectories = _make_dummy_trajectory(n_steps)

        sp = self.params.get('scenarioParams', {})
        self.update_rate:   float = float(sp.get('updateRate', 5))
        self.dt:            float = 1.0 / self.update_rate

        # Derive time vector from trajectory
        t_all = self.trajectories[0].time if self.trajectories else np.array([0.0])
        self.times: np.ndarray = np.asarray(t_all)

        # Build ownships (one per blue platform in trajectory)
        self.ownships: List[Ownship] = self._create_ownships()

        # UDP senders (lazy — opened only if enable_udp is True)
        self._senders: Dict[int, UDPSender] = {}
        if self.enable_udp:
            self._senders = {p: UDPSender(p) for p in (5001, 5002, 5003)}

        # Results accumulated across steps
        self.history: list = []

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_params(path: str) -> dict:
        with open(path) as fh:
            return json.load(fh)

    # ------------------------------------------------------------------ #
    def _create_ownships(self) -> List[Ownship]:
        ownships = []
        num_ownship = len(self.trajectories)
        num_target  = 0  # targets identified separately if needed

        for i, traj in enumerate(self.trajectories):
            initial_pos = np.zeros(3)
            initial_lla = np.array([
                float(traj.latitude[0]),
                float(traj.longitude[0]),
                float(traj.altitude[0]),
            ])
            ow = Ownship(
                platform_id  = i + 1,
                trajectory   = traj,
                params       = self.params,
                blue1_idx    = i + 1,
                initial_pos  = initial_pos,
                initial_lla  = initial_lla,
                num_ownship  = num_ownship,
                num_target   = num_target,
            )
            ownships.append(ow)
        return ownships

    # ------------------------------------------------------------------ #
    def run(self) -> list:
        """
        Execute the full simulation loop over self.times.

        Returns
        -------
        list of per-step history dicts
        """
        for step, t in enumerate(self.times):
            self._step(t)

        # Save results
        self._save_history()
        return self.history

    # ------------------------------------------------------------------ #
    def _step(self, t: float) -> None:
        """Run a single simulation timestep."""
        step_record: dict = {"time": t, "ownships": []}

        for ow in self.ownships:
            # 1. Produce sensor data (ownship + target relative positions)
            ow.produce_sensors_data(t, self.trajectories)

            # 2. Step radar
            radar_dets, radar_tracks = ow.step_radar(t, self.dt)

            # 3. Step IR (ir_tracker=None → no tracker, stub only)
            ir_dets, ir_tracks = ow.step_infrared(t, self.dt, ir_tracker=None)

            # 4. Fusion update
            fused_tracks = ow.update_fusion(t, radar_tracks, ir_tracks)

            # 5. UDP telemetry
            if self.enable_udp and self._senders:
                radar_msg, ir_msg = get_telemetry_tracks(
                    ow.ID, radar_tracks, ir_tracks, t)
                fusion_msg = get_telemetry_fusion_tracks(
                    ow.ID, fused_tracks, t)
                if radar_msg and 5001 in self._senders:
                    self._senders[5001].send(radar_msg)
                if ir_msg and 5002 in self._senders:
                    self._senders[5002].send(ir_msg)
                if fusion_msg and 5003 in self._senders:
                    self._senders[5003].send(fusion_msg)

            # 6. Sim summary
            ow.sim_summary.update_sim_history(
                t,
                [radar_dets,  [], []],
                [radar_tracks, [], []],
                ow,
            )

            step_record["ownships"].append({
                "id":            ow.ID,
                "radar_tracks":  radar_tracks,
                "ir_tracks":     ir_tracks,
                "fused_tracks":  fused_tracks,
            })

        self.history.append(step_record)

    # ------------------------------------------------------------------ #
    def _save_history(self) -> None:
        out_dir = Path("results")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "sim_history.pkl"
        with open(out_path, "wb") as fh:
            pickle.dump(self.history, fh)

        for ow in self.ownships:
            ow.sim_summary.save(ow.ID, self.params)

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Close UDP sockets."""
        for s in self._senders.values():
            s.close()
        self._senders.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m datalink.core.simulation <params.json> [traj_path]")
        sys.exit(1)
    params_path = sys.argv[1]
    traj_path   = sys.argv[2] if len(sys.argv) > 2 else None
    sim = Simulation(params_path, traj_path=traj_path, enable_udp=False)
    sim.run()
    print(f"Done. {len(sim.history)} steps.")
