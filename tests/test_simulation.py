"""
L5 test suite — UDPFunctions and Simulation.

Tests cover:
  - UDPSender send / close
  - get_telemetry_tracks format (radar + IR)
  - get_telemetry_fusion_tracks format
  - _make_dummy_trajectory shape
  - Simulation.__init__ with dummy trajectory
  - Simulation._step does not crash
  - Simulation.run returns history list
  - _load_params parses JSON correctly
"""

from __future__ import annotations

import json
import math
import socket
import tempfile
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from datalink.core.udp_functions import (
    UDPSender,
    get_telemetry_tracks,
    get_telemetry_fusion_tracks,
    make_senders,
)
from datalink.core.simulation import Simulation, _make_dummy_trajectory
from datalink.sensors.radar.track import Track


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_params_file(tmp_path: Path, extra: dict | None = None) -> str:
    params = {
        'scenarioParams': {
            'HasRadar':          False,
            'HasInfrared':       False,
            'HasEW':             False,
            'FusionManager':     'FusionManager',
            'updateRate':        5,
            'trajFileUpdateRate': 50,
            'DataLink':          False,
            'num_steps':         10,
        },
        'trackingRadar': {'PRF': 1000, 'theta_FOV': 60, 'phi_FOV': 30},
        'IRParams': {},
    }
    if extra:
        params.update(extra)
    p = tmp_path / "params.json"
    p.write_text(json.dumps(params))
    return str(p)


def _make_radar_track(tid: int = 1, t: float = 1.0) -> Track:
    state = np.array([100.0, 5.0, 200.0, -3.0, -300.0, 2.0])
    return Track(track_id=tid, state=state, state_covariance=np.eye(6) * 1e4,
                 update_time=t, object_attributes={'FoF': 1})


def _make_ir_track(tid: int = 2, t: float = 1.0) -> Track:
    state = np.array([math.atan2(2, 1), 0.01, math.atan2(3, math.hypot(1, 2)),
                      0.005, 1e-4, 0.0])
    return Track(track_id=tid, state=state, state_covariance=np.eye(6) * 1e-4,
                 update_time=t, object_attributes={'FoF': 0})


def _make_fusion_track() -> dict:
    return {
        'fusion_id':  1,
        'FoF':        1,
        'state':      np.array([1000.0, 10.0, 500.0, -5.0, -200.0, 3.0]),
        'radar_id':   1,
        'radar_state': np.array([1000.0, 10.0, 500.0, -5.0, -200.0, 3.0]),
        'ir_id':      2,
        'ir_state':   np.array([math.atan2(500, 1000), 0.01,
                                math.atan2(200, math.hypot(1000, 500)), 0.005,
                                1e-4, 0.0]),
    }


# ------------------------------------------------------------------ #
# UDPSender tests
# ------------------------------------------------------------------ #

class TestUDPSender:
    def _recv_one(self, port: int, timeout: float = 0.5) -> bytes:
        """Listen on port, return first datagram."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(timeout)
        sock.bind(("127.0.0.1", port))
        try:
            data, _ = sock.recvfrom(65535)
            return data
        finally:
            sock.close()

    def test_send_receives_message(self):
        port = 59900
        received: list[bytes] = []

        def _listen():
            received.append(self._recv_one(port, timeout=1.0))

        t = threading.Thread(target=_listen, daemon=True)
        t.start()
        import time; time.sleep(0.05)

        sender = UDPSender(port)
        sender.send("hello test")
        sender.close()
        t.join(timeout=2.0)

        assert received and received[0] == b"hello test"

    def test_context_manager_closes(self):
        with UDPSender(59901) as s:
            assert s._sock is not None
        # After exit, socket fd should be closed (send raises)
        with pytest.raises(Exception):
            s.send("after close")

    def test_make_senders_returns_four(self):
        senders = make_senders()
        try:
            assert set(senders.keys()) == {5000, 5001, 5002, 5003}
        finally:
            for s in senders.values():
                s.close()


# ------------------------------------------------------------------ #
# Telemetry formatter tests
# ------------------------------------------------------------------ #

class TestGetTelemetryTracks:
    def test_empty_tracks_returns_empty_strings(self):
        r, ir = get_telemetry_tracks(1, [], [], 1.0)
        assert r == ""
        assert ir == ""

    def test_radar_track_format(self):
        rt = _make_radar_track(tid=7)
        msg, _ = get_telemetry_tracks(1, [rt], [], 2.5)
        fields = msg.split(",")
        assert fields[0] == "1"          # ownship_id
        assert fields[1] == "R"          # type
        assert fields[3] == "7"          # trackId
        assert float(fields[4]) == pytest.approx(2.5, rel=1e-4)  # time

    def test_ir_track_format(self):
        it = _make_ir_track(tid=3)
        _, msg = get_telemetry_tracks(2, [], [it], 3.0)
        fields = msg.split(",")
        assert fields[0] == "2"
        assert fields[1] == "I"
        assert fields[3] == "3"
        az_deg = float(fields[5])
        # az should be degrees of atan2(2,1)
        assert az_deg == pytest.approx(math.degrees(math.atan2(2, 1)), rel=1e-3)

    def test_multiple_tracks_semicolon_separated(self):
        rt1 = _make_radar_track(1)
        rt2 = _make_radar_track(2)
        msg, _ = get_telemetry_tracks(1, [rt1, rt2], [], 1.0)
        parts = msg.split(";")
        assert len(parts) == 2

    def test_fof_field_set_correctly(self):
        rt = _make_radar_track(1)
        msg, _ = get_telemetry_tracks(1, [rt], [], 1.0)
        fields = msg.split(",")
        assert fields[2] == "1"  # FoF from object_attributes


class TestGetTelemetryFusionTracks:
    def test_empty_fusion_returns_empty(self):
        msg = get_telemetry_fusion_tracks(1, [], 1.0)
        assert msg == ""

    def test_fusion_track_format(self):
        ft = _make_fusion_track()
        msg = get_telemetry_fusion_tracks(1, [ft], 5.0)
        fields = msg.split(",")
        assert fields[0] == "1"          # ownship_id
        assert fields[1] == "F"          # type
        assert int(fields[5]) == 1       # is_xyz=1 (Cartesian state provided)

    def test_fusion_xyz_values_match_state(self):
        ft = _make_fusion_track()
        msg = get_telemetry_fusion_tracks(1, [ft], 1.0)
        fields = msg.split(",")
        tX = float(fields[6])
        tY = float(fields[7])
        tZ = float(fields[8])
        assert tX == pytest.approx(1000.0, rel=1e-3)
        assert tY == pytest.approx(500.0,  rel=1e-3)
        assert tZ == pytest.approx(-200.0, rel=1e-3)

    def test_radar_sub_track_present(self):
        ft = _make_fusion_track()
        msg = get_telemetry_fusion_tracks(1, [ft], 1.0)
        fields = msg.split(",")
        is_radar = int(fields[12])
        assert is_radar == 1

    def test_ir_sub_track_present(self):
        ft = _make_fusion_track()
        msg = get_telemetry_fusion_tracks(1, [ft], 1.0)
        fields = msg.split(",")
        is_ir = int(fields[16])
        assert is_ir == 1

    def test_no_radar_or_ir_tracks(self):
        ft = {
            'fusion_id': 9,
            'FoF': 0,
            'state': np.array([100.0, 0.0, 50.0, 0.0, -30.0, 0.0]),
        }
        msg = get_telemetry_fusion_tracks(1, [ft], 1.0)
        fields = msg.split(",")
        assert int(fields[12]) == 0   # is_radar
        assert int(fields[16]) == 0   # is_ir


# ------------------------------------------------------------------ #
# Dummy trajectory
# ------------------------------------------------------------------ #

class TestMakeDummyTrajectory:
    def test_returns_list(self):
        trajs = _make_dummy_trajectory(n_steps=50)
        assert isinstance(trajs, list)
        assert len(trajs) == 1

    def test_trajectory_has_time_array(self):
        trajs = _make_dummy_trajectory(n_steps=20, dt=0.1)
        assert len(trajs[0].time) == 20

    def test_has_position_and_trajectory(self):
        trajs = _make_dummy_trajectory()
        ns = trajs[0]
        assert hasattr(ns, 'Position')
        assert hasattr(ns, 'Trajectory')
        assert hasattr(ns.Trajectory, 'Velocity')


# ------------------------------------------------------------------ #
# Simulation tests
# ------------------------------------------------------------------ #

class TestSimulation:
    def _make_sim(self, tmp_path: Path) -> Simulation:
        params_file = _make_params_file(tmp_path)
        return Simulation(params_file, enable_udp=False)

    def test_instantiation(self, tmp_path):
        sim = self._make_sim(tmp_path)
        assert len(sim.ownships) >= 1
        assert sim.dt == pytest.approx(0.2, rel=1e-4)

    def test_load_params_parses_json(self, tmp_path):
        pf = _make_params_file(tmp_path)
        params = Simulation._load_params(pf)
        assert 'scenarioParams' in params
        assert params['scenarioParams']['updateRate'] == 5

    def test_step_does_not_crash(self, tmp_path):
        sim = self._make_sim(tmp_path)
        sim._step(sim.times[0])  # must not raise

    def test_run_returns_history(self, tmp_path):
        sim = self._make_sim(tmp_path)
        history = sim.run()
        assert isinstance(history, list)
        assert len(history) > 0

    def test_history_structure(self, tmp_path):
        sim = self._make_sim(tmp_path)
        sim._step(0.0)
        record = sim.history[0]
        assert 'time' in record
        assert 'ownships' in record
        assert isinstance(record['ownships'], list)

    def test_ownship_record_structure(self, tmp_path):
        sim = self._make_sim(tmp_path)
        sim._step(0.0)
        ow_rec = sim.history[0]['ownships'][0]
        assert 'id' in ow_rec
        assert 'radar_tracks' in ow_rec
        assert 'fused_tracks' in ow_rec

    def test_context_manager(self, tmp_path):
        pf = _make_params_file(tmp_path)
        with Simulation(pf, enable_udp=False) as sim:
            sim._step(0.0)
        # should not raise on exit

    def test_udp_disabled_no_senders(self, tmp_path):
        sim = self._make_sim(tmp_path)
        assert len(sim._senders) == 0

    def test_save_history_creates_file(self, tmp_path, monkeypatch):
        # Patch Path("results") to use tmp_path
        import datalink.core.simulation as sim_mod
        monkeypatch.chdir(tmp_path)
        pf = _make_params_file(tmp_path)
        sim = Simulation(pf, enable_udp=False)
        sim.run()
        assert (tmp_path / "results" / "sim_history.pkl").exists()
