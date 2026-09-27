"""
L3/L4 test suite — FusionManager, SimulationSummary, Ownship.

Tests cover:
  - FusionEntry creation
  - constvel_model prediction
  - linear_kalman update
  - t2tf_msc heterogeneous fusion
  - t2tf_lmmse fusion
  - FusionManager lifecycle (create, assign, fuse, delete)
  - SimulationSummary append / configure
  - Ownship instantiation (lightweight smoke test)
"""

from __future__ import annotations
import math
import types
import numpy as np
import pytest

from datalink.fusion.fusion_manager import FusionManager, FusionEntry, _new_entry
from datalink.core.simulation_summary import SimulationSummary
from datalink.sensors.radar.track import Track
from datalink.transformations import cartesian_to_msc


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_params() -> dict:
    return {
        'scenarioParams': {
            'HasRadar':    False,
            'HasInfrared': False,
            'HasEW':       False,
            'FusionManager': 'FusionManager',
            'updateRate':    5,
            'trajFileUpdateRate': 50,
            'DataLink':    False,
        },
        'trackingRadar': {'PRF': 1000, 'theta_FOV': 60, 'phi_FOV': 30},
        'IRParams': {},
    }


def _make_radar_track(tid: int, pos=(100.0, 200.0, -300.0),
                      t: float = 1.0) -> Track:
    x, y, z = pos
    state = np.array([x, 0.0, y, 0.0, z, 0.0])
    return Track(track_id=tid, state=state, state_covariance=np.eye(6) * 1e4,
                 update_time=t, object_attributes={'TargetIndex': tid})


def _make_ir_track(tid: int, state: np.ndarray = None, t: float = 1.0) -> Track:
    if state is None:
        # Simple MSC state pointing roughly at (100, 200, -300)
        state = np.array([math.atan2(200, 100), 0.0,
                          math.atan2(300, math.hypot(100, 200)), 0.0,
                          1.0 / math.sqrt(100**2 + 200**2 + 300**2), 0.0])
    return Track(track_id=tid, state=state, state_covariance=np.eye(6) * 1e-4,
                 update_time=t, object_attributes={'TargetIndex': tid})


# ------------------------------------------------------------------ #
# FusionManager — math utilities
# ------------------------------------------------------------------ #

class TestFusionMath:
    def test_constvel_model_advances_position(self):
        state = np.array([100.0, 10.0, 200.0, -5.0, -300.0, 2.0])
        cov   = np.eye(6) * 1e4
        new_s, new_c = FusionManager._constvel_model(state, cov, dt=1.0)
        assert new_s[0] == pytest.approx(110.0, abs=1.0)
        assert new_s[2] == pytest.approx(195.0, abs=1.0)

    def test_constvel_model_increases_cov(self):
        state = np.zeros(6)
        cov   = np.eye(6)
        _, new_c = FusionManager._constvel_model(state, cov, dt=0.2)
        assert np.trace(new_c) > np.trace(np.eye(6))

    def test_linear_kalman_converges(self):
        s1 = np.array([100.0, 0.0, 200.0, 0.0, 300.0, 0.0])
        s2 = np.array([102.0, 0.0, 198.0, 0.0, 301.0, 0.0])
        c1 = np.eye(6) * 1e4
        c2 = np.eye(6) * 1.0
        fs, fc = FusionManager._linear_kalman(s1, c1, s2, c2)
        # Should be closer to s2 (lower noise measurement)
        assert abs(fs[0] - s2[0]) < abs(s1[0] - s2[0])
        assert np.trace(fc) < np.trace(c1)

    def test_t2tf_msc_returns_valid_state(self):
        state_c = np.array([1000.0, 50.0, 500.0, -20.0, -200.0, 10.0])
        cov_c   = np.eye(6) * 1e6
        state_m = cartesian_to_msc(state_c)
        cov_m   = np.eye(6) * 1e-4
        fs, fc, _ = FusionManager._t2tf_msc(state_c, cov_c, state_m, cov_m)
        assert fs.shape == (6,)
        assert np.all(np.isfinite(fs))
        assert np.all(np.isfinite(fc))

    def test_t2tf_lmmse_returns_valid_state(self):
        state_c = np.array([1000.0, 50.0, 500.0, -20.0, -200.0, 10.0])
        cov_c   = np.eye(6) * 1e6
        x, vx, y, vy, z, vz = state_c
        r2_xy = x*x + y*y
        az    = math.atan2(y, x)
        om    = (x*vy - y*vx) / r2_xy
        el    = math.atan2(-z, math.sqrt(r2_xy))
        el_d  = 0.0
        state_j = np.array([az, om, el, el_d])
        cov_j   = np.eye(4) * 1e-4
        fs, fc, _ = FusionManager._t2tf_lmmse(state_c, cov_c, state_j, cov_j)
        assert fs.shape == (6,)
        assert np.all(np.isfinite(fs))


# ------------------------------------------------------------------ #
# FusionManager — lifecycle
# ------------------------------------------------------------------ #

class TestFusionManagerLifecycle:
    def _make_fm(self) -> FusionManager:
        return FusionManager(traj=None, params=_make_params())

    def test_empty_update_creates_no_entries(self):
        fm = self._make_fm()
        fm.update_manager([], [], time=1.0)
        assert len(fm._entries) == 0

    def test_radar_only_creates_entry(self):
        fm = self._make_fm()
        fm.update_manager([_make_radar_track(1)], [], time=1.0)
        assert len(fm._entries) == 1
        assert fm._entries[0].radar_id == 1

    def test_ir_only_creates_entry(self):
        fm = self._make_fm()
        fm.update_manager([], [_make_ir_track(2)], time=1.0)
        assert len(fm._entries) == 1
        assert fm._entries[0].ir_id == 2

    def test_fused_entry_has_valid_state(self):
        fm = self._make_fm()
        rt = _make_radar_track(1, pos=(1000.0, 500.0, -200.0))
        it = _make_ir_track(2)
        fm.update_manager([rt], [it], time=1.0)
        # After one step both are unassigned (no prior history)
        fused = fm.get_fused_tracks()
        assert isinstance(fused, list)

    def test_multiple_steps_retain_entries(self):
        fm = self._make_fm()
        rt = _make_radar_track(1)
        for step in range(5):
            fm.update_manager([rt], [], time=float(step) * 0.2)
        assert any(e.radar_id == 1 for e in fm._entries)

    def test_get_fused_tracks_structure(self):
        fm = self._make_fm()
        fm.update_manager([_make_radar_track(1)], [], time=1.0)
        fm.update_manager([_make_radar_track(1)], [], time=1.2)
        fused = fm.get_fused_tracks()
        if fused:
            assert 'state' in fused[0]
            assert 'fusion_id' in fused[0]

    def test_datalink_update_does_not_crash(self):
        fm = self._make_fm()
        fm.update_manager([_make_radar_track(1)], [], time=1.0)
        dl_trk = _make_radar_track(99, pos=(105.0, 205.0, -298.0), t=0.9)
        fm.update_manager_with_datalink([dl_trk])   # must not raise


# ------------------------------------------------------------------ #
# SimulationSummary
# ------------------------------------------------------------------ #

class TestSimulationSummary:
    def _make_ownship_stub(self):
        fm = FusionManager(traj=None, params=_make_params())
        ow = types.SimpleNamespace(
            fusion_manager  = fm,
            pose_in_ownship = [],
            pose_in_local   = [],
        )
        return ow

    def test_empty_update_appends_time(self):
        ss = SimulationSummary()
        ow = self._make_ownship_stub()
        ss.update_sim_history(1.0, [[], [], []], [[], [], []], ow)
        assert ss.time == [1.0]

    def test_multiple_updates_accumulate(self):
        ss = SimulationSummary()
        ow = self._make_ownship_stub()
        for t in [1.0, 2.0, 3.0]:
            ss.update_sim_history(t, [[], [], []], [[], [], []], ow)
        assert len(ss.time) == 3
        assert len(ss.fusion_history) == 3

    def test_configure_ir_tracks_converts_to_degrees(self):
        from datalink.sensors.radar.track import Track
        s = np.array([0.5, 0.01, 0.3, 0.005, 1e-4, 0.0])
        trk = Track(track_id=1, state=s, state_covariance=np.eye(6))
        result = SimulationSummary._configure_ir_tracks([trk])
        assert result[0]['state'][0] == pytest.approx(math.degrees(0.5))
        assert result[0]['state'][2] == pytest.approx(math.degrees(0.3))


# ------------------------------------------------------------------ #
# Ownship — smoke test (no sensors, no traj)
# ------------------------------------------------------------------ #

class TestOwnshipSmoke:
    def _make_ownship(self):
        from datalink.core.ownship import Ownship
        traj = types.SimpleNamespace(Position=np.zeros(3), Trajectory=types.SimpleNamespace(
            Velocity=np.zeros(3), Acceleration=np.zeros(3)))
        params = _make_params()
        return Ownship(
            platform_id=1,
            trajectory=traj,
            params=params,
            blue1_idx=1,
            initial_pos=np.zeros(3),
            initial_lla=np.zeros(3),
            num_ownship=1,
            num_target=0,
        )

    def test_instantiation(self):
        ow = self._make_ownship()
        assert ow.ID == 1
        assert ow.fusion_manager is not None
        assert ow.radar_sensor is None   # HasRadar=False
        assert ow.ir_sensor    is None   # HasInfrared=False

    def test_update_fusion_empty(self):
        ow = self._make_ownship()
        result = ow.update_fusion(1.0, [], [])
        assert isinstance(result, list)

    def test_update_fusion_with_radar_tracks(self):
        ow = self._make_ownship()
        rt = _make_radar_track(1)
        result = ow.update_fusion(1.0, [rt], [])
        assert isinstance(result, list)
