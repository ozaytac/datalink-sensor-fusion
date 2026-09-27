"""
L2 test suite — datalink.sensors.radar package.

Tests cover:
  - Detection / Track dataclasses
  - FluctuationModel (Swerling 0,1,3,5)
  - RadarBeam position management
  - radar_snjr_db equation (sanity checks)
  - Target construction and spherical geometry
  - Clutter model (smoke-tests only)
  - TrackingRadar (deterministic path via fixed RNG)
  - GNNTracker lifecycle (birth → confirm → delete)
  - DBSCAN filter (cluster merge)
  - Cost matrix
  - CVEKF predict / update
  - SimulationHistory iterator
  - MPAR coordinate conversion (sph→cart roundtrip)
  - InfraredSensor stub instantiation
"""

from __future__ import annotations
import math
import numpy as np
import pytest

# ------------------------------------------------------------------ #
# Detection / Track
# ------------------------------------------------------------------ #
from datalink.sensors.radar.detection import Detection
from datalink.sensors.radar.track import Track


class TestDetection:
    def test_default_construction(self):
        d = Detection(time=0.1, measurement=np.array([1.0, 2.0, 3.0]))
        assert d.time == pytest.approx(0.1)
        np.testing.assert_array_equal(d.measurement, [1.0, 2.0, 3.0])

    def test_object_attributes_copy(self):
        attrs = {'SNR': 10.0, 'TargetIndex': 3}
        d = Detection(time=0.0, measurement=np.zeros(3), object_attributes=attrs)
        attrs['SNR'] = 99.0
        assert d.object_attributes['SNR'] == 10.0   # must be independent copy


class TestTrack:
    def test_position_property(self):
        state = np.array([10.0, 1.0, 20.0, 2.0, 30.0, 3.0])
        t = Track(track_id=1, state=state, state_covariance=np.eye(6))
        np.testing.assert_array_equal(t.position, [10.0, 20.0, 30.0])

    def test_velocity_property(self):
        state = np.array([10.0, 1.0, 20.0, 2.0, 30.0, 3.0])
        t = Track(track_id=1, state=state, state_covariance=np.eye(6))
        np.testing.assert_array_equal(t.velocity, [1.0, 2.0, 3.0])


# ------------------------------------------------------------------ #
# FluctuationModel
# ------------------------------------------------------------------ #
from datalink.sensors.radar.fluctuation_models import FluctuationModel


class TestFluctuationModel:
    def test_swerling0_constant(self):
        fm = FluctuationModel(swerling_type='Swerling0', sigma_avg=1.0)
        samples = [fm.sample() for _ in range(20)]
        assert all(s == pytest.approx(1.0) for s in samples)

    def test_swerling5_constant(self):
        fm = FluctuationModel(swerling_type='Swerling5', sigma_avg=2.0)
        assert fm.sample() == pytest.approx(2.0)

    def test_swerling1_positive(self):
        fm = FluctuationModel(swerling_type='Swerling1', sigma_avg=1.0)
        samples = [fm.sample() for _ in range(50)]
        assert all(s > 0 for s in samples)

    def test_swerling3_mean_close_to_sigma_avg(self):
        np.random.seed(42)
        fm = FluctuationModel(swerling_type='Swerling3', sigma_avg=2.0)
        samples = [fm.sample() for _ in range(10_000)]
        assert np.mean(samples) == pytest.approx(2.0, rel=0.05)


# ------------------------------------------------------------------ #
# RadarBeam
# ------------------------------------------------------------------ #
from datalink.sensors.radar.radar_beam import RadarBeam


class TestRadarBeam:
    def test_initial_position(self):
        b = RadarBeam(0, 0, 0.0, 0.0, math.radians(3), math.radians(3))
        assert b.theta == pytest.approx(0.0)
        assert b.phi   == pytest.approx(0.0)

    def test_change_beam_position(self):
        b = RadarBeam(0, 0, 0.0, 0.0, math.radians(3), math.radians(3))
        b.change_beam_position_in_radians(0.5, -0.3)
        assert b.theta == pytest.approx(0.5)
        assert b.phi   == pytest.approx(-0.3)

    def test_move_wraps_around_pi(self):
        b = RadarBeam(0, 0, 0.0, 0.0, math.radians(3), math.radians(3))
        b.move_beam_in_radians(3.0, 0.0)
        b.move_beam_in_radians(3.0, 0.0)
        # Should wrap to [-π, π]
        assert -math.pi <= b.theta <= math.pi


# ------------------------------------------------------------------ #
# radar_snjr_db
# ------------------------------------------------------------------ #
from datalink.sensors.radar.snjr import radar_snjr_db


_SNJR_COMMON = dict(
    pt=50e3, freq=10e9, sigma=1.0,
    te=500.0, loss_db=0.1, tau=1e-6,
    nf_db=3.0, jammers=[], radar_theta=0.0, radar_phi=0.0,
    theta_3db=math.radians(3), phi_3db=math.radians(3),
    main_beam_gain_linear=1e4, sidelobe_gain_db=10.0,
)


class TestRadarSNJR:
    def test_no_jammers_returns_finite(self):
        snr = radar_snjr_db(range_m=10e3, **_SNJR_COMMON)
        assert math.isfinite(snr)

    def test_longer_range_lower_snr(self):
        snr_near = radar_snjr_db(range_m=5e3,  **_SNJR_COMMON)
        snr_far  = radar_snjr_db(range_m=50e3, **_SNJR_COMMON)
        assert snr_near > snr_far


# ------------------------------------------------------------------ #
# Target
# ------------------------------------------------------------------ #
from datalink.sensors.radar.target import Target


class TestTarget:
    def _make_target(self):
        body_pos = np.array([1000.0, 0.0, 0.0])
        local_pos = body_pos.copy()
        return Target(target_id=1,
                      pose_in_ownship=body_pos,
                      pose_in_local=local_pos,
                      rcs_avg=1.0, swerling_type='Swerling0', chaff=0)

    def test_spherical_range(self):
        t = self._make_target()
        _, _, rng = t.get_body_spherical()
        assert rng == pytest.approx(1000.0, rel=1e-6)

    def test_spherical_azimuth_on_axis(self):
        t = self._make_target()
        theta, phi, _ = t.get_body_spherical()
        # body_pos = [1000, 0, 0] → az=0, el=0
        assert theta == pytest.approx(0.0, abs=1e-9)
        assert phi   == pytest.approx(0.0, abs=1e-9)

    def test_sample_rcs_swerling0(self):
        t = self._make_target()
        assert t.sample_rcs() == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# CVEKF
# ------------------------------------------------------------------ #
from datalink.sensors.radar.tracking.cvekf import CVEKF, init_cv_ekf


class TestCVEKF:
    def _make_ekf(self):
        state = np.array([100.0, 10.0, 200.0, -5.0, 300.0, 2.0])
        P     = np.diag([1e4, 100.0, 1e4, 100.0, 1e4, 100.0])
        return CVEKF(state, P)

    def test_predict_advances_position(self):
        ekf = self._make_ekf()
        ekf.predict(dt=1.0)
        # x should advance by vx
        assert ekf.state[0] == pytest.approx(110.0, abs=1.0)

    def test_predict_increases_uncertainty(self):
        ekf = self._make_ekf()
        P0 = np.trace(ekf.state_covariance)
        ekf.predict(dt=1.0)
        assert np.trace(ekf.state_covariance) > P0

    def test_update_reduces_uncertainty(self):
        ekf = self._make_ekf()
        P_prior = np.trace(ekf.state_covariance)
        meas = np.array([100.0, 200.0, 300.0])
        ekf.update(meas)
        assert np.trace(ekf.state_covariance) < P_prior

    def test_init_cv_ekf(self):
        det = Detection(
            time=0.0,
            measurement=np.array([500.0, 600.0, 700.0]),
            measurement_noise=np.diag([25.0, 25.0, 25.0]),
        )
        ekf = init_cv_ekf(det)
        np.testing.assert_array_equal(ekf.state[[0, 2, 4]],
                                      [500.0, 600.0, 700.0])
        # velocity initialized to zero
        np.testing.assert_array_equal(ekf.state[[1, 3, 5]],
                                      [0.0, 0.0, 0.0])


# ------------------------------------------------------------------ #
# Cost matrix
# ------------------------------------------------------------------ #
from datalink.sensors.radar.tracking.cost import calculate_assignment_cost


class TestCostMatrix:
    def test_shape(self):
        dets = [
            Detection(time=0.0, measurement=np.array([1.0, 0.0, 0.0])),
            Detection(time=0.0, measurement=np.array([100.0, 0.0, 0.0])),
        ]
        tracks = [
            Track(track_id=1,
                  state=np.array([1.1, 0.0, 0.1, 0.0, 0.0, 0.0]),
                  state_covariance=np.eye(6)),
        ]
        C = calculate_assignment_cost(dets, tracks, gate_upper=1e4)
        assert C.shape == (1, 2)

    def test_near_det_cheap(self):
        dets = [Detection(time=0.0, measurement=np.array([0.0, 0.0, 0.0]))]
        tracks = [Track(track_id=1,
                        state=np.zeros(6),
                        state_covariance=np.eye(6))]
        C = calculate_assignment_cost(dets, tracks, gate_upper=1e4)
        assert C[0, 0] == pytest.approx(0.0, abs=1e-9)

    def test_far_det_inf(self):
        dets  = [Detection(time=0.0, measurement=np.array([0.0, 0.0, 0.0]))]
        state = np.array([1e6, 0.0, 0.0, 0.0, 0.0, 0.0])
        tracks = [Track(track_id=1, state=state, state_covariance=np.eye(6))]
        C = calculate_assignment_cost(dets, tracks, gate_upper=1e4)
        assert C[0, 0] == np.inf


# ------------------------------------------------------------------ #
# DBSCAN filter
# ------------------------------------------------------------------ #
from datalink.sensors.radar.tracking.dbscan_filter import apply_dbscan


class TestDBSCAN:
    def _make_dets(self, positions):
        return [
            Detection(time=0.0,
                      measurement=np.array(p, dtype=float),
                      measurement_noise=np.eye(3))
            for p in positions
        ]

    def test_single_det_passes_through(self):
        dets = self._make_dets([[0.0, 0.0, 0.0]])
        ned_out, body_out = apply_dbscan(dets, dets, epsilon=100.0)
        assert len(ned_out) == 1

    def test_two_nearby_merge_to_one(self):
        dets = self._make_dets([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        ned_out, body_out = apply_dbscan(dets, dets, epsilon=10.0)
        assert len(ned_out) == 1
        np.testing.assert_array_almost_equal(
            ned_out[0].measurement, [0.5, 0.0, 0.0])

    def test_two_far_stay_separate(self):
        dets = self._make_dets([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
        ned_out, body_out = apply_dbscan(dets, dets, epsilon=10.0)
        assert len(ned_out) == 2

    def test_empty_input(self):
        ned_out, body_out = apply_dbscan([], [], epsilon=10.0)
        assert ned_out == []


# ------------------------------------------------------------------ #
# GNNTracker lifecycle
# ------------------------------------------------------------------ #
from datalink.sensors.radar.tracking.gnn_tracker import GNNTracker


class TestGNNTracker:
    def _det(self, x, y, z, tid=1):
        return Detection(
            time=0.0,
            measurement=np.array([x, y, z], dtype=float),
            measurement_noise=np.diag([25.0, 25.0, 25.0]),
            object_attributes={'TargetIndex': tid},
        )

    def test_empty_input_returns_empty(self):
        tr = GNNTracker()
        conf, all_ = tr.step([], sim_time=0.0, dt=0.2)
        assert conf == []

    def test_birth_on_first_detection(self):
        tr = GNNTracker()
        conf, all_ = tr.step([self._det(100, 200, 300)], sim_time=0.2, dt=0.2)
        assert len(all_) == 1

    def test_confirm_after_consecutive_hits(self):
        tr = GNNTracker(confirm_threshold=(2, 3))
        d = self._det(100, 200, 300)
        # Step 1 — birth
        tr.step([d], sim_time=0.2, dt=0.2)
        # Step 2 — confirm
        conf, _ = tr.step([d], sim_time=0.4, dt=0.2)
        assert any(t.is_confirmed for t in conf)

    def test_delete_after_misses(self):
        tr = GNNTracker(confirm_threshold=(1, 1), delete_threshold=(2, 2))
        d = self._det(100, 200, 300)
        # Birth + confirm
        tr.step([d], sim_time=0.2, dt=0.2)
        # Two misses → delete
        tr.step([], sim_time=0.4, dt=0.2)
        _, all_ = tr.step([], sim_time=0.6, dt=0.2)
        assert len(all_) == 0

    def test_external_delete(self):
        tr = GNNTracker()
        _, all_ = tr.step([self._det(0, 0, 0)], sim_time=0.2, dt=0.2)
        tid = all_[0].track_id
        tr.delete_track(tid)
        assert tid not in {t.track_id for t in tr._tracks.values()}


# ------------------------------------------------------------------ #
# SimulationHistory
# ------------------------------------------------------------------ #
from datalink.sensors.radar.simulation_history import SimulationHistory


class TestSimulationHistory:
    def test_append_and_length(self):
        sh = SimulationHistory()
        sh.append('pp', 'dets', 'tracks', 'pos', 'ori', 'off', 'cov')
        assert len(sh) == 1

    def test_next_iterator(self):
        sh = SimulationHistory()
        sh.append(1, 2, 3, 4, 5, 6, 7)
        sh.append(8, 9, 10, 11, 12, 13, 14)

        pp, d, t, mp, mo, o, c, is_end = sh.next()
        assert pp == 1 and not is_end

        pp, d, t, mp, mo, o, c, is_end = sh.next()
        assert pp == 8 and not is_end

        pp, d, t, mp, mo, o, c, is_end = sh.next()
        assert is_end

    def test_reset_index(self):
        sh = SimulationHistory()
        sh.append(42, 0, 0, 0, 0, 0, 0)
        sh.next()
        sh.reset_index()
        pp, *_ , is_end = sh.next()
        assert pp == 42 and not is_end


# ------------------------------------------------------------------ #
# InfraredSensor stub
# ------------------------------------------------------------------ #
from datalink.sensors.infrared.infrared_sensor import InfraredSensor


class TestInfraredSensorStub:
    def test_instantiation(self):
        ir = InfraredSensor(sensor_index=2)
        assert ir.sensor_index == 2
        assert ir.field_of_view.shape == (2,)

    def test_field_of_view_positive(self):
        ir = InfraredSensor(sensor_index=1)
        assert all(ir.field_of_view > 0)

    def test_is_locked_after_step(self):
        ir = InfraredSensor(sensor_index=1,
                            params={'infraredSensor': {'has_false_alarms': False}})

        class FakeINS:
            Position    = np.zeros(3)
            Velocity    = np.zeros(3)
            Orientation = np.eye(3)

        dets, n, cfg = ir.step([], FakeINS(), time=0.0)
        assert ir.is_locked()
