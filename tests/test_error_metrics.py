"""
L6 test suite — error_metrics package.

Tests cover:
  - ospa: empty sets, identical sets, single-point, cardinality mismatch,
    cut-off, labeling component, known analytical values
  - hausdorff: empty sets, symmetric, non-symmetric, directed
  - calculate_optimization_errors: empty history, matched tracks, NaN for unmatched
  - calculate_radar_ospa / calculate_ir_ospa / calculate_fusion_ospa wrappers
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from datalink.error_metrics import (
    ospa,
    ospa_cartesian,
    ospa_msc,
    hausdorff,
    directed_hausdorff,
    calculate_optimization_errors,
    calculate_radar_ospa,
    calculate_ir_ospa,
    calculate_fusion_ospa,
)
from datalink.sensors.radar.track import Track


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _cart_track(state: np.ndarray, tid: int = 1) -> Track:
    return Track(track_id=tid, state=state, state_covariance=np.eye(6))


def _msc_track(az: float, el: float, inv_r: float = 1e-4, tid: int = 1) -> Track:
    state = np.array([az, 0.0, el, 0.0, inv_r, 0.0])
    return Track(track_id=tid, state=state, state_covariance=np.eye(6))


def _truth_cart(pos: tuple, pid: int = 1) -> dict:
    x, y, z = pos
    return {
        'state': np.array([x, 0.0, y, 0.0, z, 0.0]),
        'PlatformID': pid,
    }


def _truth_msc(az: float, el: float, pid: int = 1) -> dict:
    return {
        'state': np.array([az, 0.0, el, 0.0, 1e-4, 0.0]),
        'PlatformID': pid,
    }


# ------------------------------------------------------------------ #
# OSPA — core function
# ------------------------------------------------------------------ #

class TestOSPACore:
    def test_both_empty_returns_zeros(self):
        total, loc, card, label = ospa(np.empty((0, 3)), np.empty((0, 3)))
        assert total == 0.0 and loc == 0.0 and card == 0.0

    def test_x_empty_returns_cutoff(self):
        Y = np.array([[1.0, 0.0, 0.0]])
        total, loc, card, label = ospa(np.empty((0, 3)), Y, c=100.0, p=2.0)
        assert total == pytest.approx(100.0)
        assert card  == pytest.approx(100.0)
        assert loc   == pytest.approx(0.0)

    def test_y_empty_returns_cutoff(self):
        X = np.array([[1.0, 0.0, 0.0]])
        total, loc, card, label = ospa(X, np.empty((0, 3)), c=50.0, p=2.0)
        assert total == pytest.approx(50.0)

    def test_identical_sets_returns_zero(self):
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        total, loc, card, label = ospa(X, X.copy(), c=100.0, p=2.0)
        assert total == pytest.approx(0.0, abs=1e-10)

    def test_known_single_point(self):
        # X = {0}, Y = {3} → distance = 3, c=100 → OSPA = 3
        X = np.array([[0.0, 0.0, 0.0]])
        Y = np.array([[3.0, 0.0, 0.0]])
        total, loc, card, label = ospa(X, Y, c=100.0, p=2.0)
        assert total == pytest.approx(3.0, rel=1e-6)
        assert card  == pytest.approx(0.0, abs=1e-10)

    def test_cutoff_caps_distance(self):
        # Distance = 1000 > c=50 → OSPA = c = 50
        X = np.array([[0.0, 0.0, 0.0]])
        Y = np.array([[1000.0, 0.0, 0.0]])
        total, _, _, _ = ospa(X, Y, c=50.0, p=2.0)
        assert total == pytest.approx(50.0, rel=1e-6)

    def test_cardinality_penalty(self):
        # m=1, n=3, c=10, p=2: card = (10^2 * 2 / 3)^0.5
        X = np.array([[0.0, 0.0, 0.0]])
        Y = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])
        _, loc, card, _ = ospa(X, Y, c=10.0, p=2.0)
        expected_card = (10.0 ** 2 * 2 / 3) ** 0.5
        assert card == pytest.approx(expected_card, rel=1e-6)
        assert loc  == pytest.approx(0.0, abs=1e-10)

    def test_labeling_component_correct_ids(self):
        X = np.array([[0.0, 0.0, 0.0]])
        Y = np.array([[0.0, 0.0, 0.0]])
        _, _, _, label = ospa(X, Y, c=10.0, p=2.0,
                              x_ids=np.array([1]), y_ids=np.array([1]))
        assert label == pytest.approx(0.0, abs=1e-10)

    def test_labeling_component_wrong_ids(self):
        X = np.array([[0.0, 0.0, 0.0]])
        Y = np.array([[0.0, 0.0, 0.0]])
        _, _, _, label = ospa(X, Y, c=10.0, p=2.0,
                              x_ids=np.array([1]), y_ids=np.array([99]))
        assert label > 0.0

    def test_symmetric_for_equal_sets(self):
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        Y = np.array([[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]])
        t1, _, _, _ = ospa(X, Y, c=100.0, p=2.0)
        t2, _, _, _ = ospa(Y, X, c=100.0, p=2.0)
        assert t1 == pytest.approx(t2, rel=1e-10)

    def test_components_add_up(self):
        X = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        Y = np.array([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0],
                      [100.0, 0.0, 0.0]])
        total, loc, card, label = ospa(X, Y, c=50.0, p=2.0)
        recon = (loc**2 + card**2 + label**2) ** 0.5
        assert total == pytest.approx(recon, rel=1e-6)


# ------------------------------------------------------------------ #
# OSPA wrappers
# ------------------------------------------------------------------ #

class TestOSPAWrappers:
    def test_cartesian_empty(self):
        total, _, card, _ = ospa_cartesian([], [], c=100.0)
        assert total == 0.0

    def test_cartesian_one_each_matched(self):
        trk = _cart_track(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        tru = _truth_cart((0.0, 0.0, 0.0))
        total, loc, card, _ = ospa_cartesian([trk], [tru], c=100.0)
        assert total == pytest.approx(0.0, abs=1e-10)

    def test_cartesian_position_error(self):
        trk = _cart_track(np.array([3.0, 0.0, 4.0, 0.0, 0.0, 0.0]))
        tru = _truth_cart((0.0, 0.0, 0.0))
        total, _, _, _ = ospa_cartesian([trk], [tru], c=100.0, p=2.0)
        assert total == pytest.approx(5.0, rel=1e-5)

    def test_msc_empty(self):
        total, _, _, _ = ospa_msc([], [])
        assert total == 0.0

    def test_msc_identical_angles(self):
        trk = _msc_track(az=0.5, el=0.3)
        tru = _truth_msc(az=0.5, el=0.3)
        total, _, _, _ = ospa_msc([trk], [tru], c=0.1)
        assert total == pytest.approx(0.0, abs=1e-10)

    def test_msc_angle_error_below_cutoff(self):
        trk = _msc_track(az=0.0, el=0.0)
        tru = _truth_msc(az=0.05, el=0.0)   # 0.05 rad < c=0.1
        total, _, card, _ = ospa_msc([trk], [tru], c=0.1, p=2.0)
        assert card == pytest.approx(0.0, abs=1e-10)
        assert total == pytest.approx(0.05, rel=1e-5)

    def test_msc_angle_error_above_cutoff(self):
        trk = _msc_track(az=0.0, el=0.0)
        tru = _truth_msc(az=10.0, el=0.0)   # >> c=0.1
        total, _, _, _ = ospa_msc([trk], [tru], c=0.1, p=2.0)
        assert total == pytest.approx(0.1, rel=1e-5)


# ------------------------------------------------------------------ #
# Hausdorff
# ------------------------------------------------------------------ #

class TestHausdorff:
    def test_empty_returns_zero(self):
        hd, D = hausdorff(np.empty((0, 2)), np.empty((0, 2)))
        assert hd == 0.0

    def test_identical_sets_zero(self):
        P = np.array([[1.0, 2.0], [3.0, 4.0]])
        hd, _ = hausdorff(P, P.copy())
        assert hd == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        # P={(0,0)}, Q={(3,4)} → dist=5
        P = np.array([[0.0, 0.0]])
        Q = np.array([[3.0, 4.0]])
        hd, D = hausdorff(P, Q)
        assert hd == pytest.approx(5.0, rel=1e-6)
        assert D.shape == (1, 1)

    def test_symmetric(self):
        P = np.array([[0.0, 0.0], [1.0, 1.0]])
        Q = np.array([[0.5, 0.5], [2.0, 2.0]])
        hd_pq, _ = hausdorff(P, Q)
        hd_qp, _ = hausdorff(Q, P)
        assert hd_pq == pytest.approx(hd_qp, rel=1e-10)

    def test_asymmetric_directed(self):
        # P = {0}, Q = {0, 10}
        # dhd(P,Q) = 0 (P[0] matches Q[0] exactly)
        # dhd(Q,P) = 10 (Q[1] is far from any P point)
        P = np.array([[0.0]])
        Q = np.array([[0.0], [10.0]])
        d_pq = directed_hausdorff(P, Q)
        d_qp = directed_hausdorff(Q, P)
        assert d_pq == pytest.approx(0.0, abs=1e-10)
        assert d_qp == pytest.approx(10.0, rel=1e-6)

    def test_distance_matrix_shape(self):
        P = np.random.rand(4, 3)
        Q = np.random.rand(6, 3)
        hd, D = hausdorff(P, Q)
        assert D.shape == (4, 6)

    def test_wrong_dimensions_raises(self):
        P = np.array([[1.0, 2.0]])
        Q = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError):
            hausdorff(P, Q)


# ------------------------------------------------------------------ #
# calculate_optimization_errors
# ------------------------------------------------------------------ #

class TestCalculateOptimizationErrors:
    def _make_ownship_data(self):
        t = np.linspace(0.0, 10.0, 100)
        return types.SimpleNamespace(
            time=t,
            latitude=np.zeros(100),
            longitude=np.zeros(100),
            altitude=np.zeros(100),
            heading=np.zeros(100),
            pitch=np.zeros(100),
            roll=np.zeros(100),
        )

    def test_empty_history_returns_empty(self):
        hist = {'time': [], 'ground_truth': [], 'fusion_history': []}
        result = calculate_optimization_errors(hist, self._make_ownship_data())
        assert all(arr.size == 0 for arr in result)

    def test_unmatched_track_gives_nan(self):
        hist = {
            'time': [1.0],
            'ground_truth': [[{
                'PlatformID': 99,
                'CartesianPositionNED': np.array([100.0, 200.0, -50.0]),
                'SphericalPositionNED': np.array([0.5, 0.3, 230.0]),
            }]],
            'fusion_history': [{
                'FusionID':          [1],
                'RadarID':           [1],     # does not match PlatformID=99
                'InfraredID':        [-1],
                'FusedState':        [np.array([100.0, 0.0, 200.0, 0.0, -50.0, 0.0])],
                'FusionUpdateTime':  [1.0],
            }],
        }
        _, pos_err, az_err, el_err = calculate_optimization_errors(
            hist, self._make_ownship_data())
        assert np.isnan(pos_err[0, 0])

    def test_matched_track_gives_finite_error(self):
        hist = {
            'time': [1.0],
            'ground_truth': [[{
                'PlatformID': 1,
                'CartesianPositionNED': np.array([100.0, 200.0, -50.0]),
                'SphericalPositionNED': np.array([0.5, 0.3, 230.0]),
            }]],
            'fusion_history': [{
                'FusionID':          [1],
                'RadarID':           [1],
                'InfraredID':        [-1],
                'FusedState':        [np.array([105.0, 0.0, 200.0, 0.0, -50.0, 0.0])],
                'FusionUpdateTime':  [1.0],
            }],
        }
        _, pos_err, az_err, el_err = calculate_optimization_errors(
            hist, self._make_ownship_data())
        assert np.isfinite(pos_err[0, 0])
        assert pos_err[0, 0] == pytest.approx(5.0, rel=1e-5)

    def test_output_shape(self):
        T, N = 2, 3
        gt_entry = [{
            'PlatformID': k + 1,
            'CartesianPositionNED': np.zeros(3),
        } for k in range(T)]
        fh_entry = {
            'FusionID':         [],
            'RadarID':          [],
            'InfraredID':       [],
            'FusedState':       [],
            'FusionUpdateTime': [],
        }
        hist = {
            'time':           list(range(N)),
            'ground_truth':   [gt_entry] * N,
            'fusion_history': [fh_entry] * N,
        }
        ts, pe, ae, ee = calculate_optimization_errors(
            hist, self._make_ownship_data())
        assert pe.shape == (T, N)


# ------------------------------------------------------------------ #
# OSPA wrappers (calculate_* functions)
# ------------------------------------------------------------------ #

class TestOSPAConvenienceWrappers:
    def test_radar_ospa_empty(self):
        total, loc, card, _ = calculate_radar_ospa([], [], c=100.0)
        assert total == 0.0

    def test_radar_ospa_matched(self):
        trk = _cart_track(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        tru = _truth_cart((0.0, 0.0, 0.0))
        total, _, _, _ = calculate_radar_ospa([trk], [tru])
        assert total == pytest.approx(0.0, abs=1e-10)

    def test_ir_ospa_empty(self):
        total, _, _, _ = calculate_ir_ospa([], [])
        assert total == 0.0

    def test_ir_ospa_matched(self):
        trk = _msc_track(az=0.3, el=0.2)
        tru = _truth_msc(az=0.3, el=0.2)
        total, _, _, _ = calculate_ir_ospa([trk], [tru], c=0.1)
        assert total == pytest.approx(0.0, abs=1e-10)

    def test_fusion_ospa_cardinality(self):
        # 1 track, 2 truths → card penalty
        trk = _cart_track(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        tru1 = _truth_cart((0.0, 0.0, 0.0), pid=1)
        tru2 = _truth_cart((0.0, 0.0, 0.0), pid=2)
        total, loc, card, _ = calculate_fusion_ospa([trk], [tru1, tru2], c=50.0)
        assert card > 0.0
        assert loc == pytest.approx(0.0, abs=1e-10)
