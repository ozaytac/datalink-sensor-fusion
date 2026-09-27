"""
Tests for datalink.sensors.infrared  (L1 layer)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from datalink.sensors.infrared import MSCEKF, init_cv_msc_ekf, init_msc_rp_ekf
from datalink.transformations import cartesian_to_msc, msc_to_cartesian


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(az_deg=30.0, el_deg=10.0, noise_deg=1.0):
    return {
        "measurement":       [az_deg, el_deg],
        "measurement_noise": (noise_deg**2) * np.eye(2),
    }


# ---------------------------------------------------------------------------
# MSCEKF basic
# ---------------------------------------------------------------------------

class TestMSCEKFInit:
    def test_state_shape(self):
        cart = np.array([600.0, 5.0, 800.0, -3.0, 200.0, 2.0])
        msc  = cartesian_to_msc(cart)
        ekf  = MSCEKF(state=msc, covariance=np.eye(6),
                      process_noise=np.eye(3), measurement_noise=np.eye(2))
        assert ekf.state.shape  == (6,)
        assert ekf.covariance.shape == (6, 6)

    def test_state_is_copy(self):
        msc = cartesian_to_msc(np.array([1000.0, 0, 0, 0, 0, 0]))
        ekf = MSCEKF(msc, np.eye(6), np.eye(3), np.eye(2))
        original = ekf.state.copy()
        msc[:] = 0
        assert_allclose(ekf.state, original)


class TestMSCEKFPredict:
    def test_predict_preserves_direction_static(self):
        # Stationary target at 1000 m along x, zero velocity — range and az stable
        cart = np.array([1000.0, 0, 0, 0, 0, 0])
        msc  = cartesian_to_msc(cart)
        ekf  = MSCEKF(msc, 1e6*np.eye(6), np.eye(3), np.eye(2))
        ekf.predict(0.2)
        # Az should still be ~0, El ~0
        assert_allclose(ekf.state[0], 0.0, atol=1e-10)   # az
        assert_allclose(ekf.state[2], 0.0, atol=1e-10)   # el

    def test_predict_increases_covariance(self):
        cart = np.array([600.0, 5.0, 800.0, -3.0, 200.0, 2.0])
        msc  = cartesian_to_msc(cart)
        P0   = np.eye(6) * 1e4
        ekf  = MSCEKF(msc, P0.copy(), np.eye(3), np.eye(2))
        trace_before = np.trace(ekf.covariance)
        ekf.predict(1.0)
        assert np.trace(ekf.covariance) > trace_before

    def test_predict_positive_definite_covariance(self):
        cart = np.array([600.0, 5.0, 800.0, -3.0, 200.0, 2.0])
        msc  = cartesian_to_msc(cart)
        ekf  = MSCEKF(msc, np.eye(6)*1e4, np.eye(3), np.eye(2))
        ekf.predict(0.5)
        eigvals = np.linalg.eigvalsh(ekf.covariance)
        assert np.all(eigvals > 0), f"Non-positive eigenvalues: {eigvals}"


class TestMSCEKFUpdate:
    def test_update_reduces_uncertainty(self):
        cart = np.array([600.0, 5.0, 800.0, -3.0, 200.0, 2.0])
        msc  = cartesian_to_msc(cart)
        ekf  = MSCEKF(msc, np.eye(6)*1e6, np.eye(3), 0.01*np.eye(2))
        trace_before = np.trace(ekf.covariance)
        ekf.update(np.array([msc[0], msc[2]]))
        assert np.trace(ekf.covariance) < trace_before

    def test_update_corrects_az_bias(self):
        cart = np.array([1000.0, 0, 0, 0, 0, 0])
        msc  = cartesian_to_msc(cart)
        # Introduce az bias
        msc_biased = msc.copy()
        msc_biased[0] += np.deg2rad(5)
        ekf = MSCEKF(msc_biased, np.eye(6)*1e4, np.eye(3), 1e-6*np.eye(2))
        ekf.update(np.array([msc[0], msc[2]]))
        # After update, az should be closer to truth
        assert abs(ekf.state[0] - msc[0]) < abs(msc_biased[0] - msc[0])


# ---------------------------------------------------------------------------
# init_cv_msc_ekf
# ---------------------------------------------------------------------------

class TestInitCVMSCEKF:
    def test_returns_mscekf(self):
        det = _make_detection()
        ekf = init_cv_msc_ekf(det)
        assert isinstance(ekf, MSCEKF)

    def test_az_matches_detection(self):
        det = _make_detection(az_deg=45.0, el_deg=5.0)
        ekf = init_cv_msc_ekf(det)
        assert_allclose(ekf.state[0], np.deg2rad(45.0), atol=1e-8)

    def test_el_matches_detection(self):
        det = _make_detection(az_deg=30.0, el_deg=15.0)
        ekf = init_cv_msc_ekf(det)
        assert_allclose(ekf.state[2], np.deg2rad(15.0), atol=1e-8)

    def test_custom_range(self):
        det = _make_detection()
        ekf = init_cv_msc_ekf(det, range_estimation=(5000.0, 500.0))
        assert_allclose(ekf.state[4], 1.0/5000.0, atol=1e-10)

    def test_default_range(self):
        det = _make_detection()
        ekf = init_cv_msc_ekf(det)
        assert_allclose(ekf.state[4], 1.0/3e4, atol=1e-12)

    def test_covariance_is_positive_definite(self):
        det = _make_detection()
        ekf = init_cv_msc_ekf(det)
        eigvals = np.linalg.eigvalsh(ekf.covariance)
        assert np.all(eigvals > 0), f"Non-PD covariance: {eigvals}"

    def test_process_noise_shape(self):
        det = _make_detection()
        ekf = init_cv_msc_ekf(det)
        assert ekf.process_noise.shape == (3, 3)


# ---------------------------------------------------------------------------
# init_msc_rp_ekf
# ---------------------------------------------------------------------------

class TestInitMSCRPEKF:
    def test_returns_list_of_mscekf(self):
        det     = _make_detection()
        filters = init_msc_rp_ekf(det)
        assert isinstance(filters, list)
        assert all(isinstance(f, MSCEKF) for f in filters)

    def test_default_num_filters(self):
        det     = _make_detection()
        filters = init_msc_rp_ekf(det)
        assert len(filters) == 3

    def test_custom_num_filters(self):
        det     = _make_detection()
        filters = init_msc_rp_ekf(det, num_filters=5)
        assert len(filters) == 5

    def test_ranges_are_increasing(self):
        det     = _make_detection()
        filters = init_msc_rp_ekf(det, r_min=1e3, r_max=1e5)
        # Larger filter index → larger range → smaller invR
        inv_ranges = [f.state[4] for f in filters]
        assert inv_ranges[0] > inv_ranges[-1]

    def test_velocity_covariance_scaled(self):
        # Velocity covariance is inflated 400x relative to the base filter
        det      = _make_detection()
        base_ekf = init_cv_msc_ekf(det, range_estimation=(8e3, 100.0))
        rp_ekf   = init_msc_rp_ekf(det, r_min=8e3, r_max=8e4, num_filters=1)[0]
        # Velocity sub-block should be scaled up
        assert np.trace(rp_ekf.covariance[1::2, 1::2]) > np.trace(base_ekf.covariance[1::2, 1::2])
