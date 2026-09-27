"""
Tests for datalink.transformations  (L1 layer)

Strategy: cross-check against hand-computed expected values, verify
round-trip consistency (cart→msc→cart), and check Jacobian accuracy
via finite differences.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from datalink.transformations import (
    cartesian_to_msc,
    msc_to_cartesian,
    cart2sph_gradient,
    cartesian_to_msc_gradient,
    msc_to_cartesian_gradient,
    const_vel_jac,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A simple state: target directly ahead at 1000 m, moving east at 10 m/s
# x=1000 vx=10 y=0 vy=0 z=0 vz=0
CART_SIMPLE = np.array([1000.0, 10.0, 0.0, 0.0, 0.0, 0.0])

# A richer state with all components non-zero
CART_FULL = np.array([600.0, 5.0, 800.0, -3.0, 200.0, 2.0])


# ---------------------------------------------------------------------------
# cartesian_to_msc  /  msc_to_cartesian
# ---------------------------------------------------------------------------

class TestMSCRoundTrip:
    def test_simple_round_trip(self):
        msc  = cartesian_to_msc(CART_SIMPLE)
        cart = msc_to_cartesian(msc)
        assert_allclose(cart, CART_SIMPLE, atol=1e-10)

    def test_full_round_trip(self):
        msc  = cartesian_to_msc(CART_FULL)
        cart = msc_to_cartesian(msc)
        assert_allclose(cart, CART_FULL, atol=1e-10)

    def test_batch_round_trip(self):
        # 4 random states stacked column-wise
        rng = np.random.default_rng(0)
        X = rng.standard_normal((6, 4)) * np.array([1000, 5, 1000, 5, 500, 5])[:, None]
        msc  = cartesian_to_msc(X)
        cart = msc_to_cartesian(msc)
        assert_allclose(cart, X, atol=1e-9)

    def test_simple_known_az(self):
        # Target at (1, 0, 0): azimuth should be 0
        msc = cartesian_to_msc(np.array([1.0, 0, 0, 0, 0, 0]))
        assert_allclose(msc[0], 0.0, atol=1e-12)   # az = 0

    def test_simple_known_invR(self):
        # Target at distance 500 m along x: invR = 1/500
        msc = cartesian_to_msc(np.array([500.0, 0, 0, 0, 0, 0]))
        assert_allclose(msc[4], 1.0/500.0, atol=1e-12)

    def test_output_shape_scalar(self):
        msc = cartesian_to_msc(CART_SIMPLE)
        assert msc.shape == (6,)

    def test_output_shape_batch(self):
        X = np.column_stack([CART_SIMPLE, CART_FULL])
        msc = cartesian_to_msc(X)
        assert msc.shape == (6, 2)


# ---------------------------------------------------------------------------
# const_vel_jac
# ---------------------------------------------------------------------------

class TestConstVelJac:
    def test_identity_at_dt0(self):
        F, G = const_vel_jac(0.0)
        assert_allclose(F, np.eye(6), atol=1e-15)

    def test_propagation_matches_manual(self):
        dt = 0.2
        F, _ = const_vel_jac(dt)
        # x_new = x + vx*dt for position components
        x = np.array([100.0, 5.0, 200.0, -3.0, 50.0, 1.0])
        x_new = F @ x
        assert_allclose(x_new[0], x[0] + x[1]*dt, atol=1e-12)
        assert_allclose(x_new[2], x[2] + x[3]*dt, atol=1e-12)
        assert_allclose(x_new[4], x[4] + x[5]*dt, atol=1e-12)

    def test_shapes(self):
        F, G = const_vel_jac(0.1)
        assert F.shape == (6, 6)
        assert G.shape == (6, 3)

    def test_noise_matrix_structure(self):
        dt = 0.5
        _, G = const_vel_jac(dt)
        # Each diagonal 2×1 block should be [dt²/2; dt]
        assert_allclose(G[0, 0], dt**2/2, atol=1e-15)
        assert_allclose(G[1, 0], dt,      atol=1e-15)


# ---------------------------------------------------------------------------
# Jacobian accuracy via finite differences
# ---------------------------------------------------------------------------

EPS = 1e-5

def _fd_jacobian(f, x):
    """Numerical Jacobian of f at x using central differences."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    f0 = np.asarray(f(x), dtype=float)
    m  = f0.size
    J  = np.zeros((m, n))
    for j in range(n):
        xp, xm = x.copy(), x.copy()
        xp[j] += EPS
        xm[j] -= EPS
        J[:, j] = (np.asarray(f(xp)) - np.asarray(f(xm))) / (2*EPS)
    return J


class TestGradientFiniteDiff:
    def test_cartesian_to_msc_gradient(self):
        J_analytic = cartesian_to_msc_gradient(CART_FULL)
        J_numerical = _fd_jacobian(cartesian_to_msc, CART_FULL)
        assert_allclose(J_analytic, J_numerical, atol=1e-5, rtol=1e-4)

    def test_msc_to_cartesian_gradient(self):
        msc = cartesian_to_msc(CART_FULL)
        J_analytic  = msc_to_cartesian_gradient(msc)
        J_numerical = _fd_jacobian(msc_to_cartesian, msc)
        # The invR column produces ~1/invR² ≈ 1e6 values; central-diff
        # truncation error is O(eps²·f''') ≈ 60 at this scale → rtol=5e-4.
        assert_allclose(J_analytic, J_numerical, atol=1e-4, rtol=5e-4)

    def test_cart2sph_gradient(self):
        J_analytic  = cart2sph_gradient(CART_FULL)
        # cart2sph maps cart → [az, omega, el, elDot]
        def f(x):
            msc = cartesian_to_msc(x)
            return msc[:4]   # az, omega, el, elDot
        J_numerical = _fd_jacobian(f, CART_FULL)
        assert_allclose(J_analytic, J_numerical, atol=1e-4, rtol=1e-3)

    def test_jacobian_product_is_near_identity(self):
        """J_c2m · J_m2c ≈ I  (should be exact inverse pair at same point)."""
        J_c2m = cartesian_to_msc_gradient(CART_FULL)
        msc   = cartesian_to_msc(CART_FULL)
        J_m2c = msc_to_cartesian_gradient(msc)
        assert_allclose(J_c2m @ J_m2c, np.eye(6), atol=1e-8)
