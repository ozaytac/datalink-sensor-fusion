"""
Jacobians / gradient matrices for MSC transformations and constant-velocity dynamics.

Source files
------------
  +transformations/cart2sphGradient.m       → cart2sph_gradient        (4×6)
  +transformations/cartesianToMSCGradient.m → cartesian_to_msc_gradient (6×6)
  +transformations/mscToCartesianGradient.m → msc_to_cartesian_gradient (6×6)
  +transformations/constveljac.m            → const_vel_jac             (6×6, 6×3)
"""

import numpy as np
from scipy.linalg import block_diag


# ---------------------------------------------------------------------------
# cart2sph_gradient
# ---------------------------------------------------------------------------

def cart2sph_gradient(states: np.ndarray) -> np.ndarray:
    """
    Partial Jacobian from Cartesian to {az, omega, el, elDot}.

    Returns
    -------
    G : ndarray, shape (4, 6)
        Rows  : [az, omega, el, elDot]
        Cols  : [x, vx, y, vy, z, vz]

    Notes
    -----
    Elevation here is defined as -atan2(-z, sqrt(x²+y²)).
    """
    states = np.asarray(states, dtype=float).ravel()
    x, vx, y, vy, z, vz = states
    X, Vx, Y, Vy, Z, Vz = x, vx, y, vy, z, vz

    r2_xy = X**2 + Y**2
    r2    = r2_xy + Z**2

    # --- d(az)/d(state) ---
    daz = np.array([-Y/r2_xy, 0.0, X/r2_xy, 0.0, 0.0, 0.0])

    # --- d(el)/d(state)  [el is also called phi] ---
    # dphi_dx = -(x*z)/..., dphi_dz = +(x²+y²)^0.5/...
    dphi_dx = -(X*Z) / (r2_xy**0.5 * r2)
    dphi_dy = -(Y*Z) / (r2_xy**0.5 * r2)
    dphi_dz =  r2_xy**0.5 / r2
    del_ = np.array([dphi_dx, 0.0, dphi_dy, 0.0, dphi_dz, 0.0])

    # --- d(elDot)/d(state) ---
    dphidot_dx = (
        (2*Vz*X - Vx*Z) / (r2_xy**0.5 * r2)
        + (2*X*(Z*(Vx*X + Vy*Y) - Vz*r2_xy)) / (r2_xy**0.5 * r2**2)
        + (X*(Z*(Vx*X + Vy*Y) - Vz*r2_xy)) / (r2_xy**1.5 * r2)
    )
    d_phidot_dxdot = -(X*Z) / (r2_xy**0.5 * r2)
    dphidot_dy = (
        (2*Vz*Y - Vy*Z) / (r2_xy**0.5 * r2)
        + (2*Y*(Z*(Vx*X + Vy*Y) - Vz*r2_xy)) / (r2_xy**0.5 * r2**2)
        + (Y*(Z*(Vx*X + Vy*Y) - Vz*r2_xy)) / (r2_xy**1.5 * r2)
    )
    d_phidot_dydot = -(Y*Z) / (r2_xy**0.5 * r2)
    dphidot_dz = (
        (2*Z*(Z*(Vx*X + Vy*Y) - Vz*r2_xy)) / (r2_xy**0.5 * r2**2)
        - (Vx*X + Vy*Y) / (r2_xy**0.5 * r2)
    )
    d_phidot_dzdot = r2_xy**0.5 / r2
    deldot = np.array([
        dphidot_dx, d_phidot_dxdot,
        dphidot_dy, d_phidot_dydot,
        dphidot_dz, d_phidot_dzdot,
    ])

    # --- d(omega)/d(state) ---
    kappa = Z**2 / r2_xy + 1          # (Z²/(X²+Y²) + 1)
    d_omegadot_dx = (
        Vy / (kappa**0.5 * r2_xy)
        - 2*X*(Vy*X - Vx*Y) / (kappa**0.5 * r2_xy**2)
        + X*Z**2*(Vy*X - Vx*Y) / (kappa**1.5 * r2_xy**3)
    )
    d_omegadot_dxdot = -Y / (kappa**0.5 * r2_xy)
    d_omegadot_dy = (
        Y*Z**2*(Vy*X - Vx*Y) / (kappa**1.5 * r2_xy**3)
        - 2*Y*(Vy*X - Vx*Y) / (kappa**0.5 * r2_xy**2)
        - Vx / (kappa**0.5 * r2_xy)
    )
    d_omegadot_dydot =  X / (kappa**0.5 * r2_xy)
    d_omegadot_dz    = -Z*(Vy*X - Vx*Y) / (kappa**1.5 * r2_xy**2)
    domega = np.array([
        d_omegadot_dx, d_omegadot_dxdot,
        d_omegadot_dy, d_omegadot_dydot,
        d_omegadot_dz, 0.0,
    ])

    return np.vstack([daz, domega, del_, deldot])   # (4, 6)


# ---------------------------------------------------------------------------
# cartesian_to_msc_gradient
# ---------------------------------------------------------------------------

def cartesian_to_msc_gradient(cart: np.ndarray) -> np.ndarray:
    """
    Jacobian of cartesian_to_msc w.r.t. Cartesian state.

    Returns
    -------
    grad : ndarray, shape (6, 6)
        grad[i, j] = d(msc[i]) / d(cart[j])
        Rows : [az, omega, el, elDot, invR, dRinvR]
        Cols : [x, vx, y, vy, z, vz]

    Source: +transformations/cartesianToMSCGradient.m  (MathWorks 2018)
    """
    cart = np.asarray(cart, dtype=float).ravel()
    X, Vx = cart[0], cart[1]
    Y, Vy = cart[2], cart[3]
    Z, Vz = cart[4], cart[5]

    r2_xy = X**2 + Y**2
    r2    = r2_xy + Z**2

    # Each gradient is initially in [X,Y,Z,Vx,Vy,Vz] order, then
    # re-indexed to [x,vx,y,vy,z,vz] via idx = [0,3,1,4,2,5]
    idx = [0, 3, 1, 4, 2, 5]

    # d(invR)/d(state)
    dinvR = np.array([
        -X / r2**1.5,
        -Y / r2**1.5,
        -Z / r2**1.5,
        0.0, 0.0, 0.0,
    ])[idx]

    # d(az)/d(state)
    dAz = np.array([
        -Y / r2_xy,
         X / r2_xy,
         0.0, 0.0, 0.0, 0.0,
    ])[idx]

    # d(el)/d(state)
    dEl = np.array([
         (X*Z) / (r2_xy**0.5 * r2),
         (Y*Z) / (r2_xy**0.5 * r2),
        -r2_xy**0.5 / r2,
         0.0, 0.0, 0.0,
    ])[idx]

    # d(dRinvR)/d(state)
    dDrInvR = np.array([
        -(Vx*X**2 + 2*Vy*X*Y + 2*Vz*X*Z - Vx*Y**2 - Vx*Z**2) / r2**2,
        -(-Vy*X**2 + 2*Vx*X*Y + Vy*Y**2 + 2*Vz*Y*Z - Vy*Z**2) / r2**2,
        -(-Vz*X**2 + 2*Vx*X*Z - Vz*Y**2 + 2*Vy*Y*Z + Vz*Z**2) / r2**2,
        X / r2,
        Y / r2,
        Z / r2,
    ])[idx]

    # d(omega)/d(state)
    kappa = Z**2 / r2_xy + 1
    dOmega = np.array([
        Vy/(kappa**0.5*r2_xy) - 2*X*(Vy*X-Vx*Y)/(kappa**0.5*r2_xy**2)
            + X*Z**2*(Vy*X-Vx*Y)/(kappa**1.5*r2_xy**3),
        Y*Z**2*(Vy*X-Vx*Y)/(kappa**1.5*r2_xy**3)
            - 2*Y*(Vy*X-Vx*Y)/(kappa**0.5*r2_xy**2)
            - Vx/(kappa**0.5*r2_xy),
        -Z*(Vy*X-Vx*Y)/(kappa**1.5*r2_xy**2),
        -Y/(kappa**0.5*r2_xy),
         X/(kappa**0.5*r2_xy),
         0.0,
    ])[idx]

    # d(elDot)/d(state)
    common = Z*(X*Vx + Y*Vy) - Vz*r2_xy
    dElDot = np.array([
        -(2*X*Vz - Z*Vx)/(r2_xy**0.5*r2) - 2*X*common/(r2_xy**0.5*r2**2)
            - X*common/(r2_xy**1.5*r2),
        -(2*Y*Vz - Z*Vy)/(r2_xy**0.5*r2) - 2*Y*common/(r2_xy**0.5*r2**2)
            - Y*common/(r2_xy**1.5*r2),
        (X*Vx + Y*Vy)/(r2_xy**0.5*r2) - 2*Z*common/(r2_xy**0.5*r2**2),
        (X*Z) / (r2_xy**0.5 * r2),
        (Y*Z) / (r2_xy**0.5 * r2),
        -r2_xy**0.5 / r2,
    ])[idx]

    # Stack rows: [az, omega, el, elDot, invR, dRinvR]
    grad = np.vstack([dAz, dOmega, dEl, dElDot, dinvR, dDrInvR])

    # Flip sign on rows 2 and 3 (el → -el, elDot → -elDot)
    p = np.eye(6)
    p[2, 2] = -1.0
    p[3, 3] = -1.0
    return p @ grad


# ---------------------------------------------------------------------------
# msc_to_cartesian_gradient
# ---------------------------------------------------------------------------

def msc_to_cartesian_gradient(state: np.ndarray) -> np.ndarray:
    """
    Jacobian of msc_to_cartesian w.r.t. MSC state.

    Returns
    -------
    gradient : ndarray, shape (6, 6)
        gradient[i, j] = d(cart[i]) / d(msc[j])
        Rows : [x, vx, y, vy, z, vz]
        Cols : [az, omega, el, elDot, invR, dRinvR]

    Source: +transformations/mscToCartesianGradient.m  (MathWorks 2018)
    """
    state = np.asarray(state, dtype=float).ravel()
    invR   =  state[4]
    az     =  state[0]
    el     = -state[2]   # sign convention
    dRinvR =  state[5]
    omega  =  state[1]
    elDot  = -state[3]

    cosAzcosEl = np.cos(az)*np.cos(el)
    sinAzcosEl = np.sin(az)*np.cos(el)
    cosAzsinEl = np.cos(az)*np.sin(el)
    sinAzsinEl = np.sin(az)*np.sin(el)

    # Internal order: [1/r, az, el, rDot/r, omega, elRate]
    # → idx maps to [az, omega, el, elDot, invR, dRinvR]
    idx = [1, 4, 2, 5, 0, 3]

    dXbyZmsc = np.array([
        -cosAzcosEl / invR**2,
        -sinAzcosEl / invR,
        -cosAzsinEl / invR,
        0.0, 0.0, 0.0,
    ])[idx]

    dYbyZmsc = np.array([
        -sinAzcosEl / invR**2,
         cosAzcosEl / invR,
        -sinAzsinEl / invR,
        0.0, 0.0, 0.0,
    ])[idx]

    dZbyZmsc = np.array([
        np.sin(el) / invR**2,
        0.0,
        -np.cos(el) / invR,
        0.0, 0.0, 0.0,
    ])[idx]

    dVxbyZmsc = np.array([
        (-dRinvR*cosAzcosEl + omega*np.sin(az) + elDot*cosAzsinEl) / invR**2,
        -(dRinvR*sinAzcosEl + omega*np.cos(az) - elDot*sinAzsinEl) / invR,
        -(dRinvR*cosAzsinEl + elDot*cosAzcosEl) / invR,
         cosAzcosEl / invR,
        -np.sin(az) / invR,
        -cosAzsinEl / invR,
    ])[idx]

    dVybyZmsc = np.array([
        (-dRinvR*sinAzcosEl - omega*np.cos(az) + elDot*sinAzsinEl) / invR**2,
         (dRinvR*cosAzcosEl - omega*np.sin(az) - elDot*cosAzsinEl) / invR,
        -(dRinvR*sinAzsinEl + elDot*sinAzcosEl) / invR,
         sinAzcosEl / invR,
         np.cos(az) / invR,
        -sinAzsinEl / invR,
    ])[idx]

    dVzbyZmsc = np.array([
        (dRinvR*np.sin(el) + elDot*np.cos(el)) / invR**2,
        0.0,
        (-dRinvR*np.cos(el) + elDot*np.sin(el)) / invR,
        -np.sin(el) / invR,
        0.0,
        -np.cos(el) / invR,
    ])[idx]

    # Rows: [x, vx, y, vy, z, vz]
    gradient = np.vstack([dXbyZmsc, dVxbyZmsc, dYbyZmsc, dVybyZmsc, dZbyZmsc, dVzbyZmsc])

    p = np.eye(6)
    p[2, 2] = -1.0
    p[3, 3] = -1.0
    return gradient @ p


# ---------------------------------------------------------------------------
# const_vel_jac
# ---------------------------------------------------------------------------

def const_vel_jac(dt: float):
    """
    Jacobian of the constant-velocity Cartesian motion model.

    Returns
    -------
    dfdx : ndarray, shape (6, 6)  — state transition matrix F
    dfdw : ndarray, shape (6, 3)  — process noise input matrix G

    State order: [x, vx, y, vy, z, vz]

    Source: +transformations/constveljac.m
    """
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[dt**2 / 2.0], [dt]])
    dfdx = block_diag(A, A, A)          # 6×6
    dfdw = block_diag(B, B, B)          # 6×3
    return dfdx, dfdw
