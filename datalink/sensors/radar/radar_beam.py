"""
Radar beam position model.
"""

import math


class RadarBeam:
    """
    Tracks the current beam pointing direction.

    Attributes
    ----------
    prf   : float  — pulse repetition frequency [Hz]
    tau   : float  — pulse width [s]
    theta : float  — azimuth pointing angle [rad]
    phi   : float  — elevation pointing angle [rad]
    theta_resolution : float — 3-dB azimuth beamwidth [rad]
    phi_resolution   : float — 3-dB elevation beamwidth [rad]
    """

    def __init__(self, prf: float, tau: float,
                 theta: float, phi: float,
                 theta_res: float, phi_res: float):
        self.prf   = prf
        self.tau   = tau
        self.theta = theta
        self.phi   = phi
        self.theta_resolution = theta_res
        self.phi_resolution   = phi_res

    def change_beam_position_in_radians(self, theta: float, phi: float) -> None:
        """Point beam to (theta, phi) [rad]."""
        self.theta = theta
        self.phi   = phi

    def move_beam_in_radians(self, d_theta: float, d_phi: float) -> None:
        """Increment beam position, wrapping to [-pi, pi]."""
        self.theta = _wrap(self.theta + d_theta)
        self.phi   = _wrap(self.phi   + d_phi)


def _wrap(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle
