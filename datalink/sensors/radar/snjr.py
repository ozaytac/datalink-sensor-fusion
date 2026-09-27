"""
Radar SNR / SNJR equation.
"""

import math
from typing import Sequence


_C = 3.0e8      # speed of light [m/s]
_K = 1.38e-23   # Boltzmann constant [J/K]


def radar_snjr_db(
    pt: float,
    freq: float,
    sigma: float,
    te: float,
    loss_db: float,
    range_m: float,
    tau: float,
    nf_db: float,
    jammers: Sequence,
    radar_theta: float,
    radar_phi: float,
    theta_3db: float,
    phi_3db: float,
    main_beam_gain_linear: float,
    sidelobe_gain_db: float,
) -> float:
    """
    Compute SNR/SNJR for a single target [dB].

    Parameters
    ----------
    pt                  : Peak power [W]
    freq                : Center frequency [Hz]
    sigma               : Target RCS [m²]
    te                  : Effective noise temperature [K]
    loss_db             : Radar losses [dB]
    range_m             : Target range [m]
    tau                 : Pulse width [s]
    nf_db               : Noise figure [dB]
    jammers             : list of Jammer objects (may be empty)
    radar_theta         : Current beam azimuth [rad]
    radar_phi           : Current beam elevation [rad]
    theta_3db           : 3-dB azimuth beamwidth [rad]
    phi_3db             : 3-dB elevation beamwidth [rad]
    main_beam_gain_linear : antenna gain (linear)
    sidelobe_gain_db    : sidelobe gain [dB]

    Returns
    -------
    snr : float  — SNR [dB]
    """
    lam    = _C / freq
    b      = 1.0 / tau
    g_db   = 10.0 * math.log10(main_beam_gain_linear)

    num = (g_db + g_db
           + 10.0 * math.log10(lam**2)
           + 10.0 * math.log10(sigma)
           + 10.0 * math.log10(pt))

    den_ = (10.0 * math.log10((4.0 * math.pi)**3)
            + 10.0 * math.log10(_K)
            + 10.0 * math.log10(b)
            + nf_db
            + loss_db
            + 10.0 * math.log10(range_m**4))

    # Jammer noise temperature accumulation
    sidelobe_gain_lin = 10.0 ** (sidelobe_gain_db / 10.0)
    sum_noise_temp = te
    for jam in jammers:
        ang_diff_az = abs(jam.azimuth_to_body - radar_theta)
        ang_diff_el = abs(jam.elevation_to_body - radar_phi)
        if ang_diff_az <= theta_3db and ang_diff_el <= phi_3db:
            g_on_jammer = main_beam_gain_linear
        else:
            g_on_jammer = sidelobe_gain_lin
        lossj_lin = 10.0 ** (jam.jammer_losses / 10.0)
        gj_lin    = 10.0 ** (jam.antenna_gain  / 10.0)
        j_noise = (jam.transmitter_power * gj_lin * g_on_jammer * lam**2) / \
                  (16.0 * math.pi**2 * _K * jam.bandwidth * lossj_lin * jam.range_to_body**2)
        sum_noise_temp += j_noise

    den = den_ + 10.0 * math.log10(sum_noise_temp)
    return num - den
