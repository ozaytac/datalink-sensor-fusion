"""
Tracking Radar — generates noisy pseudo-measurements for each target.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Optional
from .radar_beam import RadarBeam
from .clutter import Clutter
from .snjr import radar_snjr_db
from .target import Target
from .detection import Detection


_C  = 3e8
_KB = 1.380649e-23


def _detectability(pd: float, pfa: float, n_pulses: int, swerling: str) -> float:
    """
    Required SNR for given Pd/Pfa (Albersheim approximation).
    Returns SNR in dB.
    """
    A = math.log(0.62 / pfa)
    B = math.log(pd / (1 - pd))
    snr_lin = A + 0.12 * A * B + 1.7 * B
    return 10.0 * math.log10(snr_lin)


class TrackingRadar:
    """
    Single-function tracking radar.

    Parameters
    ----------
    params : dict  (from params.json, trackingRadar section)
    """

    def __init__(self, params: dict):
        p = params['trackingRadar']
        cp = params['clutterParams']

        self.own_step         = 0
        self.continue_target_id = -1

        self.pulsewidth  = p['pulsewidth']
        self.theta_3db   = math.radians(p['theta_3db'])
        self.phi_3db     = math.radians(p['phi_3db'])
        self.theta_FOV_min = -math.radians(p['theta_FOV'])
        self.theta_FOV_max =  math.radians(p['theta_FOV'])
        self.phi_FOV_min   = -math.radians(p['phi_FOV'])
        self.phi_FOV_max   =  math.radians(p['phi_FOV'])

        self.beam = RadarBeam(0, 0, 0.0, 0.0, self.theta_3db, self.phi_3db)

        self.peak_power               = p['peak_power']
        self.wavelength               = p['wavelength']
        self.PRF                      = p['PRF']
        self.center_freq              = _C / self.wavelength
        self.effective_noise_temp     = p['effective_noise_temperature']
        self.bandwidth                = 1.0 / self.pulsewidth
        self.noise_figure             = p['noise_figure']
        self.false_alarm_probability  = p['false_alarm_probability']
        self.detection_probability    = p['detection_probability']

        self.K_range = p['K_range']
        self.K_theta = p['K_theta']
        self.K_phi   = p['K_phi']
        self.L_range = p['L_range']
        self.L_theta = p['L_theta']
        self.L_phi   = p['L_phi']
        self.sidelobe_diff        = p['sidelobe_diff']
        self.sidelobe_diff_linear = 10.0 ** (self.sidelobe_diff / 10.0)

        self.range_resolution = _C / (2.0 * self.bandwidth)
        self.theta_resolution = self.theta_3db
        self.phi_resolution   = self.phi_3db

        self.range_bias  = p['range_bias']
        self.theta_bias  = p['theta_bias']
        self.phi_bias    = p['phi_bias']

        rad_eff = p['radiation_efficiency_factor']
        aper_eff = p['aperture_efficiency']
        directive_lin = (4.0 * math.pi) / (self.theta_3db * self.phi_3db)
        self.antenna_gain_linear = rad_eff * directive_lin
        directive_db  = 10.0 * math.log10(directive_lin)
        self.antenna_gain_db     = rad_eff * directive_db
        self.antenna_effective_aperture = \
            (self.antenna_gain_db * self.wavelength**2) / (4.0 * math.pi)
        self.antenna_physical_aperture  = \
            self.antenna_effective_aperture / aper_eff

        self.range_min    = _C * self.pulsewidth / 2.0
        self.range_max    = _C / (2.0 * self.PRF)
        self.number_of_cell = math.ceil(
            (self.range_max - self.range_min) / self.range_resolution)

        self.search_above_range_max = p['search_above_range_max']
        self.search_below_range_min = p['search_below_range_min']
        self.search_outside_3db     = p['search_outside_3db']
        self.search_above_snr       = p['search_above_snr']
        self.include_clutter        = p['include_clutter']

        clutter_p = {
            'resolution_in_surface': self.range_resolution / 2.0,
            'range_limit':           cp['range_limit'],
            'scatterer_distribution': 'auto',
            'sigma_0':               cp['sigma_0'],
            'tau':                   self.pulsewidth,
        }
        self.clutter_model = Clutter(clutter_p)

    # ------------------------------------------------------------------ #
    def set_FOV(self, theta_min_deg: float, theta_max_deg: float,
                phi_min_deg: float, phi_max_deg: float) -> None:
        self.theta_FOV_min = math.radians(theta_min_deg)
        self.theta_FOV_max = math.radians(theta_max_deg)
        self.phi_FOV_min   = math.radians(phi_min_deg)
        self.phi_FOV_max   = math.radians(phi_max_deg)

    # ------------------------------------------------------------------ #
    def calculate_pseudo_measurements(
        self,
        altitude: float,
        dcm_ned2body: np.ndarray,
        targets: List[Target],
        jammers: list,
        sim_time: float,
        delta_time: float,
    ) -> tuple[list, float]:
        """
        Generate noisy measurements for all visible targets.

        Returns
        -------
        detections : list of raw detection tuples
            Each element: (time, theta, phi, rho, snr, meas_noise, target_id)
        time_stamp : float
        """
        self.clutter_model.set_one_step_params(altitude, dcm_ned2body)

        is_nearby = self._check_nearby(targets)
        targets   = self._merge_nearby(targets, is_nearby)

        detections: list = []
        sim_start  = sim_time - delta_time
        time_stamp = sim_start
        measured_ids: list[int] = []
        do_second = True

        for target in targets:
            tid = target.ID
            if self.continue_target_id != -1 and tid != self.continue_target_id:
                continue
            elif tid == self.continue_target_id:
                self.continue_target_id = -1

            meas, time_stamp, done = self._measure_target(
                target, jammers, sim_time, sim_start, time_stamp,
                delta_time, detections)

            if done:
                measured_ids.append(tid)
            else:
                do_second = False
                self.continue_target_id = tid
                break

        # Second pass for targets not yet measured
        if len(measured_ids) != len(targets) and do_second:
            for target in targets:
                if target.ID in measured_ids:
                    continue
                _, time_stamp, done = self._measure_target(
                    target, jammers, sim_time, sim_start, time_stamp,
                    delta_time, detections)
                if done:
                    measured_ids.append(target.ID)
                else:
                    self.continue_target_id = target.ID
                    break

        if self.continue_target_id in measured_ids:
            self.continue_target_id = -1

        self.own_step += 1
        return detections, time_stamp

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _measure_target(self, target, jammers, sim_time, sim_start,
                        time_stamp, delta_time, detections):
        """Attempt to measure a single target. Returns (meas, time_stamp, done)."""
        theta_t, phi_t, rng_t = target.get_body_spherical()
        self.beam.change_beam_position_in_radians(theta_t, phi_t)
        rcs = target.sample_rcs()

        in_fov  = (self.theta_FOV_min <= theta_t <= self.theta_FOV_max and
                   self.phi_FOV_min   <= phi_t   <= self.phi_FOV_max)
        below_rmin = rng_t < self.range_min
        above_rmax = rng_t > self.range_max
        alpha, beta = theta_t - self.beam.theta, phi_t - self.beam.phi
        out_3db = abs(alpha) > self.theta_3db or abs(beta) > self.phi_3db

        det_factor = _detectability(
            self.detection_probability, self.false_alarm_probability,
            1, target.swerling_type)

        main_gain     = self.antenna_gain_linear * math.cos(self.beam.theta)
        slobe_gain_db = 10.0 * math.log10(
            (self.antenna_gain_linear / self.sidelobe_diff_linear)
            * (2.0 - math.cos(self.beam.theta)))

        snr = radar_snjr_db(
            self.peak_power, self.center_freq, rcs,
            self.effective_noise_temp, 0.10,
            rng_t, self.pulsewidth, self.noise_figure,
            jammers, self.beam.theta, self.beam.phi,
            self.theta_3db, self.phi_3db,
            main_gain, slobe_gain_db)

        if self.include_clutter:
            snr += self.clutter_model.calculate_scr_db(
                rcs, self.theta_3db, self.phi_3db, theta_t, phi_t, rng_t)

        will_measure = True
        if not in_fov:                                   will_measure = False
        if not self.search_above_snr and det_factor > snr:   will_measure = False
        if not self.search_below_range_min and below_rmin:   will_measure = False
        if not self.search_above_range_max and above_rmax:   will_measure = False
        if not self.search_outside_3db and out_3db:          will_measure = False

        done = True
        if will_measure:
            n_pulses  = self._n_pulses(theta_t, phi_t, rng_t, rcs, det_factor)
            dwell     = n_pulses / self.PRF
            time_stamp += dwell
            if time_stamp - sim_start >= delta_time:
                return [], time_stamp, False

            if np.random.rand() <= self.detection_probability:
                th_n, ph_n, rn_n = self._noise(alpha, beta, snr)
                th_f, ph_f, rn_f = self._fixed(alpha, beta)
                th_ns, ph_ns, rn_ns = self._noise_sigma(alpha, beta, snr)
                th_fs, ph_fs, rn_fs = self._fixed_sigma(alpha, beta)

                noise_mat = np.diag([
                    math.sqrt(th_ns**2 + th_fs**2),
                    math.sqrt(ph_ns**2 + ph_fs**2),
                    math.sqrt(rn_ns**2 + rn_fs**2),
                ])

                theta_m = theta_t + th_n + th_f + self.theta_bias
                phi_m   = phi_t   + ph_n + ph_f + self.phi_bias
                rng_m   = rng_t   + rn_n + rn_f + self.range_bias

                detections.append(
                    (sim_time, theta_m, phi_m, rng_m, snr, noise_mat, target.ID))

            # Chaff
            if target.chaff == 1:
                chaff = self.clutter_model.create_chaff(
                    target.target_velocity_global,
                    self.theta_3db, self.phi_3db, theta_t, phi_t, rng_t)
                meas_noise = np.diag([self.theta_resolution,
                                      self.phi_resolution,
                                      self.range_resolution])
                for row in chaff:
                    detections.append(
                        (sim_time, row[0], row[1], row[2], 1.0, meas_noise, target.ID))

            # Clutter returns
            if self.include_clutter and will_measure:
                clutters = self.clutter_model.create_surface_clutter(
                    self.theta_3db, self.phi_3db, self.range_resolution,
                    theta_t, phi_t, rng_t)
                mn = np.diag([self.theta_resolution, self.phi_resolution,
                              self.range_resolution])
                for row in clutters:
                    detections.append(
                        (sim_time, row[0], row[1], row[2], 1.0, mn, target.ID))

        # False alarms
        fa = self._false_alarms()
        mn = np.diag([self.theta_resolution, self.phi_resolution,
                      self.range_resolution])
        for row in fa:
            detections.append((sim_time, row[0], row[1], row[2], 1.0, mn, -1))

        return detections, time_stamp, done

    def _n_pulses(self, theta, phi, rho, rcs, det_factor):
        loss = abs(math.cos(theta))**(-3) * abs(math.cos(phi))**(-6)
        num  = (4*math.pi)**3 * _KB * self.effective_noise_temp
        num *= det_factor * rho**4 * loss
        den  = (self.peak_power * self.pulsewidth
                * self.antenna_gain_linear**2
                * self.wavelength**2 * rcs)
        return math.ceil(num / den)

    def _noise(self, alpha, beta, snr_db):
        snr = 10.0 ** (snr_db / 10.0)
        rs = self.range_resolution / (self.K_range * math.sqrt(2.0 * snr))
        ts = self.theta_resolution / (self.K_theta * math.cos(alpha) * math.sqrt(2.0 * snr))
        ps = self.phi_resolution   / (self.K_phi   * math.cos(beta)  * math.sqrt(2.0 * snr))
        return np.random.normal(0, ts), np.random.normal(0, ps), np.random.normal(0, rs)

    def _noise_sigma(self, alpha, beta, snr_db):
        snr = 10.0 ** (snr_db / 10.0)
        rs = self.range_resolution / (self.K_range * math.sqrt(2.0 * snr))
        ts = self.theta_resolution / (self.K_theta * math.cos(alpha) * math.sqrt(2.0 * snr))
        ps = self.phi_resolution   / (self.K_phi   * math.cos(beta)  * math.sqrt(2.0 * snr))
        return ts, ps, rs

    def _fixed(self, alpha, beta):
        ts = self.theta_resolution / (self.L_theta * math.cos(alpha))
        ps = self.phi_resolution   / (self.L_phi   * math.cos(beta))
        rs = self.range_resolution / self.L_range
        return np.random.normal(0, ts), np.random.normal(0, ps), np.random.normal(0, rs)

    def _fixed_sigma(self, alpha, beta):
        return (self.theta_resolution / (self.L_theta * math.cos(alpha)),
                self.phi_resolution   / (self.L_phi   * math.cos(beta)),
                self.range_resolution / self.L_range)

    def _false_alarms(self):
        fa = []
        if np.random.rand() <= self.false_alarm_probability:
            fa.append([
                np.random.uniform(self.theta_FOV_min, self.theta_FOV_max),
                np.random.uniform(self.phi_FOV_min,   self.phi_FOV_max),
                np.random.uniform(self.range_min,     self.range_max),
            ])
        return fa

    def _check_nearby(self, targets):
        n = len(targets)
        nearby = np.zeros((n, n))
        for i in range(n):
            ti, pi, ri = targets[i].get_body_spherical()
            for j in range(i + 1, n):
                tj, pj, rj = targets[j].get_body_spherical()
                if (abs(ri - rj) <= self.range_resolution and
                        abs(ti - tj) <= self.theta_resolution and
                        abs(pi - pj) <= self.phi_resolution):
                    nearby[i, j] = nearby[j, i] = 1
        return nearby

    def _merge_nearby(self, targets, nearby):
        n = len(targets)
        delete = set()
        for i in range(n):
            az, el, r = targets[i].get_body_spherical()
            div = 1
            for j in range(i + 1, n):
                if nearby[i, j] == 1:
                    az2, el2, r2 = targets[j].get_body_spherical()
                    az += az2; el += el2; r += r2
                    div += 1
                    delete.add(j)
            targets[i].set_body_spherical(az / div, el / div, r / div)
        return [t for k, t in enumerate(targets) if k not in delete]
