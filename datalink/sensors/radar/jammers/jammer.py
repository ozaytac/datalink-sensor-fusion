"""
Jammer model.
"""


class Jammer:
    """
    Electronic warfare jammer attached to a target.

    Parameters
    ----------
    params : dict with keys:
        transmitter_power  [W]
        antenna_gain       [dB]
        jammer_losses      [dB]
        bandwith           [Hz]   (note: key is intentionally spelled 'bandwith', not 'bandwidth')
        azimuth_to_body    [rad]
        elevation_to_body  [rad]
        range_to_body      [m]
    """

    def __init__(self, params: dict):
        self.transmitter_power  = params['transmitter_power']
        self.antenna_gain       = params['antenna_gain']        # dB
        self.jammer_losses      = params['jammer_losses']       # dB
        self.bandwidth          = params.get('bandwidth') or params.get('bandwith')
        self.azimuth_to_body    = params['azimuth_to_body']
        self.elevation_to_body  = params['elevation_to_body']
        self.range_to_body      = params['range_to_body']

        import math
        g_lin   = 10.0 ** (self.antenna_gain  / 10.0)
        l_lin   = 10.0 ** (self.jammer_losses / 10.0)
        self.ERP = (self.transmitter_power * g_lin) / l_lin
