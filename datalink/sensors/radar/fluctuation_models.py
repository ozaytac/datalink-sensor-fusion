"""
Swerling RCS fluctuation models.
"""

import numpy as np
from scipy.stats import gamma


class FluctuationModel:
    """
    Swerling RCS model (0–5).

    Swerling 0 / 5 : non-fluctuating (constant RCS = sigma_avg)
    Swerling 1 / 2 : chi-squared DoF-2  (Gamma a=1, scale=sigma_avg)
    Swerling 3 / 4 : chi-squared DoF-4  (Gamma a=2, scale=sigma_avg/2)

    Parameters
    ----------
    swerling_type : str
        One of 'Swerling0'…'Swerling5'.
    sigma_avg : float
        Mean RCS [m²].  Must be > 0.
    """

    _VALID = {'Swerling0', 'Swerling1', 'Swerling2',
              'Swerling3', 'Swerling4', 'Swerling5'}

    def __init__(self, swerling_type: str, sigma_avg: float):
        if sigma_avg <= 0:
            raise ValueError("sigma_avg must be > 0")
        if swerling_type not in self._VALID:
            raise ValueError(f"Unknown Swerling type: {swerling_type!r}")

        self.sigma_avg = sigma_avg
        # Swerling5 is alias for Swerling0
        self.type = 'Swerling0' if swerling_type == 'Swerling5' else swerling_type

        if self.type in ('Swerling1', 'Swerling2'):
            # Gamma(a=1, scale=sigma_avg)  →  chi-sq DoF-2
            self._dist = gamma(a=1, scale=sigma_avg)
        elif self.type in ('Swerling3', 'Swerling4'):
            # Gamma(a=2, scale=sigma_avg/2) → chi-sq DoF-4
            self._dist = gamma(a=2, scale=sigma_avg / 2.0)
        else:
            self._dist = None   # Swerling0: non-fluctuating

    def sample(self) -> float:
        """Draw a single RCS sample [m²]."""
        if self._dist is None:
            return self.sigma_avg
        return float(self._dist.rvs())
