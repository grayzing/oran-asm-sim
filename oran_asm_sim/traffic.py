"""Synthetic downlink traffic: 3GPP FTP Model 3 (Poisson file arrivals of
0.5 MB, TR 36.814 / TR 38.901 evaluation methodology) per UE, modulated by a
24 h diurnal profile, per-UE lognormal activity heterogeneity and the
TVC period ("work" / "evening" etc.) traffic multiplier."""
from __future__ import annotations

import numpy as np

from .config import TrafficConfig


class FTP3Traffic:
    def __init__(self, cfg: TrafficConfig, n_ue: int, rng: np.random.Generator):
        self.cfg, self.n, self.rng = cfg, n_ue, rng
        a = rng.lognormal(0.0, cfg.activity_sigma, n_ue)
        self.activity = a / a.mean()

    def diurnal(self, t_s: float) -> float:
        d = np.asarray(self.cfg.diurnal, float)
        h = (t_s / 3600.0) % 24.0
        i = int(h) % 24
        f = h - int(h)
        return float((1 - f) * d[i] + f * d[(i + 1) % 24])

    def rate_files(self, t_s: float, period_mult: float = 1.0) -> np.ndarray:
        return self.cfg.peak_rate_files_per_s * self.diurnal(t_s) * period_mult * self.activity

    def arrivals(self, t_s: float, dt: float, period_mult: float = 1.0):
        """Returns (n_files [U], bits [U]) arriving in (t, t+dt]."""
        lam = self.rate_files(t_s, period_mult) * dt
        n = self.rng.poisson(lam)
        if self.cfg.file_size_dist == "lognormal":
            s = self.cfg.file_size_lognorm_sigma
            mu = np.log(self.cfg.file_size_bits) - s * s / 2
            # sum of n lognormals ~ approximated per-file draw scaled by n (cheap, mean-exact)
            bits = n * self.rng.lognormal(mu, s, self.n)
        else:
            bits = n * self.cfg.file_size_bits
        return n, bits.astype(float)

    def mean_offered_bps(self, t_s: float, period_mult: float = 1.0) -> np.ndarray:
        return self.rate_files(t_s, period_mult) * self.cfg.file_size_bits
