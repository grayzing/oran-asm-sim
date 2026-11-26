"""3GPP TR 38.901 large-scale channel (UMa / UMi), stochastic LOS + spatially
correlated shadow fading, sector antenna pattern, O2I loss, and the 2x2 SU-MIMO
link abstraction (SVD precoding, rank adaptation, TS 38.214 CQI tables).

All functions are vectorised over [UE, site/sector, RBG].
"""
from __future__ import annotations

import numpy as np

from .config import RadioConfig

C0 = 299_792_458.0

try:  # standard normal CDF
    from scipy.special import ndtr as _ndtr
except Exception:  # pragma: no cover
    from math import erf
    _erf = np.vectorize(erf)

    def _ndtr(x):
        return 0.5 * (1.0 + _erf(np.asarray(x) / np.sqrt(2.0)))


def db2lin(x):
    return 10.0 ** (np.asarray(x) / 10.0)


def lin2db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-300))


# --------------------------------------------------------------------------- #
# TR 38.901 Table 7.4.1-1 / 7.4.2-1
# --------------------------------------------------------------------------- #
def pathloss_38901(scenario: str, d2d, h_bs, h_ut, fc_ghz):
    """Returns (PL_LOS_dB, PL_NLOS_dB, sigma_LOS, sigma_NLOS)."""
    d2d = np.asarray(d2d, float)
    d3d = np.sqrt(d2d ** 2 + (h_bs - h_ut) ** 2)
    hb, hu = h_bs - 1.0, h_ut - 1.0
    dbp = 4.0 * hb * hu * fc_ghz * 1e9 / C0
    lf = 20.0 * np.log10(fc_ghz)
    if scenario == "UMa":
        pl1 = 28.0 + 22.0 * np.log10(d3d) + lf
        pl2 = 28.0 + 40.0 * np.log10(d3d) + lf - 9.0 * np.log10(dbp ** 2 + (h_bs - h_ut) ** 2)
        pl_los = np.where(d2d <= dbp, pl1, pl2)
        pl_nlos_p = 13.54 + 39.08 * np.log10(d3d) + lf - 0.6 * (h_ut - 1.5)
        return pl_los, np.maximum(pl_los, pl_nlos_p), 4.0, 6.0
    if scenario == "UMi":
        pl1 = 32.4 + 21.0 * np.log10(d3d) + lf
        pl2 = 32.4 + 40.0 * np.log10(d3d) + lf - 9.5 * np.log10(dbp ** 2 + (h_bs - h_ut) ** 2)
        pl_los = np.where(d2d <= dbp, pl1, pl2)
        pl_nlos_p = 22.4 + 35.3 * np.log10(d3d) + 21.3 * np.log10(fc_ghz) - 0.3 * (h_ut - 1.5)
        return pl_los, np.maximum(pl_los, pl_nlos_p), 4.0, 7.82
    raise ValueError(scenario)


def los_probability_38901(scenario: str, d2d, h_ut=1.5):
    d = np.maximum(np.asarray(d2d, float), 1e-3)
    if scenario == "UMa":
        c = 0.0 if h_ut <= 13 else ((h_ut - 13) / 10) ** 1.5
        p = (18 / d + np.exp(-d / 63) * (1 - 18 / d)) * (1 + c * 1.25 * (d / 100) ** 3 * np.exp(-d / 150))
    else:
        p = 18 / d + np.exp(-d / 36) * (1 - 18 / d)
    return np.where(d <= 18, 1.0, np.clip(p, 0, 1))


def o2i_loss_db(rng, n, fc_ghz, high_loss_mask, return_parts=False):
    """TR 38.901 7.4.3.1 O2I building penetration (low/high-loss), per UE.
    With ``return_parts`` returns (PL_tw + sigma_P * N(0,1), d_2D-in) so a
    geometric indoor distance can replace the random one."""
    l_glass = 2 + 0.2 * fc_ghz
    l_irr = 23 + 0.3 * fc_ghz
    l_conc = 5 + 4 * fc_ghz
    low = 5 - 10 * np.log10(0.3 * 10 ** (-l_glass / 10) + 0.7 * 10 ** (-l_conc / 10))
    high = 5 - 10 * np.log10(0.7 * 10 ** (-l_irr / 10) + 0.3 * 10 ** (-l_conc / 10))
    d_in = np.minimum(rng.uniform(0, 25, n), rng.uniform(0, 25, n))
    sig = np.where(high_loss_mask, 6.5, 4.4)
    base = np.where(high_loss_mask, high, low) + sig * rng.standard_normal(n)
    if return_parts:
        return base, d_in
    return base + 0.5 * d_in


def sector_antenna_gain_db(radio: RadioConfig, phi_deg, theta_deg, tilt_deg=None):
    """TR 38.901 Table 7.3-1 pattern (with array gain folded into g_max)."""
    tilt = radio.downtilt_deg if tilt_deg is None else tilt_deg
    a_h = -np.minimum(12.0 * (phi_deg / radio.hpbw_h_deg) ** 2, radio.a_max_db)
    a_v = -np.minimum(12.0 * ((theta_deg - (90.0 + tilt)) / radio.hpbw_v_deg) ** 2,
                      radio.sla_v_db)
    return radio.g_max_dbi - np.minimum(-(a_h + a_v), radio.a_max_db)


# --------------------------------------------------------------------------- #
# TS 38.214 CQI tables (spectral efficiency per layer) + SINR thresholds
# --------------------------------------------------------------------------- #
_EFF_64 = np.array([0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141,
                    2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547])
_MOD_64 = np.array([2, 2, 2, 2, 2, 2, 4, 4, 4, 6, 6, 6, 6, 6, 6])
# 10 % BLER AWGN switching points commonly used for Table 5.2.2.1-2
_THR_64 = np.array([-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1,
                    16.3, 18.7, 21.0, 22.7])
_EFF_256 = np.array([0.1523, 0.3770, 0.8770, 1.4766, 1.9141, 2.4063, 2.7305, 3.3223,
                     3.9023, 4.5234, 5.1152, 5.5547, 6.2266, 6.9141, 7.4063])
_MOD_256 = np.array([2, 2, 2, 4, 4, 4, 6, 6, 6, 6, 6, 6, 8, 8, 8])


def _thr_from_eff(eff):
    thr = np.interp(eff, _EFF_64, _THR_64)
    hi = eff > _EFF_64[-1]
    thr[hi] = 10 * np.log10(2 ** (eff[hi] / 0.75) - 1)   # Shannon w/ 0.75 bw-efficiency
    return thr


CQI_TABLES = {
    "64QAM": (_THR_64, _EFF_64, _MOD_64),
    "256QAM": (_thr_from_eff(_EFF_256), _EFF_256, _MOD_256),
}


def sinr_to_eff(sinr_db, table="64QAM"):
    thr, eff, mod = CQI_TABLES[table]
    idx = np.searchsorted(thr, sinr_db, side="right")       # CQI index, 0 = out of range
    e = np.concatenate([[0.0], eff])[idx]
    m = np.concatenate([[0], mod])[idx]
    return e, m


# --------------------------------------------------------------------------- #
class LargeScaleChannel:
    """Per (UE, site) LOS state and shadow fading with AR(1) spatial correlation
    (Gudmundson); per (UE, sector) antenna gains. Produces received power per PRB."""

    def __init__(self, dep, radio: RadioConfig, n_ue: int, rng: np.random.Generator):
        self.dep, self.radio, self.rng = dep, radio, rng
        self.n_ue = n_ue
        n_site = dep.n_sites
        self.z_los = rng.standard_normal((n_ue, n_site))
        self.z_sf = rng.standard_normal((n_ue, n_site))
        self.prev_xy = None
        p_tot_dbm = 10 * np.log10(radio.tx_power_per_chain_w * radio.n_tx_chains * 1e3)
        self.p_tx_prb_dbm = p_tot_dbm - 10 * np.log10(radio.n_prb)
        self.noise_prb_w = db2lin(-174 + 10 * np.log10(12 * radio.scs_khz * 1e3) + radio.ue_nf_db) * 1e-3
        self.los = np.zeros((n_ue, n_site), bool)

    def reset_ue(self, idx):
        """Re-draw latents for re-spawned UEs (not used with torus mobility)."""
        self.z_los[idx] = self.rng.standard_normal((len(idx), self.z_los.shape[1]))
        self.z_sf[idx] = self.rng.standard_normal((len(idx), self.z_sf.shape[1]))

    def update(self, ue_xy: np.ndarray, extra_loss_db: np.ndarray) -> np.ndarray:
        """Returns received power per PRB [U, n_sectors] in W (no fast fading)."""
        r, dep = self.radio, self.dep
        d = dep.site_xy[None, :, :] - ue_xy[:, None, :]                  # UE -> site
        d2d = np.maximum(np.hypot(d[..., 0], d[..., 1]), r.min_dist_2d_m)

        # --- spatially-correlated latents (AR(1) in travelled distance)
        if self.prev_xy is not None:
            step = np.hypot(*(ue_xy - self.prev_xy).T)
            # torus wrap shows up as a huge jump: decorrelate fully (that is physical)
            rho_los = np.exp(-step / r.los_corr_dist_m)[:, None]
            self.z_los = rho_los * self.z_los + np.sqrt(1 - rho_los ** 2) * self.rng.standard_normal(self.z_los.shape)
            dc = np.where(self.los, r.sf_corr_dist_los_m, r.sf_corr_dist_nlos_m)
            rho_sf = np.exp(-step[:, None] / dc)
            self.z_sf = rho_sf * self.z_sf + np.sqrt(1 - rho_sf ** 2) * self.rng.standard_normal(self.z_sf.shape)
        self.prev_xy = ue_xy.copy()

        h_bs = dep.site_h[None, :] if dep.site_h is not None else r.h_bs_m
        if r.los_model == "geometry" and dep.site_vis is not None:
            # deterministic LOS from the cached per-site viewsheds of the imported buildings;
            # indoor UEs take the LOS state of their nearest outdoor cell (facade)
            idx = dep.buildings.outdoor_flat_index(ue_xy)
            self.los = dep.site_vis[:, idx].T
        else:
            self.los = _ndtr(self.z_los) < los_probability_38901(r.scenario, d2d, r.h_ut_m)

        pl_l, pl_n, s_l, s_n = pathloss_38901(r.scenario, d2d, h_bs, r.h_ut_m, r.fc_ghz)
        pl = np.where(self.los, pl_l, pl_n)
        if r.shadowing:
            pl = pl + np.where(self.los, s_l, s_n) * self.z_sf
        self.d2d_site = d2d

        # --- per-sector antenna gain
        ss = dep.sector_site
        bearing = np.degrees(np.arctan2(-d[..., 1], -d[..., 0]))          # site -> UE
        phi = (bearing[:, ss] - dep.sector_az_deg[None, :] + 180.0) % 360.0 - 180.0
        hb = dep.site_h if dep.site_h is not None else np.full(dep.n_sites, r.h_bs_m)
        theta = 90.0 + np.degrees(np.arctan2(hb[None, :] - r.h_ut_m, d2d))[:, ss]
        tilt = dep.site_tilt_deg[ss][None, :] if dep.site_tilt_deg is not None else None
        g = sector_antenna_gain_db(r, phi, theta, tilt)

        rx_dbm = self.p_tx_prb_dbm + g + r.ue_gain_dbi - pl[:, ss] - extra_loss_db[:, None]
        self.los_sector = self.los[:, ss]
        return db2lin(rx_dbm) * 1e-3


# --------------------------------------------------------------------------- #
# Small-scale fading + SU-MIMO link abstraction
# --------------------------------------------------------------------------- #
def sample_mimo_channel(rng, n_ue, n_rbg, nr, nt, k_lin):
    """Rician/Rayleigh i.i.d. per RBG. k_lin: [n_ue] Rician K (0 -> Rayleigh)."""
    s = np.float32(1 / np.sqrt(2))
    z = rng.standard_normal((2, n_ue, n_rbg, nr, nt), dtype=np.float32)
    hw = (s * z[0]) + 1j * (s * z[1])                  # complex64
    if np.any(k_lin > 0):
        # rank-1 LOS component from random ULA angles (half-wavelength spacing)
        st, sr = rng.uniform(-1, 1, (2, n_ue))
        at = np.exp(1j * np.pi * np.arange(nt)[None, :] * st[:, None])
        ar = np.exp(1j * np.pi * np.arange(nr)[None, :] * sr[:, None])
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, (n_ue, n_rbg)))
        hl = ar[:, None, :, None] * at[:, None, None, :].conj() * ph[..., None, None]
        a = np.sqrt(k_lin / (k_lin + 1))[:, None, None, None]
        b = np.sqrt(1 / (k_lin + 1))[:, None, None, None]
        hw = (a * hl + b * hw).astype(np.complex64)
    return hw


def mimo_eigs(h):
    """Descending eigenvalues of H H^H, closed form for 2 Rx, eigvalsh otherwise."""
    g = h @ np.conj(np.swapaxes(h, -1, -2))
    if g.shape[-1] == 2:
        a, d = g[..., 0, 0].real, g[..., 1, 1].real
        b2 = np.abs(g[..., 0, 1]) ** 2
        tr2 = 0.5 * (a + d)
        disc = np.sqrt(np.maximum(0.25 * (a - d) ** 2 + b2, 0))
        return np.stack([tr2 + disc, np.maximum(tr2 - disc, 0)], -1)
    return np.linalg.eigvalsh(g)[..., ::-1].clip(min=0)


def link_rates(rho, eigs, radio: RadioConfig, n_prb_rbg):
    """rho: [U,B] SNR-per-Rx-antenna (P_rx/(I+N)); eigs [U,B,L].

    Returns (bits per slot [U,B], rank [U], mod_bits [U,B], wideband SINR dB [U]).
    Rank is chosen per UE (wideband RI) to maximise mean rate over RBGs."""
    re_per_prb = 12 * 14 * (1 - radio.overhead)
    lmax = min(radio.max_layers, eigs.shape[-1])
    best = None
    for r_ in range(1, lmax + 1):
        sinr = rho[..., None] * eigs[..., :r_] / r_
        eff, mod = sinr_to_eff(lin2db(sinr), radio.cqi_table)
        tot = eff.sum(-1)
        score = tot.mean(-1)
        if best is None:
            best = [score, tot, np.full(rho.shape[0], r_), mod.max(-1)]
        else:
            m = score > best[0]
            best[0] = np.where(m, score, best[0])
            best[1] = np.where(m[:, None], tot, best[1])
            best[2] = np.where(m, r_, best[2])
            best[3] = np.where(m[:, None], mod.max(-1), best[3])
    bits = best[1] * (1 - radio.bler_target) * re_per_prb * n_prb_rbg[None, :]
    return bits, best[2], best[3]
