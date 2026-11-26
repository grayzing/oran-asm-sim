"""Per-component O-RU power model and Advanced Sleep Modes, after [USMAN25].

* Eq. (2): P = sum_i P_i,ref * prod_x (x_act/x_ref)^{s_i,x} + P_amp, with the
  scaling exponents of Table II (digital) and Table III (analog).
* Eq. (4)/(5) + Table IV: CMOS technology-node scaling (identity at 5 nm).
* Eq. (6): P_amp = P_T / eta_amp(P_T), with a load-dependent efficiency and a
  quiescent bias term.
* Table VI: which sub-components are deactivated at SM1..SM4.
* Table I: deactivation / minimum sleep / activation durations.

The paper publishes per-unit totals (Table VII) but not each P_i,ref, so the
reference powers below are *calibrated* so that at the Table V operating point
the model reproduces Table VII exactly:

    state      RF unit   Digital   Total   (Table VII)
    full load   365 W     32 W     397 W   (365 / 32 / 397)
    SM1          61       27        88     (60-61 / 25-27 / 88)
    SM2          25       15        40     (23-25 / 13-15 / 40)
    SM3          18       10        28     (16-18 / 8-10  / 28)
    SM4          10        5        15     (8-10  / 3-5   / 15)

Components that Table VI never switches off (feedback VCO, control, network)
form the residual; in SM4 ("only wake-up functions retained") they are scaled
by a retention factor (RF 10/18, digital 5/10) to hit the SM4 row.

One O-RU sits on each site and controls its three sectors. Following the
independent-RF-chain assumption, every sector has its own 4T4R chain (PA,
analog FE, digital unit) and its own sleep state, so the power model and the
sleep states are applied per sector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import PowerConfig, RadioConfig

# ---------------- Table I: ASM characteristics (seconds)
ASM_TIMING = {  # level: (deactivation, minimum sleep, activation)
    1: (35.5e-6, 71e-6, 35.5e-6),
    2: (0.5e-3, 1e-3, 0.5e-3),
    3: (5e-3, 10e-3, 5e-3),
    4: (0.5, 1.0, 0.5),
}

# exponent order: BW, Ant, Load, Streams, Q, Mod
# (name, unit, P_ref [W], exponents, first SM level that deactivates it; 0 = never)
_COMPONENTS = [
    # ---- digital processing unit, Table II
    ("FFT/IFFT",          "DU", 7.0, (1, 1, 0, 0, 1.2, 0), 2),
    ("Beamforming",       "DU", 3.0, (1, 2, 0, 1, 1.2, 1), 1),
    ("Filtering",         "DU", 5.0, (1, 0, 0, 0, 1.2, 0), 2),
    ("Up/Down sampling",  "DU", 2.0, (1, 0, 1, 0, 1.0, 1), 1),
    ("DDC/DUC",           "DU", 2.0, (1, 0, 0, 0, 1.2, 1), 3),
    ("DPD",               "DU", 3.0, (1, 0, 1, 0, 1.2, 0), 3),
    ("Control",           "DU", 6.0, (0, 0.5, 0, 0.2, 0.2, 0), 0),
    ("Network",           "DU", 4.0, (1, 1, 0, 1, 0, 0), 0),
    # ---- analog front-end, Table III
    ("Predriver",         "RF", 4.5, (1, 0, 1, 0, 0, 1), 3),
    ("Modulator",         "RF", 2.5, (1, 0, 1, 0, 0, 1.2), 1),
    ("Buffer",            "RF", 2.5, (1, 0, 1, 0, 0, 0), 3),
    ("Freq. synthesis",   "RF", 4.0, (0, 0, 1, 0, 0, 0), 1),
    ("Feedback VCO",      "RF", 18.0, (1, 0, 1, 0, 0, 0), 0),
    ("Feedback mixer",    "RF", 1.5, (1, 0, 1, 0, 0, 0), 1),
    ("Clock generation",  "RF", 4.0, (1, 0, 0.5, 0, 1, 0), 2),
    ("DAC",               "RF", 11.0, (1, 0, 1, 0, 0, 1), 2),
    ("Feedback ADC",      "RF", 5.0, (1, 0, 1, 0, 0, 0), 2),   # "Feedback DAC" in Table VI
    ("LNA",               "RF", 4.0, (1, 0, 1, 0, 0, 1), 2),
    ("Second LNA",        "RF", 2.0, (1, 0, 1, 0, 0, 1), 2),
    ("Mixer",             "RF", 3.0, (1, 0, 1, 0, 0, 0), 1),
    ("VGA",               "RF", 2.0, (1, 0, 1, 0, 0, 0), 1),
    ("ADC",               "RF", 10.0, (1, 0, 1, 0, 1, 1), 2),
]
SM4_RETENTION = {"RF": 10.0 / 18.0, "DU": 5.0 / 10.0}

# Table IV technology nodes and per-generation digital power saving ([USMAN25] Sec. IV-A.2)
_NODES = np.array([65, 40, 28, 20, 14, 10, 7, 5, 3, 2], float)
_NODE_YEAR = np.array([2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026, 2028])


def intrinsic_efficiency_gops_w(node_nm: float) -> float:
    """Eq. (4) without the Q / overhead terms: 75 GOPS/W at 65 nm, 50 % power saving
    per node until 2016, 35 % afterwards. Log-interpolated between nodes."""
    eta = [75.0]
    for yr in _NODE_YEAR[1:]:
        f = 0.5 if yr <= 2016 else 0.35
        eta.append(eta[-1] / (1 - f))
    eta = np.array(eta)
    return float(np.exp(np.interp(-np.log(node_nm), -np.log(_NODES), np.log(eta))))


@dataclass
class PowerBreakdown:
    total: np.ndarray
    pa: np.ndarray
    rf: np.ndarray       # analog FE (excl. PA)
    du: np.ndarray       # digital processing unit


class ORUPowerModel:
    def __init__(self, cfg: PowerConfig, radio: RadioConfig):
        self.cfg, self.radio = cfg, radio
        self.names = [c[0] for c in _COMPONENTS]
        self.unit = np.array([c[1] for c in _COMPONENTS])
        self.p_ref = np.array([c[2] for c in _COMPONENTS])
        self.exp = np.array([c[3] for c in _COMPONENTS], float)
        self.off_from = np.array([c[4] for c in _COMPONENTS])

        # technology node scaling (identity at the 5 nm reference)
        dig = intrinsic_efficiency_gops_w(cfg.ref_node_nm) / intrinsic_efficiency_gops_w(cfg.node_nm)
        ana = 1.0 + (cfg.ref_node_nm / cfg.node_nm - 1.0) / 2.0          # Eq. (5) as printed
        node = np.where(self.unit == "DU", dig, ana)

        # static (non-load) scaling: BW, antennas, quantisation
        x = np.array([radio.bw_mhz / cfg.ref_bw_mhz, radio.n_tx_chains / cfg.ref_n_ant,
                      1.0, 1.0, cfg.q_bits / cfg.ref_q_bits, 1.0])
        self.base = self.p_ref * node * np.prod(x[None, :] ** self.exp, axis=1)   # [C]
        self.e_load = self.exp[:, 2]
        self.e_str = self.exp[:, 3]
        self.e_mod = self.exp[:, 5]

        self.p_max_chain = radio.tx_power_per_chain_w
        self.n_chain = radio.n_tx_chains

        # pre-computed sleep powers (Table VII convention: still-on components at reference load)
        self.sleep_power = {}
        self.sleep_breakdown = {}
        for k in (1, 2, 3, 4):
            on = (self.off_from == 0) | (self.off_from > k)
            p = self.base * on
            if k == 4:
                p = p * np.where(self.unit == "RF", SM4_RETENTION["RF"], SM4_RETENTION["DU"])
            self.sleep_breakdown[k] = (float(p[self.unit == "RF"].sum()), float(p[self.unit == "DU"].sum()))
            self.sleep_power[k] = float(p.sum())
        self.p_full = float(self.active(np.ones(1), np.full(1, cfg.ref_streams),
                                        np.full(1, cfg.ref_mod_bits)).total[0])
        self.p_idle = float(self.active(np.full(1, cfg.signalling_load)).total[0])

    # ------------------------------------------------------------------ #
    def pa_power(self, load):
        """Eq. (6) with eta(P_T) = eta_max (P_T/P_max)^gamma and a bias term, per sector."""
        c = self.cfg
        load = np.clip(np.asarray(load, float), 0.0, 1.0)
        pt = self.p_max_chain * load
        dyn = self.p_max_chain * load ** (1.0 - c.pa_gamma) / c.pa_eta_max
        return self.n_chain * (c.pa_quiescent_w + np.where(pt > 0, dyn, 0.0))

    def active(self, load, streams=None, mod_bits=None) -> PowerBreakdown:
        """Active-state power for PRB load in [0,1], vectorised over sectors."""
        c = self.cfg
        load = np.clip(np.asarray(load, float), 0.0, 1.0)
        if streams is None:
            streams = np.full(load.shape, float(self.radio.max_layers))
        if mod_bits is None:
            mod_bits = np.full(load.shape, float(c.ref_mod_bits))
        st = np.maximum(np.asarray(streams, float), 1.0) / c.ref_streams
        md = np.maximum(np.asarray(mod_bits, float), 2.0) / c.ref_mod_bits
        lf = np.where(self.e_load[None, :] > 0,
                      c.static_share + (1 - c.static_share) * load[:, None] ** self.e_load[None, :], 1.0)
        comp = self.base[None, :] * lf * st[:, None] ** self.e_str[None, :] * md[:, None] ** self.e_mod[None, :]
        rf = comp[:, self.unit == "RF"].sum(1)
        du = comp[:, self.unit == "DU"].sum(1)
        pa = self.pa_power(load)
        return PowerBreakdown(total=pa + rf + du, pa=pa, rf=rf, du=du)

    def table_vii(self) -> Dict[str, Dict[str, float]]:
        """Reproduce [USMAN25] Table VII at the Table V operating point."""
        full = self.active(np.ones(1), np.full(1, self.cfg.ref_streams), np.full(1, self.cfg.ref_mod_bits))
        out = {"Full load": {"RF": float(full.pa[0] + full.rf[0]), "Digital": float(full.du[0]),
                             "Total": float(full.total[0])}}
        for k in (1, 2, 3, 4):
            rf, du = self.sleep_breakdown[k]
            out[f"SM{k}"] = {"RF": rf, "Digital": du, "Total": rf + du}
        return out

    def component_table(self, load=1.0):
        lf = np.where(self.e_load > 0, self.cfg.static_share + (1 - self.cfg.static_share) * load ** self.e_load, 1)
        return {n: float(p) for n, p in zip(self.names, self.base * lf)}
