"""Configuration dataclasses for the O-RU ASM system-level simulator.

Every default is either taken from a 3GPP document (cited inline) or from the
two papers this simulator is built on:

  [USMAN25] M. Q. Usman et al., "Power Modeling of the O-RAN O-RU & Application
            of Advanced Sleep Modes for Enhanced Energy Efficiency", IEEE CCNC 2025.
  [HSU07]   W. Hsu et al., "Modeling Time-variant User Mobility in Wireless
            Mobile Networks", IEEE INFOCOM 2007.

All configs are plain dataclasses so they can be built from an RLlib
``env_config`` dict via :func:`SimConfig.from_dict`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, asdict, replace
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@dataclass
class GeometryConfig:
    # (south, west, north, east) in WGS-84 degrees. Default: Downtown Los Angeles
    # (Bunker Hill / Historic Core / Little Tokyo / Arts District / South Park).
    bbox: Tuple[float, float, float, float] = (34.0360, -118.2700, 34.0620, -118.2300)
    # Optional GeoJSON with the study-area polygon (UEs are confined to its bbox,
    # sites outside the polygon are dropped in "fill" layout).
    area_geojson: Optional[str] = None
    # Building footprints: GeoJSON FeatureCollection or raw Overpass JSON (see
    # tools/fetch_osm_buildings.py). Heights from "height"/"HEIGHT"/"building:height"
    # (metres, or feet if suffixed ' / ft), else building:levels * 3.5 m.
    # None -> a clearly-labelled SYNTHETIC DTLA-like city is generated (warning issued).
    buildings_geojson: Optional[str] = None
    building_height_field: Optional[str] = None
    building_height_scale: float = 1.0     # e.g. 0.3048 for LA County HEIGHT in feet
    building_default_height_m: float = 15.0
    building_raster_res_m: float = 5.0
    synthetic_seed: int = 0

    # Deployment
    # "viewshed": iterative max-visibility rooftop placement [MASRUR25] (needs buildings)
    # "fill": hexagonal lattice tiling the bbox; "rings": classic 19-site / 57-cell cluster
    layout: str = "viewshed"
    n_rings: int = 2
    isd_m: float = 500.0           # UMa ISD (TR 38.901 Table 7.8-1)
    center_latlon: Optional[Tuple[float, float]] = None  # default: bbox centre
    lattice_rotation_deg: float = 0.0
    sector_azimuths_deg: Tuple[float, float, float] = (30.0, 150.0, 270.0)  # TR 38.901 7.8

    # Viewshed placement parameters
    mast_height_m: float = 3.0
    cand_min_height_m: float = 12.0
    cand_max_height_m: float = 120.0
    candidate_spacing_m: float = 40.0
    max_candidates: int = 1200
    viewshed_radius_m: float = 800.0
    min_gain_frac: float = 0.005   # 0 (and large max_sites) -> strict "until no further improvement"
    coverage_target: float = 0.97
    max_sites: int = 45
    optimize_azimuth: bool = True
    placement_seed: int = 0
    tilt_ground_m: Optional[float] = 250.0  # per-site downtilt aims here; None -> radio.downtilt_deg
    cache_dir: Optional[str] = None         # default ~/.cache/oran_asm_sim or $ORAN_ASM_CACHE
    fill_margin_m: float = 100.0   # 45 sites / 135 sectors over the default DTLA bbox


# --------------------------------------------------------------------------- #
# Radio / channel
# --------------------------------------------------------------------------- #
@dataclass
class RadioConfig:
    scenario: str = "UMa"          # "UMa" or "UMi" (TR 38.901 Table 7.4.1-1)
    fc_ghz: float = 3.5            # n78, [USMAN25] Table V
    bw_mhz: float = 100.0          # [USMAN25] Table V
    scs_khz: float = 30.0
    n_prb: int = 273               # TS 38.101-1 Table 5.3.2-1 (100 MHz @ 30 kHz)
    rbg_size: int = 16             # TS 38.214 Table 5.1.2.2.1-1, config 1, BWP > 144 PRBs
    overhead: float = 0.14         # TS 38.306 4.1.2, FR1 DL overhead
    tdd_dl_fraction: float = (3 + 10 / 14) / 5  # DDDSU, S = 10D:2G:2U
    slot_s: float = 0.5e-3         # 30 kHz SCS

    h_bs_m: float = 25.0           # UMa (UMi: 10)
    h_ut_m: float = 1.5
    tx_power_per_chain_w: float = 40.0   # [USMAN25] Table V P_T, interpreted per TX chain
    n_tx_chains: int = 4                 # 4T4R per sector (power model)
    # Link abstraction: number of CSI-RS ports x UE Rx antennas. 2x2 by default.
    n_tx_ports: int = 2
    n_rx_ue: int = 2
    max_layers: int = 2

    # Sector antenna (TR 38.901 Table 7.3-1 pattern, array gain folded into g_max)
    g_max_dbi: float = 17.0
    hpbw_h_deg: float = 65.0
    hpbw_v_deg: float = 10.0
    a_max_db: float = 30.0
    sla_v_db: float = 30.0
    downtilt_deg: float = 12.0
    ue_gain_dbi: float = 0.0
    ue_nf_db: float = 9.0          # TR 38.802 / ITU-R M.2412
    min_dist_2d_m: float = 35.0    # TR 38.901 Table 7.8-1 (UMa)

    # Stochastic path loss
    los_model: str = "geometry"    # "geometry" (cached viewsheds of imported buildings) or "stochastic" (38.901 P_LOS)
    shadowing: bool = True
    los_corr_dist_m: float = 50.0  # TR 38.901 Table 7.6.3.1-2
    sf_corr_dist_los_m: float = 37.0   # UMa (UMi: 10)
    sf_corr_dist_nlos_m: float = 50.0  # UMa (UMi: 13)
    rician_k_los_db: float = 9.0   # TR 38.901 Table 7.5-6 (UMa LOS)

    # Indoor / in-car penetration (TR 38.901 7.4.3)
    paused_indoor_prob: float = 0.8   # paused UEs (at communities) are indoors w.p.
    high_loss_building_prob: float = 0.5
    car_penetration_mu_db: float = 9.0
    car_penetration_sigma_db: float = 5.0

    # Link adaptation
    cqi_table: str = "64QAM"       # TS 38.214 Table 5.2.2.1-2
    bler_target: float = 0.1
    interference_fading: bool = False   # Rayleigh-fade interferer powers per RBG

    # Mobility management
    ho_hysteresis_db: float = 3.0
    rsrp_min_dbm: float = -130.0   # RSRP per RE; below this a UE is in outage
    ho_pingpong_window_s: float = 1.0   # HO back to the previous cell within this time = ping-pong

    @property
    def n_rbg(self) -> int:
        return -(-self.n_prb // self.rbg_size)


# --------------------------------------------------------------------------- #
# Heterogeneous network: macro umbrella + mmWave small-cell tiers
# --------------------------------------------------------------------------- #
@dataclass
class TierConfig:
    """One radio tier (carrier / O-RU class). Fields override :class:`RadioConfig`
    for the sectors of this tier; everything not listed here (UE noise figure,
    MIMO ports, CQI table, mobility management, ...) is shared from ``radio``.

    ``tx_power_dbm`` is the total conducted power of one sector (cell), split
    evenly over ``n_tx_chains``."""
    name: str = "mmwave"
    numerology: int = 3            # mu: SCS = 15 * 2^mu kHz, slot = 1 ms / 2^mu (TS 38.211 4.2)
    fc_ghz: float = 35.0
    bw_mhz: float = 100.0
    n_prb: int = 66                # TS 38.101-2 Table 5.3.2-1 (100 MHz @ 120 kHz)
    rbg_size: int = 8              # TS 38.214 Table 5.1.2.2.1-1, config 2, BWP 37-72 PRBs
    overhead: float = 0.18         # TS 38.306 4.1.2, FR2 DL overhead
    tx_power_dbm: float = 35.0     # per sector; 15 dBm/MHz (ITU-R M.2412 dense urban mmWave PSD)
    n_tx_chains: int = 2           # hybrid beamforming: 2 digital chains (dual-pol panel)
    scenario: str = "UMi"          # TR 38.901 UMi street canyon (valid to 100 GHz)
    h_bs_m: float = 10.0           # used where no rooftop height applies (hex layouts)
    min_dist_2d_m: float = 10.0    # TR 38.901 Table 7.8-1 (UMi)
    # 8x8 panel: 8 dBi element (TR 38.901 Table 7.3-1) + 18 dB array gain, folded into g_max
    g_max_dbi: float = 26.0
    hpbw_h_deg: float = 65.0
    hpbw_v_deg: float = 10.0
    downtilt_deg: Optional[float] = None   # None -> per-site tilt from geometry.tilt_ground_m
    ue_gain_dbi: float = 0.0
    rsrp_offset_db: float = 0.0    # cell range expansion bias added to this tier's RSRP in cell selection
    controllable: bool = True      # False -> always on (command 0), not an RL agent
    power: Dict[str, Any] = field(default_factory=dict)   # PowerConfig overrides for this tier

    @property
    def scs_khz(self) -> float:
        return 15.0 * 2 ** self.numerology

    @property
    def slot_s(self) -> float:
        return 1e-3 / 2 ** self.numerology

    def to_radio(self, base: "RadioConfig") -> "RadioConfig":
        umi = self.scenario.upper() == "UMI"
        return replace(base, scenario="UMi" if umi else "UMa", fc_ghz=self.fc_ghz, bw_mhz=self.bw_mhz,
                       scs_khz=self.scs_khz, n_prb=self.n_prb, rbg_size=self.rbg_size,
                       overhead=self.overhead, slot_s=self.slot_s, h_bs_m=self.h_bs_m,
                       n_tx_chains=self.n_tx_chains,
                       tx_power_per_chain_w=10 ** (self.tx_power_dbm / 10) * 1e-3 / self.n_tx_chains,
                       g_max_dbi=self.g_max_dbi, hpbw_h_deg=self.hpbw_h_deg, hpbw_v_deg=self.hpbw_v_deg,
                       downtilt_deg=base.downtilt_deg if self.downtilt_deg is None else self.downtilt_deg,
                       ue_gain_dbi=self.ue_gain_dbi, min_dist_2d_m=self.min_dist_2d_m,
                       sf_corr_dist_los_m=10.0 if umi else 37.0,     # TR 38.901 Table 7.6.3.1-2
                       sf_corr_dist_nlos_m=13.0 if umi else 50.0)


def _macro_tier() -> TierConfig:
    return TierConfig(name="macro", numerology=1, fc_ghz=6.0, bw_mhz=40.0,
                      n_prb=106,            # TS 38.101-1 Table 5.3.2-1 (40 MHz @ 30 kHz)
                      rbg_size=16,          # config 2, BWP 73-144 PRBs
                      overhead=0.14,        # FR1
                      tx_power_dbm=49.0,    # per sector; 33 dBm/MHz over 40 MHz
                      n_tx_chains=4, scenario="UMa", h_bs_m=25.0, min_dist_2d_m=35.0,
                      g_max_dbi=17.0, hpbw_h_deg=65.0, hpbw_v_deg=10.0,
                      downtilt_deg=6.0,     # shallow tilt for umbrella coverage of the whole area
                      controllable=False, power={"sleep_component_load": "idle"})


def _mmwave_tier() -> TierConfig:
    # 10 dB cell range expansion offloads UEs from the macros (typical HetNet CRE: 6-12 dB)
    return TierConfig(rsrp_offset_db=10.0, power={"sleep_component_load": "idle"})


@dataclass
class HetNetConfig:
    """Two-tier topology: always-on tri-sector macro O-RUs (umbrella coverage) and
    mmWave O-RUs on the ``geometry.layout`` sites. There is one macro per
    ``mmwave_per_macro`` mmWave sites (rounded, at least 1), spread evenly over the
    study area (centroidal Voronoi layout); a single macro sits at the area centre.
    ``enabled=False`` gives the homogeneous single-tier network described by
    ``radio`` alone."""
    enabled: bool = True
    macro: TierConfig = field(default_factory=_macro_tier)
    small: TierConfig = field(default_factory=_mmwave_tier)
    mmwave_per_macro: float = 6.0      # macro count = max(1, round(n_layout_sites / mmwave_per_macro))
    n_macro: Optional[int] = None      # explicit macro count (overrides mmwave_per_macro)
    # explicit macro positions [(lat, lon), ...] (overrides both counts)
    macro_latlons: Optional[Tuple[Tuple[float, float], ...]] = None
    macro_height_m: float = 25.0       # UMa h_BS (TR 38.901 Table 7.8-1); on a taller roof: roof + mast
    macro_exclusion_m: float = 50.0    # small-cell sites closer than this to a macro are dropped
    macro_layout_seed: int = 0


@dataclass
class Tier:
    """A resolved tier: the effective radio / power configs of its sectors."""
    name: str
    radio: "RadioConfig"
    power: "PowerConfig"
    controllable: bool
    rsrp_offset_db: float
    downtilt_deg: Optional[float]   # fixed tilt, or None -> geometry.tilt_ground_m rule


# --------------------------------------------------------------------------- #
# Mobility: Time-variant Community model [HSU07]
# --------------------------------------------------------------------------- #
@dataclass
class PeriodConfig:
    """One TVC time-period type (NMP or CMP in [HSU07] Sec. III-B)."""
    name: str
    p_l: float                    # P(local next | roaming)   ([HSU07] Fig. 3)
    p_r: float                    # P(roaming next | local)
    community_size_m: float       # C_n / C_c
    t_max_pause_local_s: float
    t_max_pause_roam_s: float
    mean_epoch_local_m: float     # L_l
    mean_epoch_roam_m: float      # L_r
    # "random": each UE gets its own random community (original model);
    # "poi": community chosen among named POIs with the given weights (Sec. V ext.)
    community_mode: str = "random"
    poi_tags: Tuple[str, ...] = ()
    traffic_multiplier: float = 1.0


def _default_periods() -> Dict[str, PeriodConfig]:
    # Probabilities taken from [HSU07] Table II "Model 1" (matched to the MIT trace):
    #   NMP: p_l=0.5 p_r=0.2 ; CMP: p_l=0.8 p_r=0.2.
    # Lengths re-scaled to metres for a ~3.5 km x 2.9 km downtown area.
    return {
        "home": PeriodConfig("home", p_l=0.5, p_r=0.2, community_size_m=150.0,
                             t_max_pause_local_s=1800.0, t_max_pause_roam_s=600.0,
                             mean_epoch_local_m=60.0, mean_epoch_roam_m=600.0,
                             community_mode="random"),
        "work": PeriodConfig("work", p_l=0.8, p_r=0.2, community_size_m=120.0,
                             t_max_pause_local_s=1800.0, t_max_pause_roam_s=300.0,
                             mean_epoch_local_m=40.0, mean_epoch_roam_m=600.0,
                             community_mode="poi", poi_tags=("work",)),
        "lunch": PeriodConfig("lunch", p_l=0.8, p_r=0.2, community_size_m=120.0,
                              t_max_pause_local_s=900.0, t_max_pause_roam_s=300.0,
                              mean_epoch_local_m=40.0, mean_epoch_roam_m=500.0,
                              community_mode="poi", poi_tags=("food",)),
        "evening": PeriodConfig("evening", p_l=0.8, p_r=0.2, community_size_m=150.0,
                                t_max_pause_local_s=1800.0, t_max_pause_roam_s=300.0,
                                mean_epoch_local_m=50.0, mean_epoch_roam_m=600.0,
                                community_mode="poi", poi_tags=("night", "food")),
    }


def _default_schedule() -> List[Tuple[float, float, str]]:
    # (start_hour, end_hour, period). NMP = "home", CMPs = work/lunch/evening.
    return [(0.0, 9.0, "home"), (9.0, 12.0, "work"), (12.0, 13.0, "lunch"),
            (13.0, 17.0, "work"), (17.0, 18.0, "home"), (18.0, 22.0, "evening"),
            (22.0, 24.0, "home")]


@dataclass
class MobilityConfig:
    periods: Dict[str, PeriodConfig] = field(default_factory=_default_periods)
    schedule: List[Tuple[float, float, str]] = field(default_factory=_default_schedule)
    pedestrian_speed_mps: Tuple[float, float] = (0.5, 1.5)
    vehicle_speed_mps: Tuple[float, float] = (5.0, 15.0)   # [HSU07] Table II v in [5,15]
    vehicle_fraction: float = 0.2


# --------------------------------------------------------------------------- #
# Traffic
# --------------------------------------------------------------------------- #
@dataclass
class TrafficConfig:
    model: str = "ftp3"            # 3GPP FTP model 3 (TR 36.814 / TR 38.901 eval.)
    file_size_bits: float = 0.5e6 * 8   # 0.5 MB
    file_size_dist: str = "fixed"  # "fixed" or "lognormal"
    file_size_lognorm_sigma: float = 1.0
    peak_rate_files_per_s: float = 1.2   # per UE at diurnal factor 1
    activity_sigma: float = 0.8    # per-UE lognormal activity heterogeneity
    # 24 hourly multipliers (business-district shape: daytime + evening peak)
    diurnal: Tuple[float, ...] = (0.25, 0.18, 0.14, 0.12, 0.12, 0.15, 0.25, 0.45,
                                  0.70, 0.85, 0.90, 0.95, 1.00, 0.95, 0.90, 0.88,
                                  0.85, 0.85, 0.90, 0.95, 0.90, 0.75, 0.55, 0.38)
    max_buffer_bits: float = 64e6  # per-UE buffer cap; excess is dropped
    delay_budget_s: float = 0.5    # QoS: estimated queueing delay budget


# --------------------------------------------------------------------------- #
# Power model / ASMs  [USMAN25]
# --------------------------------------------------------------------------- #
@dataclass
class PowerConfig:
    # Reference parameters ([USMAN25] Table V)
    ref_bw_mhz: float = 100.0
    ref_n_ant: int = 4
    ref_streams: int = 4
    ref_q_bits: int = 12
    ref_mod_bits: int = 6
    ref_node_nm: float = 5.0
    node_nm: float = 5.0           # technology node of the simulated O-RU
    q_bits: int = 12

    # PA (Eq. 6): P_amp = N * (P_q + P_T / eta(P_T)),  eta(P_T) = eta_max (P_T/P_max)^gamma
    pa_quiescent_w: float = 10.0
    pa_eta_max: float = 40.0 / (291.0 / 4 - 10.0)  # calibrated -> 291 W PA at full load
    pa_gamma: float = 0.5          # class-B-like efficiency back-off

    # Share of each load-dependent component's power that does NOT scale with load.
    static_share: float = 0.5
    # PRB-equivalent load that is always on in an active cell (SSB/PDCCH/CSI-RS).
    signalling_load: float = 0.07
    # SSB periodicity (TS 38.213 4.1; default 20 ms) breaks idle periods of an active cell
    ssb_period_s: float = 0.020
    # Transition power = mix of active-idle and the target sleep level power
    transition_mix: float = 0.5
    # Constant site-level power (e.g. cooling / O-DU share), added per site
    site_overhead_w: float = 0.0
    # Operating point of the components that stay on in SMk:
    #  "reference": at the Table V reference load (the [USMAN25] Table VII convention);
    #  "idle": at the active-idle point (signalling load), so P_SMk <= P_idle always holds.
    #  "reference" is only consistent for a PA-dominated O-RU; the HetNet tiers use "idle".
    sleep_component_load: str = "reference"


# --------------------------------------------------------------------------- #
# Environment / RL
# --------------------------------------------------------------------------- #
@dataclass
class EnvConfig:
    decision_interval_s: float = 1.0   # one RL step (Near-RT RIC loop is 10 ms - 1 s)
    n_mini: int = 10                   # scheduling mini-periods sampled per step
    pf_virtual_slots: int = 8          # PF time-sharing resolution inside a mini-period
    ls_updates_per_step: int = 1       # large-scale channel/LOS/shadowing refreshes per step
    episode_steps: int = 3600
    start_hour: float = 8.0            # time-of-day at reset ("random" -> uniform)
    random_start: bool = False
    warmup_s: float = 30.0             # always-on warm-up (fills buffers / PF averages)
    n_ues: Optional[int] = None        # None -> ues_per_sector * n_sectors
    ues_per_sector: float = 10.0       # TR 38.901 / ITU-R M.2412 calibration density
    obs_mode: str = "flat"             # "flat" Box or "dict"
    # Reward = -w_energy * P/P_ref - w_qos * delay_violation - w_outage * outage - w_drop * drop
    w_energy: float = 1.0
    w_qos: float = 2.0
    w_outage: float = 4.0
    w_drop: float = 2.0
    action_set: Tuple[int, ...] = (0, 1, 2, 3, 4)   # 0=no ASM, 1..3=opportunistic SMk, 4=SM4 off
    seed: Optional[int] = None


@dataclass
class SimConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    radio: RadioConfig = field(default_factory=RadioConfig)
    mobility: MobilityConfig = field(default_factory=MobilityConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    hetnet: HetNetConfig = field(default_factory=HetNetConfig)

    # ------------------------------------------------------------------ #
    def tiers(self) -> List[Tier]:
        """Resolved tiers. HetNet: [macro (0), small cells (1)]; otherwise [cell (0)]."""
        if not self.hetnet.enabled:
            return [Tier("cell", self.radio, self.power, True, 0.0, None)]
        out = []
        for tc in (self.hetnet.macro, self.hetnet.small):
            out.append(Tier(tc.name, tc.to_radio(self.radio), replace(self.power, **tc.power),
                            tc.controllable, tc.rsrp_offset_db, tc.downtilt_deg))
        return out

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "SimConfig":
        """Build from a (possibly nested / partial) dict, e.g. RLlib env_config.

        Accepts ``{"radio": {"scenario": "UMi"}, "env": {"n_ues": 300}}`` or
        dotted keys ``{"radio.scenario": "UMi"}``.
        """
        cfg = cls()
        if not d:
            return cfg
        d = dict(d)
        nested: Dict[str, Dict[str, Any]] = {}
        for k, v in list(d.items()):
            if "." in k:
                sec, key = k.split(".", 1)
                nested.setdefault(sec, {})[key] = v
            elif isinstance(v, dict) and hasattr(cfg, k):
                nested.setdefault(k, {}).update(v)
            elif isinstance(v, SimConfig):
                return v
        for sec, kv in nested.items():
            for key in [k for k in kv if "." in k]:          # "hetnet.small.fc_ghz"
                head, rest = key.split(".", 1)
                kv.setdefault(head, {})[rest] = kv.pop(key)
            sub = getattr(cfg, sec)
            for key, val in kv.items():
                if not hasattr(sub, key):
                    raise KeyError(f"unknown config key {sec}.{key}")
                if sec == "hetnet" and isinstance(getattr(sub, key), TierConfig):
                    if isinstance(val, dict):
                        val = replace(getattr(sub, key), **val)
                if sec == "mobility" and key == "periods" and isinstance(val, dict):
                    val = {n: (p if isinstance(p, PeriodConfig) else PeriodConfig(name=n, **p))
                           for n, p in val.items()}
                if isinstance(getattr(sub, key), tuple) and isinstance(val, list):
                    val = tuple(val)
                setattr(sub, key, val)
        cfg.apply_scenario_defaults(explicit=nested.get("radio", {}), geo=nested.get("geometry", {}))
        return cfg

    def apply_scenario_defaults(self, explicit: Dict[str, Any] = None, geo: Dict[str, Any] = None):
        """UMi defaults (TR 38.901 Table 7.8-1) unless explicitly overridden."""
        explicit = explicit or {}
        geo = geo or {}
        if self.radio.scenario.upper() == "UMI":
            self.radio.scenario = "UMi"
            defaults = dict(h_bs_m=10.0, min_dist_2d_m=10.0, sf_corr_dist_los_m=10.0,
                            sf_corr_dist_nlos_m=13.0, rician_k_los_db=9.0)
            for k, v in defaults.items():
                if k not in explicit:
                    setattr(self.radio, k, v)
            if "isd_m" not in geo:
                self.geometry.isd_m = 200.0
        else:
            self.radio.scenario = "UMa"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Named presets (merge with your own overrides: {**PRESETS["dtla_19site"], ...})
# --------------------------------------------------------------------------- #
PRESETS: Dict[str, Dict[str, Any]] = {
    # viewshed rooftop placement over the DTLA bbox (default)
    "dtla_full": {},
    # classic TR 38.901 19-site / 57-cell hex cluster centred on DTLA (~2.5x faster)
    "dtla_19site": {"geometry": {"layout": "rings", "n_rings": 2}},
    # hexagonal lattice tiling the bbox (geometric LOS still from the buildings)
    "dtla_hex": {"geometry": {"layout": "fill"}},
    # 7 sites / 21 sectors, short episodes: for debugging RL pipelines
    "fast": {"geometry": {"layout": "rings", "n_rings": 1},
             "env": {"n_mini": 5, "episode_steps": 300, "warmup_s": 5}},
}


def preset(name: str, **overrides) -> SimConfig:
    d: Dict[str, Any] = {}
    for sec, kv in PRESETS[name].items():
        d.setdefault(sec, {}).update(kv)
    for sec, kv in overrides.items():
        d.setdefault(sec, {}).update(kv)
    return SimConfig.from_dict(d)
