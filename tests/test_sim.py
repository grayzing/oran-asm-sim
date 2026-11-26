import time

import numpy as np
import pytest

from oran_asm_sim import ORanASMEnv, ORanASMMultiAgentEnv, SimConfig
from oran_asm_sim.channel import (link_rates, los_probability_38901, mimo_eigs, pathloss_38901,
                                  sample_mimo_channel, sinr_to_eff)
from oran_asm_sim.config import PowerConfig, RadioConfig
from oran_asm_sim.geometry import (DTLA_POIS, BuildingRaster, LocalProjection, build_deployment,
                                   hex_ring_sites, make_synthetic_buildings)
from oran_asm_sim.power import ASM_TIMING, ORUPowerModel, intrinsic_efficiency_gops_w
from oran_asm_sim.scheduler import pf_schedule
from oran_asm_sim.simulator import ORANSystemSimulator

SMALL = {"geometry": {"layout": "rings", "n_rings": 1}, "env": {"warmup_s": 2, "n_mini": 5, "seed": 0}}


# ---------------------------------------------------------------- power model
def test_table_vii_reproduced():
    t = ORUPowerModel(PowerConfig(), RadioConfig()).table_vii()
    assert t["Full load"]["Total"] == pytest.approx(397, abs=0.5)
    assert t["Full load"]["RF"] == pytest.approx(365, abs=0.5)
    for k, (rf_lo, rf_hi, du_lo, du_hi, tot) in {
            "SM1": (60, 61, 25, 27, 88), "SM2": (23, 25, 13, 15, 40),
            "SM3": (16, 18, 8, 10, 28), "SM4": (8, 10, 3, 5, 15)}.items():
        assert rf_lo <= t[k]["RF"] <= rf_hi
        assert du_lo <= t[k]["Digital"] <= du_hi
        assert t[k]["Total"] == pytest.approx(tot, abs=0.5)


def test_power_monotone():
    m = ORUPowerModel(PowerConfig(), RadioConfig())
    p = m.active(np.linspace(0, 1, 11)).total
    assert np.all(np.diff(p) > 0)
    assert m.p_idle > m.sleep_power[1] > m.sleep_power[2] > m.sleep_power[3] > m.sleep_power[4]
    assert m.sleep_power[4] / m.p_full < 0.05          # ~97 % saving (paper conclusion)


def test_asm_table_i():
    assert ASM_TIMING[1] == (35.5e-6, 71e-6, 35.5e-6)
    for k in (1, 2, 3, 4):
        d, mn, a = ASM_TIMING[k]
        assert mn == pytest.approx(d + a)


def test_node_scaling():
    assert intrinsic_efficiency_gops_w(65) == pytest.approx(75)
    assert intrinsic_efficiency_gops_w(40) == pytest.approx(150)
    assert intrinsic_efficiency_gops_w(3) > intrinsic_efficiency_gops_w(5)


# ---------------------------------------------------------------- channel
def test_38901_uma_pathloss_values():
    # UMa LOS @ d2D=100 m, fc=3.5 GHz: 28 + 22log10(d3D) + 20log10(3.5)
    pl_l, pl_n, s_l, s_n = pathloss_38901("UMa", np.array([100.0]), 25, 1.5, 3.5)
    d3 = np.hypot(100, 23.5)
    assert pl_l[0] == pytest.approx(28 + 22 * np.log10(d3) + 20 * np.log10(3.5))
    assert pl_n[0] >= pl_l[0] and (s_l, s_n) == (4.0, 6.0)
    p = los_probability_38901("UMa", np.array([10, 18, 100, 1000]))
    assert p[0] == 1 and p[1] == 1 and 0 < p[3] < p[2] < 1


def test_cqi_mapping_and_mimo():
    e, m = sinr_to_eff(np.array([-20, 0, 30]))
    assert e[0] == 0 and e[2] == pytest.approx(5.5547) and m[2] == 6
    rng = np.random.default_rng(0)
    H = sample_mimo_channel(rng, 2000, 4, 2, 2, np.zeros(2000))
    ev = mimo_eigs(H)
    ref = np.linalg.eigvalsh(H @ np.conj(np.swapaxes(H, -1, -2)))[..., ::-1]
    assert np.allclose(ev, ref, rtol=1e-4, atol=1e-4)
    assert ev.sum(-1).mean() == pytest.approx(4.0, rel=0.05)      # E[tr HH^H] = Nr*Nt
    r = RadioConfig()
    bits, rank, _ = link_rates(np.full((2000, 4), 1000.0), ev, r, np.full(4, 16.0))
    assert (rank == 2).mean() > 0.8                                 # high SNR -> 2 layers


# ---------------------------------------------------------------- geometry
def test_projection_and_layout():
    pr = LocalProjection(34.05, -118.25)
    xy = pr.to_xy(34.06, -118.24)
    assert xy[1] == pytest.approx(1112, rel=0.01) and xy[0] == pytest.approx(922, rel=0.01)
    assert np.allclose(pr.to_latlon(xy), [34.06, -118.24])
    assert len(hex_ring_sites(2, 500)) == 19
    cfg = SimConfig.from_dict({"geometry": {"layout": "fill"}, "radio": {"los_model": "stochastic"}})
    dep = build_deployment(cfg.geometry, cfg.radio)
    assert dep.n_sectors == 3 * dep.n_sites
    assert len(dep.poi_names) == len(DTLA_POIS)
    assert np.all((dep.site_xy >= dep.area_min) & (dep.site_xy <= dep.area_max))


# ---------------------------------------------------------------- scheduler
def test_pf_conservation_and_fairness():
    rng = np.random.default_rng(0)
    U, B, S = 8, 18, 2
    serv = np.array([0] * 4 + [1] * 4)
    rates = np.full((U, B), 1e6)
    backlog = np.full(U, 1e9)
    served, busy, share = pf_schedule(serv, rates, backlog, np.ones(U), np.ones(S), S, rng)
    assert np.allclose(served, served[0])                    # equal channels -> equal PF shares
    assert np.all(served <= backlog)
    assert np.allclose(busy, 1.0)
    # one-shot PF with equal metrics gives all RBGs to one UE per sector; over time PF equalises.
    small = np.array([1e5, 0, 0, 0, 1e5, 1e5, 0, 0])
    served, busy, _ = pf_schedule(serv, rates, small, np.ones(U), np.ones(S), S, rng)
    assert np.allclose(served, small)
    assert np.all(busy < 0.02)


# ---------------------------------------------------------------- engine / env
def test_sm4_all_off_gives_outage_and_low_power():
    sim = ORANSystemSimulator(SimConfig.from_dict(SMALL), seed=1)
    for _ in range(3):
        m = sim.step(np.full(sim.S, 4))
    assert m["outage_frac"] == pytest.approx(1.0)
    assert m["power_total_w"] == pytest.approx(sim.S * sim.power.sleep_power[4], rel=1e-6)


def test_asm_saves_energy():
    res = {}
    for lvl in (0, 2):
        sim = ORANSystemSimulator(SimConfig.from_dict(SMALL), seed=5)
        res[lvl] = np.mean([sim.step(np.full(sim.S, lvl))["power_total_w"] for _ in range(5)])
    assert res[2] < 0.8 * res[0]


def test_sm4_wakeup_takes_activation_time():
    cfg = SimConfig.from_dict(SMALL)
    sim = ORANSystemSimulator(cfg, seed=1)
    sim.step(np.full(sim.S, 4))
    sim.step(np.full(sim.S, 4))
    assert np.all(sim.state == 2)
    sim.step(np.zeros(sim.S, int))            # 0.5 s activation inside a 1 s step
    assert np.all(sim.state == 0)


def test_mobility_bounds_and_communities():
    sim = ORANSystemSimulator(SimConfig.from_dict(SMALL), seed=2)
    for _ in range(30):
        sim.step(np.zeros(sim.S, int))
        xy = sim.mob.xy
        assert np.all(xy >= sim.dep.area_min - 1e-6) and np.all(xy <= sim.dep.area_max + 1e-6)


def test_env_api_single_and_multi():
    env = ORanASMEnv(dict(SMALL, env=dict(SMALL["env"], episode_steps=3)))
    o, info = env.reset(seed=1)
    assert env.observation_space.contains(o)
    done, n = False, 0
    while not done:
        o, r, te, tr, info = env.step(env.action_space.sample())
        assert np.isfinite(r) and env.observation_space.contains(o)
        done = te or tr
        n += 1
    assert n == 3
    for mode in ("pooled", "set"):
        ma = ORanASMMultiAgentEnv(dict(SMALL, ma_obs_mode=mode))
        obs, _ = ma.reset(seed=0)
        dep = ma.core.sim.dep
        assert set(obs) == set(ma.agents) and len(ma.agents) == dep.n_sites      # one agent per O-RU
        assert ma.sectors_of.shape == (dep.n_sites, 3)
        assert np.array_equal(np.sort(ma.sectors_of.ravel()), np.arange(ma.S))
        assert all((dep.sector_site[ma.sectors_of[i]] == i).all() for i in range(ma.n_oru))
        assert all(ma.action_spaces[a].shape == (3,) for a in ma.agents)
        obs, rew, te, tr, _ = ma.step({a: ma.action_spaces[a].sample() for a in ma.agents})
        assert "__all__" in te and all(np.isfinite(v) for v in rew.values())
        assert all(ma.observation_spaces[a].contains(obs[a]) for a in ma.agents)


def test_oru_controls_its_three_sectors():
    ma = ORanASMMultiAgentEnv(dict(SMALL))
    ma.reset(seed=3)
    ma.step({ma.agents[1]: [0, 4, 2]})                    # missing agents act 0
    s0, s1, s2 = ma.sectors_of[1]
    cmd = ma.core._last_cmd
    assert cmd[s0] == 0 and cmd[s1] == 4 and cmd[s2] == 2
    assert (np.delete(cmd, ma.sectors_of[1]) == 0).all()
    assert np.array_equal(ma.sector_actions(ma.sector_actions({ma.agents[1]: [0, 4, 2]})[ma.sectors_of]),
                          ma.sector_actions({ma.agents[1]: [0, 4, 2]}))


def test_oru_pooling_is_permutation_invariant_and_sleeping_sector_sees_ues():
    ma = ORanASMMultiAgentEnv(dict(SMALL))
    ma.reset(seed=3)
    acts = {a: ([4, 0, 0] if i == 0 else [0, 0, 0]) for i, a in enumerate(ma.agents)}
    for _ in range(2):
        obs, *_ = ma.step(acts)
    sec, ue, k, _, _ = ma.sector_ue_kpis()
    D = ma.sector_obs_dim
    # reference: pool each sector's KPI set after a random permutation
    rng = np.random.default_rng(0)
    for i, a in enumerate(ma.agents):
        for j, s in enumerate(ma.sectors_of[i]):
            ks = k[sec == s]
            if len(ks) == 0:
                continue
            ks = ks[rng.permutation(len(ks))]
            ref = np.concatenate([[c.mean(), c.std(), c.min(), c.max()] for c in ks.T])
            assert np.allclose(obs[a][j * D:j * D + 12], ref, atol=1e-5)
    s_sleep = ma.sectors_of[0, 0]
    assert ma.core.sim.state[s_sleep] == 2 and (sec == s_sleep).sum() > 0     # SM4 sector still sees its UEs


def test_reproducible():
    a = ORANSystemSimulator(SimConfig.from_dict(SMALL), seed=11).step(np.full(21, 2))
    b = ORANSystemSimulator(SimConfig.from_dict(SMALL), seed=11).step(np.full(21, 2))
    assert a["power_total_w"] == b["power_total_w"] and a["served_bps"] == b["served_bps"]


# ---------------------------------------------------------------- imported geometry / placement
def _city(tmp_path):
    import json
    bbox = (34.046, -118.256, 34.053, -118.246)
    p = tmp_path / "b.geojson"
    make_synthetic_buildings(bbox, str(p), seed=1)
    return bbox, str(p)


def test_overpass_json_and_heights(tmp_path):
    import json
    osm = {"elements": [{"type": "way", "id": 1, "tags": {"building": "yes", "height": "52 m"},
                         "geometry": [{"lat": 34.0485, "lon": -118.2515}, {"lat": 34.0485, "lon": -118.2505},
                                      {"lat": 34.0495, "lon": -118.2505}, {"lat": 34.0495, "lon": -118.2515},
                                      {"lat": 34.0485, "lon": -118.2515}]},
                        {"type": "way", "id": 2, "tags": {"building": "yes", "building:levels": "10"},
                         "geometry": [{"lat": 34.0475, "lon": -118.2515}, {"lat": 34.0475, "lon": -118.2505},
                                      {"lat": 34.0480, "lon": -118.2505}, {"lat": 34.0475, "lon": -118.2515}]}]}
    p = tmp_path / "osm.json"
    p.write_text(json.dumps(osm))
    proj = LocalProjection(34.048, -118.251)
    r = BuildingRaster.from_file(str(p), proj, proj.to_xy(34.046, -118.254), proj.to_xy(34.050, -118.248), 5.0)
    assert r.n_buildings == 2 and r.h.max() == pytest.approx(52) and 35 in set(np.unique(r.h).tolist())
    xy_in = proj.to_xy(34.0490, -118.2510)
    assert r.is_indoor(xy_in[None])[0] and r.indoor_distance(xy_in[None])[0] > 20


def test_viewshed_matches_raytrace_and_placement_cached(tmp_path):
    from oran_asm_sim.placement import place_sites, viewshed
    bbox, path = _city(tmp_path)
    cfg = SimConfig.from_dict({"geometry": {"bbox": bbox, "buildings_geojson": path, "cache_dir": str(tmp_path),
                                            "max_sites": 8, "fill_margin_m": 20, "candidate_spacing_m": 30}})
    dep = build_deployment(cfg.geometry, cfg.radio)
    b = dep.buildings
    assert dep.buildings_source == "file" and dep.n_sites >= 3 and dep.site_vis.shape == (dep.n_sites, b.h.size)
    curve = dep.placement.coverage_curve
    assert np.all(np.diff(curve) > 0)                                   # greedy adds visibility each pick
    gains = np.diff(np.concatenate([[0], curve]))
    assert np.all(np.diff(gains) <= 1e-9)                               # submodular: diminishing gains
    # viewshed vs brute-force ray tracing on outdoor cells
    rng = np.random.default_rng(0)
    cells = np.argwhere(b.outdoor)
    xy = b.origin + (cells[rng.choice(len(cells), 800)][:, ::-1] + 0.5) * b.res
    vs = dep.site_vis[:, b.outdoor_flat_index(xy)].T
    rt = b.los_raytrace(xy, dep.site_xy, cfg.radio.h_ut_m, dep.site_h, n_samples=300)
    assert (vs == rt).mean() > 0.95
    # cache hit returns identical placement
    t = time.time()
    dep2 = build_deployment(cfg.geometry, cfg.radio)
    assert time.time() - t < 2.0 and np.allclose(dep2.site_xy, dep.site_xy)
    assert np.array_equal(dep2.site_vis, dep.site_vis)


def test_simulator_uses_geometry_los(tmp_path):
    bbox, path = _city(tmp_path)
    cfg = SimConfig.from_dict({"geometry": {"bbox": bbox, "buildings_geojson": path, "cache_dir": str(tmp_path),
                                            "max_sites": 6, "fill_margin_m": 20, "candidate_spacing_m": 30},
                               "env": {"warmup_s": 1, "n_ues": 60, "n_mini": 2}})
    sim = ORANSystemSimulator(cfg, seed=0)
    sim.step(np.zeros(sim.S, int))
    idx = sim.dep.buildings.outdoor_flat_index(sim.mob.xy)
    assert np.array_equal(sim.chan.los, sim.dep.site_vis[:, idx].T)
    assert np.allclose(sim._indoor, sim.dep.buildings.is_indoor(sim.mob.xy))


def test_umi_stochastic_fallback():
    cfg = SimConfig.from_dict({"radio": {"scenario": "UMi", "los_model": "stochastic"},
                               "geometry": {"layout": "rings", "n_rings": 1},
                               "env": {"warmup_s": 1, "n_ues": 50, "n_mini": 2}})
    assert cfg.geometry.isd_m == 200 and cfg.radio.h_bs_m == 10
    sim = ORANSystemSimulator(cfg, seed=0)
    assert sim.dep.buildings is None and sim.dep.site_vis is None
    sim.step(np.zeros(sim.S, int))


# ---------------------------------------------------------------- handover KPIs
def test_handover_breakdown_consistent_and_sm4_forces_ho():
    sim = ORANSystemSimulator(SimConfig.from_dict(SMALL), seed=3)
    S = sim.S
    tot = dict(handovers=0, ho_mobility=0, ho_forced=0, ho_intra_site=0, ho_inter_site=0, ho_pingpong=0)
    for _ in range(5):
        m = sim.step(np.zeros(S, int))
        assert m["ho_forced"] == 0            # nothing switched off
        for k in tot:
            tot[k] += m[k]
        assert m["handovers"] == m["ho_mobility"] + m["ho_forced"] == m["ho_intra_site"] + m["ho_inter_site"]
        assert m["ho_out_sector"].sum() == m["ho_in_sector"].sum() == m["handovers"]
        assert m["ho_pingpong"] <= m["handovers"]
    assert sim.ue_handovers.sum() == tot["handovers"]

    # switch off the most loaded sector: its UEs must be handed over (forced) or dropped
    s = int(np.argmax(np.bincount(sim.serving[sim.serving >= 0], minlength=S)))
    n_on_s = int(np.sum(sim.serving == s))
    assert n_on_s > 0
    a = np.zeros(S, int)
    a[s] = 4
    m = sim.step(a)
    assert m["ho_forced"] + m["rlf"] >= n_on_s
    assert m["ho_out_sector"][s] >= m["ho_forced"] > 0
    assert not np.any(sim.serving == s)
