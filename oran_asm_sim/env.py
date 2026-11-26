"""Gymnasium environments for per-sector O-RU ASM control.

* :class:`ORanASMEnv` - single-agent (centralised rApp/xApp view). Observation
  is a flat ``Box`` (or ``Dict``), action is ``MultiDiscrete([5] * n_ctrl)``.
* :class:`ORanASMMultiAgentEnv` - RLlib ``MultiAgentEnv`` with one agent per
  controllable O-RU (one per site), each controlling the sleep commands of its 3
  sectors with a ``MultiDiscrete([5] * 3)`` action.

In the default HetNet topology the always-on macro O-RU is not an agent: only the
mmWave sectors (``sim.ctrl``) appear in the action space and in the per-sector
observation; the macro's load is summarised in the global features.

Both accept an RLlib-style ``env_config`` dict, see :meth:`SimConfig.from_dict`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ._compat import Env, MultiAgentEnv, spaces
from .config import SimConfig
from .simulator import ORANSystemSimulator

N_LEVELS = 5
SECTOR_FEATURES = (
    "util", "n_ue", "backlog", "offered", "served", "power", "sinr", "sleep_frac",
    "pot_demand", "cmd0", "cmd1", "cmd2", "cmd3", "cmd4", "st_on", "st_deact", "st_sm4", "st_act",
)
GLOBAL_FEATURES = ("hour_sin", "hour_cos", "outage", "qos", "macro_util", "macro_ue_frac")
CAP_REF_BPS = 500e6


def _cfg(env_config) -> SimConfig:
    if isinstance(env_config, SimConfig):
        return env_config
    d = dict(env_config or {})
    # RLlib's EnvContext carries worker_index etc.; ignore non-config keys
    for k in ("worker_index", "vector_index", "num_workers", "remote"):
        d.pop(k, None)
    return SimConfig.from_dict(d)


class ORanASMEnv(Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_config: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = _cfg(env_config)
        seed = self.cfg.env.seed
        wi = getattr(env_config, "worker_index", 0) if env_config is not None else 0
        vi = getattr(env_config, "vector_index", 0) if env_config is not None else 0
        if seed is not None:
            seed = int(seed) + 1000 * int(wi) + int(vi)
        self._seed = seed
        self.render_mode = render_mode
        self.sim = ORANSystemSimulator(self.cfg, seed=seed)
        self.ctrl = self.sim.ctrl                    # controllable sector indices (into sim.S)
        self.S = len(self.ctrl)                      # action dimension
        self.action_levels = np.asarray(self.cfg.env.action_set, int)
        self.action_space = spaces.MultiDiscrete([len(self.action_levels)] * self.S)
        F, Gf = len(SECTOR_FEATURES), len(GLOBAL_FEATURES)
        if self.cfg.env.obs_mode == "dict":
            self.observation_space = spaces.Dict({
                "sectors": spaces.Box(-1.0, 2.0, (self.S, F), np.float32),
                "global": spaces.Box(-1.0, 1.0, (Gf,), np.float32)})
        else:
            self.observation_space = spaces.Box(-1.0, 2.0, (self.S * F + Gf,), np.float32)
        self._last_cmd = np.zeros(self.sim.S, int)   # last command per sector (all sectors; macro = 0)

    # ------------------------------------------------------------------ #
    def sector_features(self) -> np.ndarray:
        """Features of the controllable sectors, [n_ctrl, F]."""
        sim, m = self.sim, self.sim.last
        Sa, c = sim.S, self.ctrl
        f = np.zeros((self.S, len(SECTOR_FEATURES)), np.float32)
        ups = max(sim.U / Sa, 1.0)
        z = np.zeros(Sa)
        f[:, 0] = m.get("util_sector", z)[c]
        f[:, 1] = np.clip(m.get("n_ue_sector", z)[c] / (2 * ups), 0, 2)
        f[:, 2] = np.log10(1 + sim.sector_backlog()[c]) / 9.0
        f[:, 3] = np.clip(sim.sector_offered_bps()[c] / CAP_REF_BPS, 0, 2)
        f[:, 4] = np.clip(m.get("served_sector_bps", z)[c] / CAP_REF_BPS, 0, 2)
        f[:, 5] = m.get("power_sector_w", z)[c] / sim.power.p_full[c]
        f[:, 6] = np.clip(sim.sector_mean_sinr_db()[c] / 30.0, -1, 2)
        f[:, 7] = m.get("sleep_frac_sector", z)[c]
        pot = np.bincount(sim.best_sector_if_all_on(), minlength=Sa)[c]
        f[:, 8] = np.clip(pot / (2 * ups), 0, 2)
        ar = np.arange(self.S)
        f[ar, 9 + np.clip(self._last_cmd[c], 0, 4)] = 1.0
        f[ar, 14 + sim.state[c]] = 1.0
        return f

    def global_features(self) -> np.ndarray:
        sim = self.sim
        h = (sim.t / 3600.0) % 24.0
        m = sim.last
        fixed = ~sim.controllable                     # always-on (macro) sectors
        util = m.get("util_sector")
        n_ue = m.get("n_ue_sector")
        macro_util = float(util[fixed].mean()) if fixed.any() and util is not None else 0.0
        macro_ue = float(n_ue[fixed].sum() / max(sim.U, 1)) if fixed.any() and n_ue is not None else 0.0
        return np.array([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                         m.get("outage_frac", 0.0), m.get("qos_violation_frac", 0.0),
                         macro_util, macro_ue], np.float32)

    def _obs(self):
        sf, gf = self.sector_features(), self.global_features()
        if self.cfg.env.obs_mode == "dict":
            return {"sectors": sf, "global": gf}
        return np.concatenate([sf.reshape(-1), gf]).astype(np.float32)

    def reward(self, m: Dict[str, Any]) -> float:
        e = self.cfg.env
        return float(-(e.w_energy * m["power_norm"] + e.w_qos * m["qos_violation_frac"]
                       + e.w_outage * m["outage_frac"] + e.w_drop * m["drop_frac"]))

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._seed = seed
        elif self._seed is not None:
            self._seed += 7919          # new but reproducible episode
        start = (options or {}).get("start_hour")
        self.sim.reset(self._seed, start)
        self._last_cmd = np.zeros(self.sim.S, int)
        return self._obs(), self._info(self.sim.last)

    def step(self, action):
        a = np.asarray(action, int).reshape(-1)
        assert a.shape[0] == self.S, f"expected {self.S} actions (controllable sectors), got {a.shape[0]}"
        cmd = np.zeros(self.sim.S, int)
        cmd[self.ctrl] = self.action_levels[a]
        self._last_cmd = cmd
        m = self.sim.step(cmd)
        r = self.reward(m)
        truncated = self.sim.step_count >= self.cfg.env.episode_steps
        return self._obs(), r, False, truncated, self._info(m)

    @staticmethod
    def _info(m):
        keys = ("time_s", "hour", "period", "power_total_w", "power_fixed_w", "power_ctrl_w", "power_norm", "energy_j", "offered_bps",
                "served_bps", "drop_frac", "blocked_frac", "outage_frac", "qos_violation_frac", "handovers",
                "ho_mobility", "ho_forced", "ho_intra_site", "ho_inter_site", "ho_pingpong",
                "ho_pingpong_frac", "rlf", "reconnects", "ho_rate_per_ue_s",
                "ue_tput_mean_bps", "ue_tput_p5_bps", "wake_latency_s", "backlog_bits")
        return {k: m[k] for k in keys if k in m}

    def render(self):
        from .viz import plot_network
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
        plot_network(self.sim, ax)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img


# --------------------------------------------------------------------------- #
ORU_KPIS = ("demand", "sinr", "throughput")
POOL_OPS = ("mean", "std", "min", "max")
STATE_EXTRAS = ("st_on", "st_deact", "st_sm4", "st_act", "cmd0", "cmd1", "cmd2", "cmd3", "cmd4")


class ORanASMMultiAgentEnv(MultiAgentEnv):
    """One agent per controllable O-RU. An O-RU sits on one site and controls the
    sleep commands of that site's three sectors. The always-on macro O-RU of the
    HetNet topology is not an agent (its sectors always get command 0).

    Agent ``oru_i`` controls the sectors ``self.sectors_of[i]`` (in azimuth order) and
    acts with ``MultiDiscrete([L] * 3)``, one command index per sector.

    The observation of an O-RU stacks one feature block per sector it controls. Each
    block holds the downlink demand, SINR and throughput of the sector's associated
    UEs, aggregated with a permutation-invariant operator, plus the sector's energy
    efficiency.

    * ``ma_obs_mode="pooled"`` (default): a flat ``Box`` of ``3 * D``. Per sector:
      mean / std / min / max of each KPI, set size, sums of demand and throughput,
      own EE (bits/J).
    * ``ma_obs_mode="set"``: the raw KPI sets, ``{"ue": [3,K,3], "mask": [3,K],
      "oru": [3, 1 + extras]}``; the top-``max_ues`` UEs by demand are kept per sector.

    The associated UEs of a sector that is not on air (SM4 or transitioning) are the
    UEs that would select it as best server if it were on, with SINR evaluated towards
    that sector.

    ``oru_state_features=True`` (default) appends each sector's cell state and last
    command (one-hot). Reward: the shared reward of :class:`ORanASMEnv`, optionally
    blended (``local_reward_mix``) with the O-RU's own normalised power.
    """

    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        wi = int(getattr(env_config, "worker_index", 0) or 0)
        vi = int(getattr(env_config, "vector_index", 0) or 0)
        d = dict(env_config or {})
        self.local_mix = float(d.pop("local_reward_mix", 0.0))
        self.ma_obs_mode = d.pop("ma_obs_mode", "pooled")
        self.max_ues = int(d.pop("max_ues", 64))
        self.state_feats = bool(d.pop("oru_state_features", True))
        self.core = ORanASMEnv(d)
        if self.core._seed is not None:          # decorrelate RLlib env runners
            self.core._seed += 1000 * wi + vi
        sim = self.core.sim
        self.S = sim.S
        site = np.asarray(sim.dep.sector_site, int)
        ctrl_sec = sim.ctrl
        self.oru_site = np.unique(site[ctrl_sec])                     # site index of each agent
        if not np.array_equal(np.sort(ctrl_sec), np.where(np.isin(site, self.oru_site))[0]):
            raise ValueError("an O-RU must be either fully controllable or fully always-on")
        n_oru = len(self.oru_site)
        counts = np.bincount(site[ctrl_sec], minlength=site.max() + 1)[self.oru_site]
        if not np.all(counts == counts[0]):
            raise ValueError(f"every O-RU must control the same number of sectors, got {counts.tolist()}")
        self.n_oru, self.sectors_per_oru = n_oru, int(counts[0])
        order = ctrl_sec[np.argsort(site[ctrl_sec], kind="stable")]
        self.sectors_of = order.reshape(n_oru, self.sectors_per_oru)                 # [n_oru, 3]
        self.agents = self.possible_agents = [f"oru_{i}" for i in range(n_oru)]
        K3 = self.sectors_per_oru
        n_extra = len(STATE_EXTRAS) if self.state_feats else 0
        L = len(self.core.action_levels)
        if self.ma_obs_mode == "set":
            sp = spaces.Dict({"ue": spaces.Box(-1.0, 2.0, (K3, self.max_ues, len(ORU_KPIS)), np.float32),
                              "mask": spaces.Box(0.0, 1.0, (K3, self.max_ues), np.float32),
                              "oru": spaces.Box(-1.0, 2.0, (K3, 1 + n_extra), np.float32)})
        else:
            self.sector_obs_dim = len(ORU_KPIS) * len(POOL_OPS) + 3 + 1 + n_extra
            sp = spaces.Box(-1.0, 2.0, (K3 * self.sector_obs_dim,), np.float32)
        self.observation_spaces = {a: sp for a in self.agents}
        self.action_spaces = {a: spaces.MultiDiscrete([L] * K3) for a in self.agents}
        self.observation_space = sp          # old-API aliases
        self.action_space = self.action_spaces[self.agents[0]]

    def get_observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    def get_action_space(self, agent_id):
        return self.action_spaces[agent_id]

    def sector_actions(self, actions) -> np.ndarray:
        """Per-sector action indices [S] (all sectors; always-on sectors get 0) from
        per-O-RU actions: a dict keyed by agent ID (missing agents act 0) or an array
        [n_oru, 3]."""
        act = np.zeros(self.S, int)
        if isinstance(actions, dict):
            for i, a in enumerate(self.agents):
                if a in actions:
                    act[self.sectors_of[i]] = np.asarray(actions[a], int).reshape(-1)
        else:
            act[self.sectors_of] = np.asarray(actions, int).reshape(self.n_oru, self.sectors_per_oru)
        return act

    # ------------------------------------------------------------------ #
    def sector_ue_kpis(self):
        """Per (UE, sector) association pairs and their normalised KPIs.
        Returns (sector [P], ue [P], kpis [P,3], demand_bps [U], served_bps [U]) with
        KPIs = demand, SINR, throughput."""
        sim, m = self.core.sim, self.core.sim.last
        T = self.core.cfg.env.decision_interval_s
        on = sim.state == 0
        u_srv = np.where(sim.serving >= 0)[0]
        s_srv = sim.serving[u_srv]
        best = sim.best_sector_if_all_on()
        u_pot = np.where(~on[best])[0]
        s_pot = best[u_pot]
        ue = np.concatenate([u_srv, u_pot])
        sec = np.concatenate([s_srv, s_pot])
        off = m.get("ue_offered_bps", np.zeros(sim.U))
        srv = m.get("ue_served_bps", np.zeros(sim.U))
        demand_bps = off + sim.backlog / T
        k = np.zeros((ue.size, 3), np.float32)
        k[:, 0] = np.log10(1 + demand_bps[ue]) / 8.0
        k[:, 1] = np.clip(sim.sinr_toward(ue, sec), -10.0, 40.0) / 40.0
        k[:, 2] = np.where(on[sec], np.log10(1 + srv[ue]) / 8.0, 0.0)
        return sec, ue, k, demand_bps, srv

    def _sector_own_features(self):
        sim, m = self.core.sim, self.core.sim.last
        S = self.S
        p = m.get("power_sector_w", np.full(S, sim.power.p_idle))
        ee = m.get("served_sector_bps", np.zeros(S)) / np.maximum(p, 1e-6)       # bits / J
        f = [np.log10(1 + ee)[:, None] / 7.0]
        if self.state_feats:
            st = np.zeros((S, 4), np.float32)
            st[np.arange(S), sim.state] = 1
            cmd = np.zeros((S, 5), np.float32)
            cmd[np.arange(S), np.clip(self.core._last_cmd, 0, 4)] = 1
            f += [st, cmd]
        return np.concatenate(f, 1).astype(np.float32)

    def sector_obs(self):
        """Per-sector observation blocks: ``[S, D]`` (pooled) or a dict of per-sector arrays (set)."""
        S = self.S
        sec, ue, k, demand_bps, srv = self.sector_ue_kpis()
        own = self._sector_own_features()
        if self.ma_obs_mode == "set":
            K = self.max_ues
            order = np.lexsort((-k[:, 0], sec))             # by sector, demand descending
            sec_o, k_o = sec[order], k[order]
            start = np.searchsorted(sec_o, np.arange(S))
            rank = np.arange(sec_o.size) - start[sec_o]
            keep = rank < K
            ue_arr = np.zeros((S, K, 3), np.float32)
            mask = np.zeros((S, K), np.float32)
            ue_arr[sec_o[keep], rank[keep]] = k_o[keep]
            mask[sec_o[keep], rank[keep]] = 1
            return {"ue": ue_arr, "mask": mask, "oru": own}
        # permutation-invariant pooling: mean / std / min / max per KPI + |set| + sums
        cnt = np.bincount(sec, minlength=S).astype(float)
        has = cnt > 0
        pooled = []
        for j in range(3):
            x = k[:, j].astype(float)
            s1 = np.bincount(sec, weights=x, minlength=S)
            s2 = np.bincount(sec, weights=x * x, minlength=S)
            mean = np.where(has, s1 / np.maximum(cnt, 1), 0.0)
            std = np.sqrt(np.maximum(np.where(has, s2 / np.maximum(cnt, 1), 0.0) - mean ** 2, 0.0))
            mn = np.full(S, np.inf)
            mx = np.full(S, -np.inf)
            np.minimum.at(mn, sec, x)
            np.maximum.at(mx, sec, x)
            pooled += [mean, std, np.where(has, mn, 0.0), np.where(has, mx, 0.0)]
        ups = max(self.core.sim.U / S, 1.0)
        on_pair = self.core.sim.state[sec] == 0
        pooled += [np.clip(cnt / (2 * ups), 0, 2),
                   np.log10(1 + np.bincount(sec, weights=demand_bps[ue], minlength=S)) / 9.0,
                   np.log10(1 + np.bincount(sec, weights=np.where(on_pair, srv[ue], 0.0), minlength=S)) / 9.0]
        obs = np.concatenate([np.stack(pooled, 1), own], 1).astype(np.float32)
        return np.clip(obs, -1.0, 2.0)

    def _obs(self):
        so, idx = self.sector_obs(), self.sectors_of
        if self.ma_obs_mode == "set":
            return {a: {k: v[idx[i]] for k, v in so.items()} for i, a in enumerate(self.agents)}
        return {a: so[idx[i]].reshape(-1) for i, a in enumerate(self.agents)}

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        _, info = self.core.reset(seed=seed, options=options)
        return self._obs(), {a: {} for a in self.agents}

    def step(self, action_dict):
        _, r, term, trunc, info = self.core.step(self.sector_actions(action_dict)[self.core.ctrl])
        m = self.core.sim.last
        p_full = self.core.sim.power.p_full
        p_own = m["power_sector_w"][self.sectors_of].sum(1) / p_full[self.sectors_of].sum(1)
        w = self.core.cfg.env.w_energy
        rew = {a: float((1 - self.local_mix) * r - self.local_mix * w * p_own[i])
               for i, a in enumerate(self.agents)}
        terms = {a: term for a in self.agents}
        truncs = {a: trunc for a in self.agents}
        terms["__all__"], truncs["__all__"] = term, trunc
        infos = {a: {} for a in self.agents}
        infos[self.agents[0]] = dict(info)
        return self._obs(), rew, terms, truncs, infos
