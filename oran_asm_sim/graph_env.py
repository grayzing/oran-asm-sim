"""Per-O-RU environment with set-structured observations shared between neighbouring O-RUs.

Local observation of O-RU i at decision step t (whiteboard model)::

    O_{i,t} = [ Phi(i) | { Lambda(x_j, s_{i,j}, gamma_{i,j}) : j in H_i } | U_ThptLim(i) | state(i) ]

* ``Phi(i)``: energy efficiency of the whole O-RU, i.e. its aggregate DL throughput over the step
  divided by the power of its sectors (bits/J, log-scaled: ``log10(1 + EE) / 7``).
* ``H_i``: the UEs associated with O-RU i: those served by one of its sectors, plus, for a sector
  that is not on air (SM4 or transitioning), the UEs that would select it as best server if it were
  on. Each UE j contributes one row:

  - ``x_j``     estimated DL buffer of UE j (the traffic load), ``log10(1 + bits) / 9``, i.e. 1.0 = 1 Gb;
  - ``s_ij``    CQI (0..15, TS 38.214 Table 5.2.2.1-2 thresholds on the wideband SINR towards the
                sector) divided by 15;
  - ``gamma_ij`` DL throughput over the step, ``log10(1 + bps) / 9`` (0 for a sector that is off);
  - a one-hot of which of the O-RU's sectors the UE belongs to, so that a permutation-invariant
    sum over H_i can still tell the per-sector loads apart (the action is per sector).

  The env returns the padded set (top ``max_ues`` UEs by buffer) and its mask; the sum over H_i
  (the DeepSet) is learned inside the Q-network (:mod:`oran_asm_sim.vdn.gnn`).
* ``U_ThptLim(i)``: per-sector utilisation (busy fraction of the step), i.e. how close each sector
  is to its throughput limit. Optional (``thpt_lim_feature``).
* ``state(i)``: per-sector cell state (one-hot of on / deact / SM4 / act) and command in force
  (one-hot). Optional (``oru_state_features``), plus |H_i| / max_ues.

Observation sharing
-------------------
With ``obs_sharing=True`` every O-RU broadcasts its local observation to its one-hop neighbours
(the ``max_neighbors`` nearest O-RU sites, optionally within ``neighbor_radius_m``). A broadcast
takes ``share_latency_s`` to arrive, so at decision time an O-RU holds its *own* fresh observation
but its neighbours' observations as they were ``share_latency_s`` earlier.

The simulator resolves a decision step into ``env.n_mini`` scheduling mini-periods
(``dt_mini = decision_interval_s / n_mini``), so latency is modelled at that resolution: it is
rounded up to ``n = ceil(latency / dt_mini)`` mini-periods, ``n = d * n_mini + m``. The shared
observation is snapshotted ``m`` mini-periods before the end of the step (its throughput / power /
utilisation are averaged over the part of the step elapsed so far) and delivered ``d`` decision
steps later. With the defaults (1 s steps, ``n_mini = 10``), a 40 ms latency becomes 100 ms of
staleness; ``n_mini = 25`` makes it exact. ``self.share_staleness_s`` holds the modelled value.

Per-agent observation vector (flat ``Box``; :class:`ObsLayout` describes it)::

    [ own (L) | own_shared (L, only if staleness > 0) | nbr_idx (M) | nbr_shared (M * L) ]

``own_shared`` (what this O-RU broadcasts) is not used for acting; the joint replay buffer keeps it
and rebuilds the neighbour blocks from ``nbr_idx`` at training time, instead of storing M copies.
``nbr_idx`` holds the agent indices of the neighbours (-1 = padding).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from ._compat import spaces
from .env import ORanASMMultiAgentEnv

# TS 38.214 Table 5.2.2.1-2 (up to 64QAM): SINR [dB] at which CQI k (1..15) reaches 10 % BLER
CQI_SINR_DB = np.array([-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7])
UE_FEATURES = ("buffer", "cqi", "throughput")


def sinr_to_cqi(sinr_db: np.ndarray) -> np.ndarray:
    """Wideband CQI index 0..15 (0 = out of range)."""
    return np.searchsorted(CQI_SINR_DB, np.asarray(sinr_db), side="right")


@dataclass(frozen=True)
class ObsLayout:
    """Layout of the flat per-agent observation of :class:`ORanASMGraphEnv`."""
    max_ues: int              # K: padded UE-set size
    ue_dim: int               # features per UE row (incl. the sector one-hot)
    oru_dim: int              # O-RU-level features
    max_neighbors: int        # M (0 = no sharing)
    has_shared_copy: bool     # own_shared present (shared observations are stale)

    @property
    def local_dim(self) -> int:
        return self.max_ues * self.ue_dim + self.max_ues + self.oru_dim

    @property
    def compact_dim(self) -> int:
        """Per-agent part kept by the replay buffer: own (+ own_shared)."""
        return self.local_dim * (2 if self.has_shared_copy else 1)

    @property
    def obs_dim(self) -> int:
        return self.compact_dim + self.max_neighbors * (1 + self.local_dim)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def of(cls, d) -> "ObsLayout":
        return d if isinstance(d, cls) else cls(**d)


class _EnvCfg(dict):
    """A plain dict that keeps RLlib's EnvContext worker / vector indices."""


def latency_split(latency_s: float, decision_interval_s: float, n_mini: int):
    """(d_steps, m_minis, staleness_s): latency rounded up to whole mini-periods,
    ``n = d_steps * n_mini + m_minis``."""
    dt = decision_interval_s / n_mini
    n = max(0, math.ceil(latency_s / dt - 1e-9))
    d, m = divmod(n, n_mini)
    return d, m, n * dt


class ORanASMGraphEnv(ORanASMMultiAgentEnv):
    """One agent per controllable O-RU, ``MultiDiscrete([L] * 3)`` sector commands (as
    :class:`ORanASMMultiAgentEnv`), with the set-structured, neighbour-shared observation above.

    Extra ``env_config`` keys (besides the simulator config and ``local_reward_mix``):

    ``max_ues`` (48), ``obs_sharing`` (True), ``max_neighbors`` (6), ``neighbor_radius_m`` (None),
    ``share_latency_s`` (0.04), ``thpt_lim_feature`` (True), ``oru_state_features`` (True).
    """

    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        d = _EnvCfg(env_config or {})
        for k in ("worker_index", "vector_index"):
            setattr(d, k, getattr(env_config, k, 0) or 0)
        self.obs_sharing = bool(d.pop("obs_sharing", True))
        max_nbr = int(d.pop("max_neighbors", 6))
        radius = d.pop("neighbor_radius_m", None)
        self.share_latency_s = float(d.pop("share_latency_s", 0.04))
        self.thpt_lim = bool(d.pop("thpt_lim_feature", True))
        d.setdefault("max_ues", 48)
        d["ma_obs_mode"] = "pooled"            # parent spaces are replaced below
        super().__init__(d)
        sim, N, K3 = self.core.sim, self.n_oru, self.sectors_per_oru

        self.oru_of_sector = np.full(self.S, -1)
        self.pos_in_oru = np.zeros(self.S, int)
        for i, secs in enumerate(self.sectors_of):
            self.oru_of_sector[secs] = i
            self.pos_in_oru[secs] = np.arange(K3)

        # one-hop neighbour graph: nearest O-RU sites
        xy = np.asarray(sim.dep.site_xy)[self.oru_site]
        dist = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        np.fill_diagonal(dist, np.inf)
        M = min(max_nbr, N - 1) if self.obs_sharing else 0
        self.nbr_idx = np.full((N, M), -1)
        for i in range(N):
            cand = np.argsort(dist[i], kind="stable")[:M]
            if radius is not None:
                cand = cand[dist[i, cand] <= float(radius)]
            self.nbr_idx[i, :len(cand)] = cand

        e = self.core.cfg.env
        self.share_delay_steps, self.share_mini_offset, self.share_staleness_s = (
            latency_split(self.share_latency_s, e.decision_interval_s, e.n_mini) if M else (0, 0, 0.0))
        oru_dim = 2 + (K3 if self.thpt_lim else 0) + (9 * K3 if self.state_feats else 0)
        self.obs_layout = ObsLayout(max_ues=self.max_ues, ue_dim=len(UE_FEATURES) + K3, oru_dim=oru_dim,
                                    max_neighbors=M, has_shared_copy=bool(M) and self.share_staleness_s > 0)
        lay = self.obs_layout
        hi = np.full(lay.obs_dim, 2.0, np.float32)
        hi[lay.compact_dim:lay.compact_dim + M] = N - 1
        sp = spaces.Box(np.full(lay.obs_dim, -1.0, np.float32), hi, dtype=np.float32)
        self.observation_spaces = {a: sp for a in self.agents}
        self.observation_space = sp

        # delay line of shared observations, fed by a simulator mini-period hook
        self._shared = deque(maxlen=self.share_delay_steps + 1)
        if M:
            self._hook_mini = e.n_mini - 1 - self.share_mini_offset
            sim.mini_hooks.append(self._snapshot_hook)

    # ------------------------------------------------------------------ #
    def local_obs(self, ue_tput_bps, sec_served_bps, sec_power_w, sec_util) -> np.ndarray:
        """Local observation of every O-RU, [N, L], from the current simulator state plus
        averages over the step so far (per-UE throughput, per-sector served rate / power / busy
        fraction)."""
        sim, lay, N, K, K3 = self.core.sim, self.obs_layout, self.n_oru, self.max_ues, self.sectors_per_oru
        on = sim.state == 0
        u_srv = np.where(sim.serving >= 0)[0]
        best = sim.best_sector_if_all_on()
        u_pot = np.where(~on[best])[0]
        ue = np.concatenate([u_srv, u_pot])
        sec = np.concatenate([sim.serving[u_srv], best[u_pot]])
        oru = self.oru_of_sector[sec]
        keep = oru >= 0                                      # drop UEs of always-on (macro) sectors
        ue, sec, oru = ue[keep], sec[keep], oru[keep]

        f = np.zeros((ue.size, lay.ue_dim), np.float32)
        f[:, 0] = np.log10(1.0 + sim.backlog[ue]) / 9.0
        f[:, 1] = sinr_to_cqi(sim.sinr_toward(ue, sec)) / 15.0
        f[:, 2] = np.where(on[sec], np.log10(1.0 + ue_tput_bps[ue]) / 9.0, 0.0)
        f[np.arange(ue.size), 3 + self.pos_in_oru[sec]] = 1.0

        order = np.lexsort((-f[:, 0], oru))                  # by O-RU, largest buffer first
        oru_o = oru[order]
        rank = np.arange(oru_o.size) - np.searchsorted(oru_o, np.arange(N))[oru_o]
        sel = rank < K
        ue_set = np.zeros((N, K, lay.ue_dim), np.float32)
        mask = np.zeros((N, K), np.float32)
        ue_set[oru_o[sel], rank[sel]] = f[order][sel]
        mask[oru_o[sel], rank[sel]] = 1.0

        so = self.sectors_of
        ee = sec_served_bps[so].sum(1) / np.maximum(sec_power_w[so].sum(1), 1e-6)       # bits / J
        cnt = np.bincount(oru, minlength=N)
        g = [np.log10(1.0 + ee)[:, None] / 7.0, np.minimum(cnt / K, 2.0)[:, None]]
        if self.thpt_lim:
            g.append(sec_util[so])
        if self.state_feats:
            g.append(np.eye(4)[sim.state[so]].reshape(N, -1))
            g.append(np.eye(5)[np.clip(sim.last_actions[so], 0, 4)].reshape(N, -1))
        out = np.concatenate([ue_set.reshape(N, -1), mask, np.concatenate(g, 1)], 1)
        return np.clip(out, -1.0, 2.0).astype(np.float32)

    def fresh_local_obs(self) -> np.ndarray:
        m, S, U = self.core.sim.last, self.S, self.core.sim.U
        z = np.zeros(S)
        return self.local_obs(m.get("ue_served_bps", np.zeros(U)), m.get("served_sector_bps", z),
                              m.get("power_sector_w", z), m.get("util_sector", z))

    def _snapshot_hook(self, sim, i_mini, acc):
        if i_mini != self._hook_mini:
            return
        el = (i_mini + 1) * sim.dt_mini
        self._shared.append(self.local_obs(acc["served"] / el, acc["served_sec"] / el,
                                           acc["energy"] / el, acc["busy"] / el))

    def shared_obs(self) -> np.ndarray:
        """Neighbour-visible observation of every O-RU at this decision time, [N, L]."""
        if not self._shared:
            self._shared.append(self.fresh_local_obs())
        while len(self._shared) < self._shared.maxlen:       # episode start: hold the oldest
            self._shared.appendleft(self._shared[0])
        return self._shared[0]

    def _obs(self):
        lay = self.obs_layout
        own = self.fresh_local_obs()
        if lay.max_neighbors == 0:
            return {a: own[i] for i, a in enumerate(self.agents)}
        sh = self.shared_obs() if lay.has_shared_copy else own
        nbr = np.where((self.nbr_idx >= 0)[..., None], sh[np.maximum(self.nbr_idx, 0)], 0.0)    # [N, M, L]
        parts = [own] + ([sh] if lay.has_shared_copy else []) + [self.nbr_idx.astype(np.float32),
                                                                 nbr.reshape(self.n_oru, -1)]
        x = np.concatenate(parts, 1).astype(np.float32)
        return {a: x[i] for i, a in enumerate(self.agents)}

    def reset(self, *, seed=None, options=None):
        self._shared.clear()                                 # warm-up steps refill it
        return super().reset(seed=seed, options=options)
