"""Replay of *joint* sequences.

RLlib's MultiAgentEpisodeReplayBuffer samples independent single-agent transitions: no sequences
(needed by the LSTM) and no guarantee that the agents of a row share a time step (needed by the sum
over agents). Here one row is a contiguous window of the whole team, [T, N, ...].

With ``obs_layout`` (observations of :class:`~oran_asm_sim.graph_env.ORanASMGraphEnv`), only each
agent's own (+ shared) observation is stored, ``[..., :compact_dim]``; the neighbour blocks are
rebuilt by the Q-network from the static neighbour index, returned as ``nbr_idx`` [B, N, M]."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
from ray.rllib.core.columns import Columns
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.replay_buffers.base import ReplayBufferInterface

AGENT_REWARDS = "agent_rewards"
NBR_IDX = "nbr_idx"


class JointEpisodeReplayBuffer(ReplayBufferInterface):
    """FIFO over episodes; ``capacity`` counts env steps. ``rewards`` are averaged over the agents, i.e.
    the team reward (identical for all agents unless ``local_reward_mix`` > 0); ``agent_rewards``
    [B, T, N] keeps them per agent (independent Q-learning). ``storage_dtype`` (e.g. "float16")
    applies to stored observations; samples are float32."""

    def __init__(self, capacity: int = 50_000, seed: Optional[int] = None, obs_layout=None,
                 storage_dtype: str = "float32", **kwargs):
        self.capacity = int(capacity)
        self.layout = None
        if obs_layout is not None:
            from ..graph_env import ObsLayout
            self.layout = ObsLayout.of(obs_layout)
        self.dtype = np.dtype(storage_dtype)
        self.rng = np.random.default_rng(seed)
        self.episodes: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._n = 0                                      # env steps stored
        self.metrics = MetricsLogger(root=False)

    def __len__(self) -> int:
        return self._n

    def get_metrics(self):
        self.metrics.log_value("num_env_steps_stored", self._n, reduce="mean", window=1)
        self.metrics.log_value("num_episodes_stored", len(self.episodes), reduce="mean", window=1)
        return self.metrics.reduce()

    # ----------------------------------------------------------------- add
    def add(self, episodes: List[MultiAgentEpisode], **kwargs) -> None:
        for eps in episodes if isinstance(episodes, (list, tuple)) else [episodes]:
            self._add_chunk(eps)
        # evict oldest episodes, but always keep the most recent one
        while self._n > self.capacity and len(self.episodes) > 1:
            _, old = self.episodes.popitem(last=False)
            self._n -= len(old["actions"])

    def _add_chunk(self, eps: MultiAgentEpisode) -> None:
        agents = [eps.agent_episodes[a] for a in sorted(eps.agent_episodes)]
        T = len(agents[0]) if agents else 0
        if any(len(a) != T for a in agents):
            raise ValueError("VDN requires all agents to act at every env step")
        if T == 0:
            return
        obs = np.stack([np.asarray(a.get_observations(), np.float32) for a in agents], 1)     # [T+1, N, O]
        act = np.stack([np.asarray(a.get_actions()) for a in agents], 1).astype(np.int16)     # [T, N(, H)]
        rew = np.stack([np.asarray(a.get_rewards(), np.float32) for a in agents], 1)          # [T, N]
        nbr = None
        if self.layout is not None:
            c, M = self.layout.compact_dim, self.layout.max_neighbors
            nbr = obs[0, :, c:c + M].round().astype(np.int64)                                # [N, M], static
            obs = obs[..., :c]
        obs = obs.astype(self.dtype)
        # Rollout fragments of one episode arrive in order under the same id. If its earlier part was
        # evicted, the continuation simply starts a new stored episode.
        ep = self.episodes.get(eps.id_)
        if ep is None:
            ep = self.episodes[eps.id_] = {"obs": [obs[0]], "actions": [], "rewards": [], "terminated": False,
                                           "nbr_idx": nbr}
        ep["obs"].extend(obs[1:])
        ep["actions"].extend(act)
        ep["rewards"].extend(rew)
        ep["terminated"] = bool(eps.is_terminated)       # time-limit truncation still bootstraps
        self._n += T

    # -------------------------------------------------------------- sample
    def sample(self, num_items: Optional[int] = None, *, seq_len: int = 32, burn_in: int = 0, **kwargs) -> SampleBatch:
        """``num_items`` rows of up to ``seq_len`` transitions, uniformly over stored steps. The first
        ``burn_in`` ones only warm up the LSTM (masked from the loss); windows at an episode start
        need no burn-in, since the zero state is exact there. Rows are right-padded."""
        train_len = seq_len - burn_in
        eps = list(self.episodes.values())
        lens = np.array([len(e["actions"]) for e in eps])
        first = eps[0]
        N, O = first["obs"][0].shape
        B = int(num_items or 1)
        obs = np.zeros((B, seq_len + 1, N, O), np.float32)
        act = np.zeros((B, seq_len, *first["actions"][0].shape), np.int64)      # [N] or [N, H]
        rew = np.zeros((B, seq_len, N), np.float32)
        nbr = np.zeros((B, *first["nbr_idx"].shape), np.int64) if first.get("nbr_idx") is not None else None
        term = np.zeros((B, seq_len), bool)
        mask = np.zeros((B, seq_len), bool)
        for b, i in enumerate(self.rng.choice(len(eps), size=B, p=lens / lens.sum())):
            e, L = eps[i], lens[i]
            t0 = int(self.rng.integers(max(L - train_len, 0) + 1))      # first trained step
            s, end = max(t0 - burn_in, 0), min(t0 + train_len, L)
            n = end - s
            obs[b, :n + 1] = np.stack(e["obs"][s:end + 1])
            act[b, :n] = np.stack(e["actions"][s:end])
            rew[b, :n] = e["rewards"][s:end]
            mask[b, t0 - s:n] = True
            if nbr is not None:
                nbr[b] = e["nbr_idx"]
            term[b, n - 1] = e["terminated"] and end == L
        out = {Columns.OBS: obs, Columns.ACTIONS: act, Columns.REWARDS: rew.mean(-1), AGENT_REWARDS: rew,
               Columns.TERMINATEDS: term, Columns.LOSS_MASK: mask}
        if nbr is not None:
            out[NBR_IDX] = nbr
        return SampleBatch(out)

    # ---------------------------------------------------------- checkpoint
    def get_state(self) -> Dict[str, Any]:
        return {"episodes": list(self.episodes.items()), "n": self._n, "rng": self.rng.bit_generator.state}

    def set_state(self, state: Dict[str, Any]) -> None:
        self.episodes = OrderedDict(state["episodes"])
        self._n = state["n"]
        self.rng.bit_generator.state = state["rng"]
