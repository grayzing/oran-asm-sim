"""Recurrent Q-network over set-structured O-RU observations, with one round of message passing
between neighbouring O-RUs (for :class:`~oran_asm_sim.graph_env.ORanASMGraphEnv`).

Per O-RU i (parameters shared by all O-RUs)::

    h_i  = rho([ sum_{j in H_i} lambda(x_j, s_ij, gamma_ij, sector_j) | O-RU features ])      DeepSet
    m_i  = max_{k in N(i)} psi([h_i | h~_k])          one GNN layer, max-pool over the neighbours,
                                                      h~_k = encoding of k's (stale) shared observation
    z_i  = ReLU(Lin([h_i | m_i]))                     (Lin(h_i) without message passing)
    Q_i  = dueling heads(LSTM(z_i))                   one head per sector, Q_i = sum of the heads

Acting (decentralised): each O-RU's flat observation holds its own local observation and its
neighbours' shared ones (see :class:`~oran_asm_sim.graph_env.ObsLayout`).
Training: the joint replay batch holds each O-RU's own + shared observation once, [B, T+1, N, C],
plus ``nbr_idx`` [B, N, M]; neighbour encodings are gathered from it. Both paths compute the same
function (tested).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from ray.rllib.algorithms.dqn.dqn_learner import QF_NEXT_PREDS, QF_PREDS, QF_TARGET_NEXT_PREDS
from ray.rllib.core.columns import Columns
from torch import nn

from ..graph_env import ObsLayout
from .module import DRQNModule, RecurrentQNet, State

NBR_IDX = "nbr_idx"


class DeepSetEncoder(nn.Module):
    """Local observation [..., L] -> embedding [..., d]; permutation invariant in the UE rows."""

    def __init__(self, lay: ObsLayout, d: int):
        super().__init__()
        self.lay = lay
        self.phi = nn.Sequential(nn.Linear(lay.ue_dim, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(d + lay.oru_dim, d), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K, F = self.lay.max_ues, self.lay.ue_dim
        lead = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        valid = flat[:, K * F:K * F + K] > 0.5                       # [P, K] binary mask
        # phi only on the occupied UE slots (most of the padded set is empty), then a scatter-sum
        rows = flat[:, :K * F].reshape(-1, K, F)[valid]               # [V, F]
        owner = valid.nonzero()[:, 0]
        emb = self.phi(rows)
        pooled = emb.new_zeros(flat.shape[0], emb.shape[-1]).index_add(0, owner, emb)
        h = self.rho(torch.cat([pooled, flat[:, K * F + K:]], -1))
        return h.reshape(*lead, -1)


class GraphRecurrentQNet(nn.Module):
    def __init__(self, lay: ObsLayout, n_heads: int, n_actions: int, enc_dim: int, fc_dim: int, rnn_dim: int,
                 dueling: bool, message_passing: bool = True):
        super().__init__()
        self.lay = lay
        self.mp = bool(message_passing) and lay.max_neighbors > 0
        self.enc = DeepSetEncoder(lay, enc_dim)
        self.msg = nn.Sequential(nn.Linear(2 * enc_dim, enc_dim), nn.ReLU()) if self.mp else None
        # q.fc is the "Lin" after the max-pool; then LSTM and the dueling sector heads
        self.q = RecurrentQNet(enc_dim * (2 if self.mp else 1), n_heads, n_actions, fc_dim, rnn_dim, dueling)

    @property
    def rnn_dim(self) -> int:
        return self.q.rnn_dim

    def _aggregate(self, h: torch.Tensor, hn: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """h [..., d], neighbour embeddings hn [..., M, d], valid [..., M] -> [h | max-pooled messages]."""
        e = self.msg(torch.cat([h.unsqueeze(-2).expand_as(hn), hn], -1))
        m = e.masked_fill(~valid.unsqueeze(-1), 0.0).amax(-2)    # messages are >= 0 (ReLU): 0 if no neighbour
        return torch.cat([h, m], -1)

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        """Acting format: per-agent observations [..., obs_dim]."""
        lay, L = self.lay, self.lay.local_dim
        h = self.enc(obs[..., :L])
        if not self.mp:
            return h
        c, M = lay.compact_dim, lay.max_neighbors
        idx = obs[..., c:c + M]
        hn = self.enc(obs[..., c + M:].unflatten(-1, (M, L)))
        return self._aggregate(h, hn, idx >= 0)

    def embed_joint(self, obs: torch.Tensor, nbr_idx: Optional[torch.Tensor]) -> torch.Tensor:
        """Training format: obs [B, T, N, compact_dim], nbr_idx [B, N, M] -> [B, T, N, d_in]."""
        L = self.lay.local_dim
        h = self.enc(obs[..., :L])
        if not self.mp:
            return h
        hs = self.enc(obs[..., L:2 * L]) if self.lay.has_shared_copy else h
        B, T, N, d = hs.shape
        idx = nbr_idx.long()
        bi = torch.arange(B, device=obs.device)[:, None, None]
        hn = hs.transpose(1, 2)[bi, idx.clamp(min=0)].permute(0, 3, 1, 2, 4)      # [B, T, N, M, d]
        valid = (idx >= 0).unsqueeze(1).expand(B, T, N, idx.shape[-1])
        return self._aggregate(h, hn, valid)

    def forward(self, obs: torch.Tensor, state: Optional[State] = None) -> Tuple[torch.Tensor, State]:
        return self.q(self.embed(obs), state)

    def forward_joint(self, obs: torch.Tensor, nbr_idx: Optional[torch.Tensor]) -> torch.Tensor:
        """Unrolls every agent's sequence from a zero state: [B, T, N, H, A]."""
        x = self.embed_joint(obs, nbr_idx)
        B, T, N, _ = x.shape
        q, _ = self.q(x.transpose(1, 2).reshape(B * N, T, -1))
        return q.reshape(B, N, T, *q.shape[2:]).transpose(1, 2)


class GraphDRQNModule(DRQNModule):
    """:class:`DRQNModule` on :class:`GraphRecurrentQNet`. ``model_config`` needs ``obs_layout``
    (``ObsLayout.to_dict()``); optional ``enc_dim`` (default ``fc_dim``) and ``message_passing``
    (default True; always off when the env shares no observations)."""

    def _make_net(self, n_heads: int, n_actions: int) -> nn.Module:
        mc = self.model_config
        lay = ObsLayout.of(mc["obs_layout"])
        if self.observation_space.shape[0] != lay.obs_dim:
            raise ValueError(f"obs_layout expects obs_dim {lay.obs_dim}, got {self.observation_space.shape}")
        return GraphRecurrentQNet(lay, n_heads, n_actions, mc.get("enc_dim") or mc["fc_dim"], mc["fc_dim"],
                                  mc["rnn_dim"], mc["dueling"], mc.get("message_passing", True))

    def _forward_train(self, batch, **kw):
        obs, idx = batch[Columns.OBS], batch.get(NBR_IDX)
        q = self._heads(self.net.forward_joint(obs, idx))
        out = {QF_PREDS: q[:, :-1]}
        with torch.no_grad():
            out[QF_TARGET_NEXT_PREDS] = self._heads(self._target_net.forward_joint(obs, idx))[:, 1:]
        if self.double_q:
            out[QF_NEXT_PREDS] = q[:, 1:].detach()
        return out
