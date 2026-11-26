"""Recurrent deep Q-network (DRQN) RLModule, shared by all O-RUs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import torch
from ray.rllib.algorithms.dqn.dqn_learner import QF_NEXT_PREDS, QF_PREDS, QF_TARGET_NEXT_PREDS
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.utils import make_target_network
from ray.rllib.core.rl_module.apis import InferenceOnlyAPI, QNetAPI, TargetNetworkAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.schedules.scheduler import Scheduler
from torch import nn

State = Dict[str, torch.Tensor]


class RecurrentQNet(nn.Module):
    """obs [B, T, O] -> Linear/ReLU -> LSTM -> (dueling) Q-values [B, T, H, A].

    One Q-head per action dimension (H = 1 for a Discrete action, H = 3 for an O-RU
    commanding its three sectors). The agent's Q-value is the sum over its heads."""

    def __init__(self, obs_dim: int, n_heads: int, n_actions: int, fc_dim: int, rnn_dim: int, dueling: bool):
        super().__init__()
        self.rnn_dim, self.n_heads, self.n_actions = rnn_dim, n_heads, n_actions
        self.fc = nn.Sequential(nn.Linear(obs_dim, fc_dim), nn.ReLU())
        self.rnn = nn.LSTM(fc_dim, rnn_dim, batch_first=True)
        self.adv = nn.Linear(rnn_dim, n_heads * n_actions)
        self.val = nn.Linear(rnn_dim, n_heads) if dueling else None

    def forward(self, obs: torch.Tensor, state: Optional[State] = None) -> Tuple[torch.Tensor, State]:
        hc = None if state is None else (state["h"].unsqueeze(0).contiguous(), state["c"].unsqueeze(0).contiguous())
        x, (h, c) = self.rnn(self.fc(obs), hc)
        q = self.adv(x).unflatten(-1, (self.n_heads, self.n_actions))
        if self.val is not None:
            q = self.val(x).unsqueeze(-1) + q - q.mean(-1, keepdim=True)
        return q, {"h": h.squeeze(0), "c": c.squeeze(0)}


class DRQNModule(TorchRLModule, InferenceOnlyAPI, TargetNetworkAPI, QNetAPI):
    """Per-agent recurrent Q-function.

    Action space: ``Discrete(A)`` or ``MultiDiscrete([A] * H)`` (one head per dimension). For
    MultiDiscrete, Q-values carry a head axis [..., H, A] and actions [..., H]; for Discrete,
    the head axis is dropped.

    Acting:   obs [B, 1, O] (B = agents) + the agent's own LSTM state -> eps-greedy action.
    Training: obs [B, T+1, N, O] (joint sequences, N agents) -> Q-values [B, T, N, (H,) A]; the
              mixing (sum over heads and agents) is done by the Learner.
    """

    def setup(self):
        if not (isinstance(self.observation_space, gym.spaces.Box) and len(self.observation_space.shape) == 1):
            raise ValueError("DRQNModule needs a flat Box observation (ma_obs_mode='pooled'), got "
                             f"{self.observation_space}")
        sp = self.action_space
        if isinstance(sp, gym.spaces.Discrete):
            self.multi, n_heads, n_actions = False, 1, int(sp.n)
        elif isinstance(sp, gym.spaces.MultiDiscrete) and sp.nvec.ndim == 1 and len(set(sp.nvec.tolist())) == 1:
            self.multi, n_heads, n_actions = True, len(sp.nvec), int(sp.nvec[0])
        else:
            raise ValueError(f"DRQNModule needs a Discrete or uniform 1-D MultiDiscrete action space, got {sp}")
        mc = self.model_config
        self.double_q = mc["double_q"]
        self.epsilon = Scheduler(fixed_value_or_schedule=mc["epsilon"], framework="torch")
        self.net = RecurrentQNet(self.observation_space.shape[0], n_heads, n_actions,
                                 mc["fc_dim"], mc["rnn_dim"], mc["dueling"])

    # -------------------------------------------------------------- acting
    def get_initial_state(self) -> State:
        return {k: torch.zeros(self.net.rnn_dim) for k in ("h", "c")}

    def _act(self, batch: Dict[str, Any], epsilon: float) -> Dict[str, Any]:
        out = self.compute_q_values(batch)
        q = out[QF_PREDS]
        a = q.argmax(-1)                                     # [B, T(, H)]
        if epsilon > 0:
            # Masks share the shape of `a`, so every head explores independently. (RLlib's default
            # DQN module draws a [B] mask, which broadcasts to [B, B] against recurrent [B, 1] actions.)
            rnd = torch.randint_like(a, q.shape[-1])
            a = torch.where(torch.rand(a.shape, device=a.device) < epsilon, rnd, a)
        return {Columns.ACTIONS: a, Columns.STATE_OUT: out[Columns.STATE_OUT]}

    def _forward_inference(self, batch, **kw):
        return self._act(batch, 0.0)

    def _forward_exploration(self, batch, t: int = 0, **kw):
        self.epsilon.update(t)
        return self._act(batch, float(self.epsilon.get_current_value()))

    # ------------------------------------------------------------ training
    def _forward_train(self, batch, **kw):
        obs = batch[Columns.OBS]
        B, T1, N, _ = obs.shape
        seq = obs.transpose(1, 2).reshape(B * N, T1, -1)             # one LSTM sequence per agent

        def unfold(q):
            q = self._heads(q)
            return q.reshape(B, N, T1, *q.shape[2:]).transpose(1, 2)    # [B, T+1, N, (H,) A]

        # Online and target nets each unroll all T+1 observations from a zero state, so Q(t) and the
        # bootstrap Q(t+1) always see the hidden state of their own network.
        q = unfold(self.net(seq)[0])
        out = {QF_PREDS: q[:, :-1]}
        with torch.no_grad():
            out[QF_TARGET_NEXT_PREDS] = unfold(self._target_net(seq)[0])[:, 1:]
        if self.double_q:
            out[QF_NEXT_PREDS] = q[:, 1:].detach()
        return out

    # ---------------------------------------------------------- RLlib APIs
    def _heads(self, q: torch.Tensor) -> torch.Tensor:
        return q if self.multi else q.squeeze(-2)

    def compute_q_values(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        q, state = self.net(batch[Columns.OBS], batch.get(Columns.STATE_IN))
        return {QF_PREDS: self._heads(q), Columns.STATE_OUT: state}

    def forward_target(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        q, state = self._target_net(batch[Columns.OBS], batch.get(Columns.STATE_IN))
        return {QF_PREDS: self._heads(q), Columns.STATE_OUT: state}

    def make_target_networks(self) -> None:
        self._target_net = make_target_network(self.net)

    def get_target_network_pairs(self) -> List[Tuple[nn.Module, nn.Module]]:
        return [(self.net, self._target_net)]

    def get_non_inference_attributes(self) -> List[str]:
        return ["_target_net"]
