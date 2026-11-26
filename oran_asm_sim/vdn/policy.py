"""Greedy decentralised execution of a trained DRQN outside RLlib, as a ``baselines.run_policy`` policy."""
from __future__ import annotations

import numpy as np
import torch
from ray.rllib.algorithms.dqn.dqn_learner import QF_PREDS
from ray.rllib.core.columns import Columns


class DRQNPolicy:
    """``run_policy(ma_env.core, DRQNPolicy(module, ma_env), ...)``. The LSTM state is kept per O-RU;
    call :meth:`reset` before every episode."""

    name = "VDN (recurrent DQN)"

    def __init__(self, module, ma_env):
        self.module, self.ma_env = module, ma_env
        self.reset()

    def reset(self):
        n = len(self.ma_env.agents)
        self.state = {k: v.expand(n, -1) for k, v in self.module.get_initial_state().items()}

    @torch.no_grad()
    def __call__(self, env, obs=None):
        o = self.ma_env._obs()
        x = torch.from_numpy(np.stack([o[a] for a in self.ma_env.agents])).unsqueeze(1)      # [n_oru, 1, O]
        out = self.module.compute_q_values({Columns.OBS: x, Columns.STATE_IN: self.state})
        self.state = out[Columns.STATE_OUT]
        act = out[QF_PREDS][:, 0].argmax(-1).numpy()                                        # [n_oru, H]
        return env.action_levels[self.ma_env.sector_actions(act)]
