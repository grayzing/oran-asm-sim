"""TD loss for VDN and its independent-Q-learning ablation.

``mixer="vdn"``: TD error of Q_tot = sum_i Q_i on the team reward (Sunehag et al., 2018).
``mixer="none"``: independent Q-learning (IQL), one TD error per agent on its own reward, averaged.

An agent with a MultiDiscrete action (an O-RU commanding several sectors) has Q_i = sum over its
per-dimension heads, in both cases."""
from __future__ import annotations

import torch
from ray.rllib.algorithms.dqn.dqn_learner import (QF_LOSS_KEY, QF_MAX_KEY, QF_MEAN_KEY, QF_MIN_KEY, QF_NEXT_PREDS,
                                                  QF_PREDS, QF_TARGET_NEXT_PREDS, TD_ERROR_MEAN_KEY, DQNLearner)
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from torch import nn

from .replay import AGENT_REWARDS


class VDNTorchLearner(DQNLearner, TorchLearner):
    """Batch (from :class:`JointEpisodeReplayBuffer`): obs [B, T+1, N, O], actions [B, T, N(, H)],
    rewards / terminateds / loss_mask [B, T], agent_rewards [B, T, N]. Q-values are [B, T, N, (H,) A].
    Target-network syncing is inherited from DQNLearner."""

    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        q = fwd_out[QF_PREDS]
        act = batch[Columns.ACTIONS].long()
        B, T, N = act.shape[:3]
        q_i = q.gather(-1, act.unsqueeze(-1)).squeeze(-1).reshape(B, T, N, -1).sum(-1)       # [B, T, N]

        q_next = fwd_out[QF_TARGET_NEXT_PREDS]
        # Q_i (and Q_tot) is additive over heads, so its argmax is the vector of per-head argmaxes.
        best = (fwd_out[QF_NEXT_PREDS] if config.double_q else q_next).argmax(-1, keepdim=True)
        q_i_next = q_next.gather(-1, best).squeeze(-1).reshape(B, T, N, -1).sum(-1)
        cont = 1.0 - batch[Columns.TERMINATEDS].float()
        mask = batch[Columns.LOSS_MASK].float()

        if getattr(config, "mixer", "vdn") == "vdn":
            pred = q_i.sum(-1)                                                                # [B, T]
            target = batch[Columns.REWARDS] + config.gamma * cont * q_i_next.sum(-1)
        else:                                                                                 # IQL
            pred = q_i
            r = batch[AGENT_REWARDS] if AGENT_REWARDS in batch else batch[Columns.REWARDS].unsqueeze(-1)
            target = r + config.gamma * cont.unsqueeze(-1) * q_i_next
            mask = mask.unsqueeze(-1).expand_as(pred)
        target = target.detach()

        n = mask.sum().clamp(min=1.0)
        loss_fn = nn.HuberLoss if config.td_error_loss_fn == "huber" else nn.MSELoss
        loss = (loss_fn(reduction="none")(pred, target) * mask).sum() / n

        valid = pred[mask.bool()].detach()
        self.metrics.log_dict({
            QF_LOSS_KEY: loss,
            QF_MEAN_KEY: valid.mean(),
            QF_MIN_KEY: valid.min(),
            QF_MAX_KEY: valid.max(),
            TD_ERROR_MEAN_KEY: ((pred - target).abs() * mask).sum() / n,
        }, key=module_id, window=1)
        return loss
