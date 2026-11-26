"""VDN loss: TD error of Q_tot = sum_i Q_i on the team reward. An agent with a MultiDiscrete action
(an O-RU commanding several sectors) has Q_i = sum over its per-dimension heads."""
from __future__ import annotations

import torch
from ray.rllib.algorithms.dqn.dqn_learner import (QF_LOSS_KEY, QF_MAX_KEY, QF_MEAN_KEY, QF_MIN_KEY, QF_NEXT_PREDS,
                                                  QF_PREDS, QF_TARGET_NEXT_PREDS, TD_ERROR_MEAN_KEY, DQNLearner)
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from torch import nn


class VDNTorchLearner(DQNLearner, TorchLearner):
    """Batch (from :class:`JointEpisodeReplayBuffer`): obs [B, T+1, N, O], actions [B, T, N(, H)],
    rewards / terminateds / loss_mask [B, T]. Q-values are [B, T, N, (H,) A].
    Target-network syncing is inherited from DQNLearner."""

    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        q = fwd_out[QF_PREDS]
        act = batch[Columns.ACTIONS].long()
        q_taken = q.gather(-1, act.unsqueeze(-1)).squeeze(-1)
        q_tot = q_taken.flatten(2).sum(-1)                                         # [B, T]

        q_next = fwd_out[QF_TARGET_NEXT_PREDS]
        # Q_tot is additive, so its joint argmax is the vector of per-head argmaxes.
        best = (fwd_out[QF_NEXT_PREDS] if config.double_q else q_next).argmax(-1, keepdim=True)
        q_tot_next = q_next.gather(-1, best).squeeze(-1).flatten(2).sum(-1)
        target = batch[Columns.REWARDS] + config.gamma * (1.0 - batch[Columns.TERMINATEDS].float()) * q_tot_next
        target = target.detach()

        mask = batch[Columns.LOSS_MASK].float()
        n = mask.sum().clamp(min=1.0)
        loss_fn = nn.HuberLoss if config.td_error_loss_fn == "huber" else nn.MSELoss
        loss = (loss_fn(reduction="none")(q_tot, target) * mask).sum() / n

        valid = q_tot[mask.bool()].detach()
        self.metrics.log_dict({
            QF_LOSS_KEY: loss,
            QF_MEAN_KEY: valid.mean(),
            QF_MIN_KEY: valid.min(),
            QF_MAX_KEY: valid.max(),
            TD_ERROR_MEAN_KEY: ((q_tot - target).abs() * mask).sum() / n,
        }, key=module_id, window=1)
        return loss
