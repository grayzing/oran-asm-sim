"""VDN algorithm and config for RLlib's new API stack."""
from __future__ import annotations

import numpy as np
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.dqn.dqn import DQN, DQNConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.metrics import (ENV_RUNNER_RESULTS, ENV_RUNNER_SAMPLING_TIMER, LEARNER_RESULTS,
                                     LEARNER_UPDATE_TIMER, NUM_ENV_STEPS_SAMPLED_LIFETIME, REPLAY_BUFFER_RESULTS,
                                     REPLAY_BUFFER_SAMPLE_TIMER, SYNCH_WORKER_WEIGHTS_TIMER, TIMERS)

from .learner import VDNTorchLearner
from .module import DRQNModule

TEAM_POLICY = "oru_policy"
KPIS = ("power_norm", "qos_violation_frac", "outage_frac", "drop_frac")


def map_to_team_policy(agent_id, episode=None, **kw):
    return TEAM_POLICY


class ASMKpiCallback(RLlibCallback):
    """Logs the team reward and simulator KPIs as step means."""

    def on_episode_step(self, *, episode, metrics_logger=None, **kw):
        # Per step, because (a) RLlib's episode_return sums over agents, i.e. counts the shared reward
        # n_agents times, and (b) `episode` only holds the current rollout fragment.
        rewards = list(episode.get_rewards(-1).values())
        metrics_logger.log_value("team_reward_mean", float(np.mean(rewards)), reduce="mean", window=1000)
        for info in episode.get_infos(-1).values():
            for k in KPIS:
                if k in info:
                    metrics_logger.log_value(k, float(info[k]), reduce="mean", window=1000)


class VDNConfig(DQNConfig):
    """DQNConfig with joint-sequence replay. ``train_batch_size_per_learner`` counts *sequences*;
    ``model_config["max_seq_len"]`` is their length, including ``burn_in_len``."""

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or VDN)
        self.updates_per_round = 4
        (self
         .multi_agent(policies={TEAM_POLICY}, policy_mapping_fn=map_to_team_policy)
         .callbacks(ASMKpiCallback)
         .rl_module(model_config={"fc_dim": 64, "rnn_dim": 64, "max_seq_len": 24})
         .env_runners(rollout_fragment_length=8)
         .training(lr=5e-4, gamma=0.95, train_batch_size_per_learner=32, burn_in_len=8, grad_clip=10.0,
                   double_q=True, dueling=True, n_step=1, num_atoms=1, td_error_loss_fn="huber",
                   target_network_update_freq=500, num_steps_sampled_before_learning_starts=1000,
                   epsilon=[(0, 1.0), (20_000, 0.05)],
                   # a string: Algorithm does `"EpisodeReplayBuffer" in config["type"]`
                   replay_buffer_config={"type": "oran_asm_sim.vdn.replay.JointEpisodeReplayBuffer",
                                         "capacity": 50_000}))

    def training(self, *, updates_per_round=NotProvided, **kwargs):
        """``updates_per_round``: gradient steps after each round of env-runner sampling."""
        super().training(**kwargs)
        if updates_per_round is not NotProvided:
            self.updates_per_round = updates_per_round
        return self

    def get_default_rl_module_spec(self):
        return RLModuleSpec(module_class=DRQNModule, model_config=self.model_config)

    def get_default_learner_class(self):
        return VDNTorchLearner

    def validate(self) -> None:
        super().validate()
        if self.n_step != 1 or self.num_atoms != 1 or self.noisy:
            self._value_error("VDN supports n_step=1, num_atoms=1 and noisy=False only.")
        if self.training_intensity:
            self._value_error("VDN uses `updates_per_round` instead of `training_intensity`.")
        if len(self.policies) != 1:
            self._value_error("VDN shares one Q-network between all agents: define exactly one policy.")


class VDN(DQN):
    @classmethod
    def get_default_config(cls) -> AlgorithmConfig:
        return VDNConfig()

    def training_step(self) -> None:
        cfg, m = self.config, self.metrics
        with m.log_time((TIMERS, ENV_RUNNER_SAMPLING_TIMER)):
            episodes, results = synchronous_parallel_sample(
                worker_set=self.env_runner_group, concat=True, sample_timeout_s=cfg.sample_timeout_s,
                _uses_new_env_runners=True, _return_metrics=True)
        m.aggregate(results, key=ENV_RUNNER_RESULTS)
        self.local_replay_buffer.add(episodes)

        ts = m.peek((ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME), default=0)
        if len(self.local_replay_buffer) == 0 or ts < cfg.num_steps_sampled_before_learning_starts:
            return
        policy_id = next(iter(cfg.policies))
        for _ in range(cfg.updates_per_round):
            with m.log_time((TIMERS, REPLAY_BUFFER_SAMPLE_TIMER)):
                rows = self.local_replay_buffer.sample(cfg.total_train_batch_size, burn_in=cfg.burn_in_len,
                                                       seq_len=cfg.model_config["max_seq_len"])
            with m.log_time((TIMERS, LEARNER_UPDATE_TIMER)):
                learner_results = self.learner_group.update(
                    batch=MultiAgentBatch({policy_id: rows}, env_steps=int(rows["loss_mask"].sum())),
                    timesteps={NUM_ENV_STEPS_SAMPLED_LIFETIME: ts})
            m.aggregate(learner_results, key=LEARNER_RESULTS)
        m.aggregate([self.local_replay_buffer.get_metrics()], key=REPLAY_BUFFER_RESULTS)
        with m.log_time((TIMERS, SYNCH_WORKER_WEIGHTS_TIMER)):
            self.env_runner_group.sync_weights(from_worker_or_learner_group=self.learner_group,
                                               policies={policy_id}, inference_only=True)
