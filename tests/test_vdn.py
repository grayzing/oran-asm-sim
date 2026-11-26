"""VDN + recurrent DQN (skipped without Ray / torch)."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("ray.rllib")

import gymnasium as gym  # noqa: E402
import ray  # noqa: E402
import torch  # noqa: E402
from ray.rllib.core.columns import Columns  # noqa: E402
from ray.rllib.env.multi_agent_env import MultiAgentEnv  # noqa: E402
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode  # noqa: E402

from oran_asm_sim import ORanASMMultiAgentEnv  # noqa: E402
from oran_asm_sim.baselines import run_policy  # noqa: E402
from oran_asm_sim.vdn import (TEAM_POLICY, VDN, DRQNModule, DRQNPolicy, JointEpisodeReplayBuffer,  # noqa: E402
                              VDNConfig)

SMALL = {"geometry": {"layout": "rings", "n_rings": 1}, "env": {"warmup_s": 2, "n_mini": 2, "seed": 0,
                                                                "episode_steps": 20, "random_start": True}}


@pytest.fixture(scope="module")
def ray_local():
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, object_store_memory=150_000_000,
             ignore_reinit_error=True)
    yield
    ray.shutdown()


def _module(obs_dim=3, n_actions=4, n_heads=None, **mc):
    cfg = dict(double_q=True, dueling=True, epsilon=0.5, fc_dim=8, rnn_dim=8, max_seq_len=6, **mc)
    act = gym.spaces.Discrete(n_actions) if n_heads is None else gym.spaces.MultiDiscrete([n_actions] * n_heads)
    return DRQNModule(observation_space=gym.spaces.Box(-9, 9, (obs_dim,), np.float32),
                      action_space=act, model_config=cfg)


# ------------------------------------------------------------------ module
def test_stepwise_acting_equals_training_unroll():
    """Acting one step at a time with carried state gives the Q-values used in the loss, per agent."""
    m = _module()
    m.make_target_networks()
    obs = torch.randn(2, 6, 5, 3)                                       # B, T+1, N, O
    out = m._forward_train({Columns.OBS: obs})
    assert out["qf_preds"].shape == out["qf_target_next_preds"].shape == (2, 5, 5, 4)
    state = {k: v.expand(5, -1) for k, v in m.get_initial_state().items()}
    for t in range(6):
        o = m.compute_q_values({Columns.OBS: obs[1, t].unsqueeze(1), Columns.STATE_IN: state})
        state = o[Columns.STATE_OUT]
        if t < 5:
            assert torch.allclose(o["qf_preds"][:, 0], out["qf_preds"][1, t], atol=1e-5)
        if t > 0:                                                       # target net == online net here
            assert torch.allclose(o["qf_preds"][:, 0], out["qf_target_next_preds"][1, t - 1], atol=1e-5)


def test_exploration_shapes_and_rate():
    m = _module()
    batch = {Columns.OBS: torch.zeros(4000, 1, 3),
             Columns.STATE_IN: {k: v.expand(4000, -1) for k, v in m.get_initial_state().items()}}
    a = m._forward_exploration(batch, t=0)[Columns.ACTIONS]
    g = m._forward_inference(batch)[Columns.ACTIONS]
    assert a.shape == g.shape == (4000, 1)
    assert (a != g).float().mean().item() == pytest.approx(0.5 * 3 / 4, abs=0.04)


def test_multidiscrete_heads_shapes_and_exploration():
    m = _module(n_heads=3)
    m.make_target_networks()
    out = m._forward_train({Columns.OBS: torch.randn(2, 6, 5, 3)})
    assert out["qf_preds"].shape == out["qf_target_next_preds"].shape == (2, 5, 5, 3, 4)
    batch = {Columns.OBS: torch.zeros(4000, 1, 3),
             Columns.STATE_IN: {k: v.expand(4000, -1) for k, v in m.get_initial_state().items()}}
    a = m._forward_exploration(batch, t=0)[Columns.ACTIONS]
    g = m._forward_inference(batch)[Columns.ACTIONS]
    assert a.shape == g.shape == (4000, 1, 3)
    assert (a != g).float().mean().item() == pytest.approx(0.5 * 3 / 4, abs=0.04)


# ------------------------------------------------------------------ replay
def _episode(T, N=3, terminated=False, split=None):
    """Observation = [agent, t]; returns the episode as chunks, as env runners deliver them."""
    ids = [f"a{i}" for i in range(N)]
    sp = gym.spaces.Box(-99, 99, (2,), np.float32)
    eps = MultiAgentEpisode(observation_space={a: sp for a in ids},
                            action_space={a: gym.spaces.Discrete(99) for a in ids})
    def ob(t):
        return {a: np.array([i, t], np.float32) for i, a in enumerate(ids)}
    eps.add_env_reset(observations=ob(0))
    chunks = []
    for t in range(T):
        last = t == T - 1
        eps.add_env_step(ob(t + 1), {a: t for a in ids}, {a: float(t) for a in ids},
                         terminateds={"__all__": last and terminated},
                         truncateds={"__all__": last and not terminated})
        if split is not None and t + 1 == split:
            chunks.append(eps)
            eps = eps.cut(len_lookback_buffer=1)
    return chunks + [eps]


@pytest.mark.parametrize("split", [None, 7])
def test_replay_rows_are_time_aligned_joint_sequences(split):
    buf = JointEpisodeReplayBuffer(seed=0)
    for c in _episode(20, split=split):
        buf.add([c])
    assert len(buf) == 20 and len(buf.episodes) == 1
    s = buf.sample(200, seq_len=8, burn_in=3)
    starts = set()
    for o, a, r, m in zip(s["obs"], s["actions"], s["rewards"], s["loss_mask"]):
        t0 = int(o[0, 0, 1])
        assert np.array_equal(o[..., 0], np.tile(np.arange(3), (9, 1))) or not m[-1]
        n = int(np.nonzero(m)[0][-1]) + 1
        assert np.array_equal(o[:n + 1, :, 1], np.tile(np.arange(t0, t0 + n + 1)[:, None], (1, 3)))
        assert np.array_equal(a[:n], np.tile(np.arange(t0, t0 + n)[:, None], (1, 3)))
        assert np.array_equal(r[:n], np.arange(t0, t0 + n))
        burn = int(np.nonzero(m)[0][0])
        assert m.sum() == 5 and (burn == 3 or t0 == 0)          # burn-in only skipped at episode starts
        starts.add(t0 + burn)
    assert starts == set(range(16))                             # every training window is reachable


def test_replay_short_episode_termination_and_eviction():
    buf = JointEpisodeReplayBuffer(capacity=25, seed=0)
    buf.add(_episode(4, terminated=True))
    s = buf.sample(5, seq_len=8, burn_in=2)
    assert (s["loss_mask"].sum(1) == 4).all()
    assert (s["terminateds"][:, 3]).all() and s["terminateds"].sum() == 5
    buf.add(_episode(20))
    assert len(buf) == 24
    buf.add(_episode(20))
    assert len(buf) == 20 and len(buf.episodes) == 1
    assert not buf.sample(50, seq_len=8)["terminateds"].any()   # truncation bootstraps


# ------------------------------------------------------------------ loss
def test_vdn_loss_matches_hand_computation(ray_local):
    cfg = (VDNConfig().environment(ORanASMMultiAgentEnv, env_config=SMALL).env_runners(num_env_runners=0)
           .training(gamma=0.9, td_error_loss_fn="mse"))
    algo = cfg.build_algo()
    learner = algo.learner_group._learner
    B, T, N, A = 2, 3, 4, 5
    rng = torch.Generator().manual_seed(0)
    q, qn, qt = (torch.rand(B, T, N, A, generator=rng) for _ in range(3))
    batch = {"actions": torch.randint(A, (B, T, N), generator=rng), "rewards": torch.rand(B, T, generator=rng),
             "terminateds": torch.zeros(B, T, dtype=torch.bool), "loss_mask": torch.ones(B, T, dtype=torch.bool)}
    batch["terminateds"][1, 2] = True
    batch["loss_mask"][0, 0] = False
    loss = learner.compute_loss_for_module(
        module_id=TEAM_POLICY, config=algo.config, batch=batch,
        fwd_out={"qf_preds": q, "qf_next_preds": qn, "qf_target_next_preds": qt})
    terms = []
    for b in range(B):
        for t in range(T):
            if not batch["loss_mask"][b, t]:
                continue
            tot = sum(q[b, t, i, batch["actions"][b, t, i]] for i in range(N))
            nxt = sum(qt[b, t, i, qn[b, t, i].argmax()] for i in range(N))
            y = batch["rewards"][b, t] + 0.9 * (0.0 if batch["terminateds"][b, t] else 1.0) * nxt
            terms.append((tot - y) ** 2)
    assert loss.item() == pytest.approx(torch.stack(terms).mean().item(), rel=1e-5)
    algo.stop()


def test_vdn_loss_sums_heads_and_agents(ray_local):
    cfg = (VDNConfig().environment(ORanASMMultiAgentEnv, env_config=SMALL).env_runners(num_env_runners=0)
           .training(gamma=0.9, td_error_loss_fn="mse"))
    algo = cfg.build_algo()
    learner = algo.learner_group._learner
    B, T, N, H, A = 2, 3, 4, 3, 5
    rng = torch.Generator().manual_seed(1)
    q, qn, qt = (torch.rand(B, T, N, H, A, generator=rng) for _ in range(3))
    batch = {"actions": torch.randint(A, (B, T, N, H), generator=rng), "rewards": torch.rand(B, T, generator=rng),
             "terminateds": torch.zeros(B, T, dtype=torch.bool), "loss_mask": torch.ones(B, T, dtype=torch.bool)}
    loss = learner.compute_loss_for_module(
        module_id=TEAM_POLICY, config=algo.config, batch=batch,
        fwd_out={"qf_preds": q, "qf_next_preds": qn, "qf_target_next_preds": qt})
    terms = []
    for b in range(B):
        for t in range(T):
            tot = sum(q[b, t, i, h, batch["actions"][b, t, i, h]] for i in range(N) for h in range(H))
            nxt = sum(qt[b, t, i, h, qn[b, t, i, h].argmax()] for i in range(N) for h in range(H))
            terms.append((tot - (batch["rewards"][b, t] + 0.9 * nxt)) ** 2)
    assert loss.item() == pytest.approx(torch.stack(terms).mean().item(), rel=1e-5)
    algo.stop()


# ------------------------------------------------------------------ end to end
def test_train_checkpoint_restore_and_execute(ray_local, tmp_path):
    cfg = (VDNConfig().environment(ORanASMMultiAgentEnv, env_config=SMALL).env_runners(num_env_runners=0)
           .training(num_steps_sampled_before_learning_starts=16, target_network_update_freq=16)
           .reporting(min_sample_timesteps_per_iteration=48, min_time_s_per_iteration=0))
    algo = cfg.build_algo()
    w0 = {k: v.clone() for k, v in algo.learner_group._learner.module[TEAM_POLICY].state_dict().items()}
    res = algo.train()
    mod = algo.learner_group._learner.module[TEAM_POLICY]
    assert np.isfinite(float(res["learners"][TEAM_POLICY]["qf_loss"]))
    er = res["env_runners"]
    kpi_reward = -(er["power_norm"] + 2 * er["qos_violation_frac"] + 4 * er["outage_frac"] + 2 * er["drop_frac"])
    assert er["team_reward_mean"] == pytest.approx(kpi_reward, rel=1e-3)
    assert any(not torch.equal(w0[k], v) for k, v in mod.state_dict().items() if k.startswith("net."))
    assert any(not torch.equal(w0[k], v) for k, v in mod.state_dict().items() if k.startswith("_target_net."))
    er = algo.env_runner.module[TEAM_POLICY]                                     # weights are synced
    assert all(torch.equal(v, mod.state_dict()[k]) for k, v in er.state_dict().items() if k.startswith("net."))

    path = algo.save_to_path(str(tmp_path / "ckpt"))
    algo.stop()
    algo2 = VDN.from_checkpoint(path)
    mod2 = algo2.learner_group._learner.module[TEAM_POLICY]
    assert all(torch.equal(v, mod2.state_dict()[k]) for k, v in mod.state_dict().items())
    algo2.train()
    algo2.stop()
    ma = ORanASMMultiAgentEnv(SMALL)
    assert len(run_policy(ma.core, DRQNPolicy(mod, ma), 3, seed=1)) == 3


class RecallGame(MultiAgentEnv):
    """A cue is visible at t=0 only (to everyone); at the last step the team is rewarded with the
    fraction of agents that repeat it. Unsolvable without memory: the memoryless optimum is 1/3."""

    def __init__(self, config=None):
        super().__init__()
        self.agents = self.possible_agents = ["a", "b", "c"]
        self.observation_spaces = {a: gym.spaces.Box(0, 1, (4,), np.float32) for a in self.agents}
        self.action_spaces = {a: gym.spaces.Discrete(3) for a in self.agents}
        self.rng = np.random.default_rng(0)

    def _obs(self):
        o = np.zeros(4, np.float32)
        if self.t == 0:
            o[self.cue] = 1
        o[3] = self.t == 3
        return {a: o.copy() for a in self.agents}

    def reset(self, *, seed=None, options=None):
        self.t, self.cue = 0, int(self.rng.integers(3))
        return self._obs(), {}

    def step(self, act):
        done = self.t == 3
        r = float(np.mean([act[a] == self.cue for a in self.agents])) if done else 0.0
        self.t += 1
        return self._obs(), {a: r for a in self.agents}, {"__all__": done}, {"__all__": False}, {}


def test_vdn_learns_a_task_that_needs_memory(ray_local):
    cfg = (VDNConfig().environment(RecallGame).env_runners(num_env_runners=0)
           .rl_module(model_config={"fc_dim": 32, "rnn_dim": 32, "max_seq_len": 4})
           .training(lr=3e-3, burn_in_len=0, updates_per_round=4, target_network_update_freq=100,
                     num_steps_sampled_before_learning_starts=200, epsilon=[(0, 1.0), (2000, 0.05)])
           .reporting(min_sample_timesteps_per_iteration=1000, min_time_s_per_iteration=0)
           .debugging(seed=0))
    algo = cfg.build_algo()
    for _ in range(6):
        algo.train()
    mod, env, score = algo.learner_group._learner.module[TEAM_POLICY], RecallGame(), []
    for _ in range(30):
        obs, _ = env.reset()
        state = {k: v.expand(3, -1) for k, v in mod.get_initial_state().items()}
        done = False
        while not done:
            x = torch.from_numpy(np.stack(list(obs.values()))).unsqueeze(1)
            out = mod._forward_inference({Columns.OBS: x, Columns.STATE_IN: state})
            state = out[Columns.STATE_OUT]
            obs, r, te, _, _ = env.step(dict(zip(env.agents, out[Columns.ACTIONS][:, 0].tolist())))
            done = te["__all__"]
        score.append(r["a"])
    algo.stop()
    assert np.mean(score) > 0.9
