"""Graph env (shared observations with latency) + DeepSet/GNN recurrent Q-network + ablations."""
import numpy as np
import pytest

from oran_asm_sim.graph_env import ORanASMGraphEnv, ObsLayout, latency_split, sinr_to_cqi

SMALL = {"geometry": {"layout": "rings", "n_rings": 1},
         "env": {"warmup_s": 2, "n_mini": 5, "seed": 0, "episode_steps": 20, "start_hour": 13.0}}


def _env(**kw):
    return ORanASMGraphEnv({**SMALL, **kw})


def _rollout(env, n, seed=0):
    rng = np.random.default_rng(seed)
    obs = [env.reset(seed=1)[0]]
    for _ in range(n):
        obs.append(env.step({a: rng.integers(5, size=3) for a in env.agents})[0])
    return [np.stack([o[a] for a in env.agents]) for o in obs]


# ------------------------------------------------------------------ env
def test_latency_split_and_cqi():
    assert latency_split(0.0, 1.0, 10) == (0, 0, 0.0)
    assert latency_split(0.04, 1.0, 10)[:2] == (0, 1) and latency_split(0.04, 1.0, 10)[2] == pytest.approx(0.1)
    assert latency_split(0.04, 1.0, 25)[2] == pytest.approx(0.04)
    assert latency_split(1.0, 1.0, 10)[:2] == (1, 0)
    assert latency_split(1.55, 1.0, 10)[:2] == (1, 6)
    assert sinr_to_cqi(np.array([-20.0, -6.7, 0.0, 22.7, 40.0])).tolist() == [0, 1, 3, 15, 15]


def test_observation_layout_and_neighbours():
    env = _env(share_latency_s=0.04)
    lay = env.obs_layout
    o = _rollout(env, 2)[-1]
    assert o.shape == (env.n_oru, lay.obs_dim) and env.observation_space.contains(o[0])
    assert lay.has_shared_copy and lay.max_neighbors == env.n_oru - 1
    L, c, M, K, F = lay.local_dim, lay.compact_dim, lay.max_neighbors, lay.max_ues, lay.ue_dim
    assert np.array_equal(o[:, c:c + M], env.nbr_idx) and (env.nbr_idx != np.arange(env.n_oru)[:, None]).all()
    own = o[:, :L]
    mask = own[:, K * F:K * F + K]
    ue = own[:, :K * F].reshape(-1, K, F)
    assert mask.sum() > 0 and np.all(ue[mask == 0] == 0)
    assert np.allclose(ue[mask == 1][:, 3:].sum(-1), 1)                       # sector one-hot
    assert np.all(np.diff(np.where(mask == 1, ue[..., 0], -1), axis=1) <= 1e-7)   # sorted by buffer
    # no sharing: just the local observation
    env0 = _env(obs_sharing=False)
    assert env0.obs_layout.max_neighbors == 0 and env0.obs_layout.obs_dim == env0.obs_layout.local_dim


def test_zero_latency_neighbours_see_fresh_observations():
    env = _env(share_latency_s=0.0)
    lay = env.obs_layout
    assert not lay.has_shared_copy
    L, c, M = lay.local_dim, lay.compact_dim, lay.max_neighbors
    o = _rollout(env, 3)[-1]
    nbr = o[:, c + M:].reshape(env.n_oru, M, L)
    assert np.array_equal(nbr, o[env.nbr_idx][:, :, :L])


def test_whole_step_latency_delivers_old_observations():
    env = _env(share_latency_s=2.0)            # exactly 2 decision steps
    assert (env.share_delay_steps, env.share_mini_offset) == (2, 0)
    lay = env.obs_layout
    L, c, M = lay.local_dim, lay.compact_dim, lay.max_neighbors
    seq = _rollout(env, 6)
    for t in range(2, 7):
        assert np.array_equal(seq[t][:, L:2 * L], seq[t - 2][:, :L])                # own shared = own 2 steps ago
        nbr = seq[t][:, c + M:].reshape(env.n_oru, M, L)
        assert np.array_equal(nbr, seq[t - 2][env.nbr_idx][:, :, :L])


def test_intra_step_latency_snapshots_inside_the_step():
    """Latency of one mini-period: the shared observation equals the local observation computed
    by an identical twin one mini-period before the end of the step."""
    a = _env(share_latency_s=0.2)              # n_mini = 5 -> dt_mini = 0.2 s
    assert (a.share_delay_steps, a.share_mini_offset) == (0, 1)
    b = _env(share_latency_s=0.0)
    snaps = []

    def hook(sim, i, acc):
        if i == 3:
            el = 4 * sim.dt_mini
            snaps.append(b.local_obs(acc["served"] / el, acc["served_sec"] / el, acc["energy"] / el,
                                     acc["busy"] / el))
    b.core.sim.mini_hooks.append(hook)
    L = a.obs_layout.local_dim
    sa, sb = _rollout(a, 4), _rollout(b, 4)
    for t in range(1, 5):
        assert np.array_equal(sa[t][:, :L], sb[t][:, :L])                          # same trajectory
        assert np.array_equal(sa[t][:, L:2 * L], snaps[-5 + t])
    assert not np.array_equal(sa[4][:, L:2 * L], sa[4][:, :L])


# ------------------------------------------------------------------ network
torch = pytest.importorskip("torch")
pytest.importorskip("ray.rllib")

import gymnasium as gym  # noqa: E402
import ray  # noqa: E402
from ray.rllib.core.columns import Columns  # noqa: E402
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode  # noqa: E402

from oran_asm_sim.baselines import run_policy  # noqa: E402
from oran_asm_sim.vdn import (TEAM_POLICY, VDN, DRQNPolicy, GraphDRQNModule, JointEpisodeReplayBuffer,  # noqa: E402
                              VDNConfig)

LAY = ObsLayout(max_ues=4, ue_dim=6, oru_dim=3, max_neighbors=2, has_shared_copy=True)
NBR = np.array([[1, 2], [0, -1], [3, 1], [-1, -1]])


def _module(lay=LAY, mp=True):
    cfg = dict(double_q=True, dueling=True, epsilon=0.0, fc_dim=8, rnn_dim=8, enc_dim=8, max_seq_len=6,
               obs_layout=lay.to_dict(), message_passing=mp)
    m = GraphDRQNModule(observation_space=gym.spaces.Box(-9, 9, (lay.obs_dim,), np.float32),
                        action_space=gym.spaces.MultiDiscrete([5] * 3), model_config=cfg)
    m.make_target_networks()
    return m


def _acting_obs(compact, nbr, lay=LAY):
    """compact [..., N, compact_dim] -> per-agent acting observations [..., N, obs_dim]."""
    L = lay.local_dim
    sh = compact[..., L:2 * L] if lay.has_shared_copy else compact[..., :L]
    blocks = sh[..., np.maximum(nbr, 0), :] * torch.as_tensor(nbr >= 0, dtype=torch.float32)[..., None]
    idx = torch.as_tensor(nbr, dtype=torch.float32).expand(*compact.shape[:-1], -1)
    return torch.cat([compact, idx, blocks.flatten(-2)], -1)


def _rand_compact(*shape, lay=LAY):
    x = torch.randn(*shape, lay.compact_dim)
    K, F = lay.max_ues, lay.ue_dim
    for off in (0, lay.local_dim) if lay.has_shared_copy else (0,):
        x[..., off + K * F:off + K * F + K] = (torch.rand(*shape, K) < 0.6).float()
    return x


def test_deepset_is_permutation_invariant_and_ignores_padding():
    m = _module()
    K, F = LAY.max_ues, LAY.ue_dim
    x = _rand_compact(5)[:, :LAY.local_dim]
    perm = torch.randperm(K)
    xp = x.clone()
    xp[:, :K * F] = x[:, :K * F].reshape(5, K, F)[:, perm].reshape(5, -1)
    xp[:, K * F:K * F + K] = x[:, K * F:K * F + K][:, perm]
    enc = m.net.enc
    assert torch.allclose(enc(x), enc(xp), atol=1e-6)
    xz = x.clone()
    pad = x[:, K * F:K * F + K] == 0
    xz[:, :K * F] = torch.where(pad.repeat_interleave(F, 1), torch.randn(5, K * F), x[:, :K * F])
    assert torch.allclose(enc(x), enc(xz), atol=1e-6)


@pytest.mark.parametrize("mp", [True, False])
def test_stepwise_acting_equals_joint_training_unroll(mp):
    m = _module(mp=mp)
    compact = _rand_compact(2, 6, 4)                                         # B, T+1, N
    out = m._forward_train({Columns.OBS: compact, "nbr_idx": torch.as_tensor(NBR).expand(2, -1, -1)})
    assert out["qf_preds"].shape == out["qf_target_next_preds"].shape == (2, 5, 4, 3, 5)
    acting = _acting_obs(compact, NBR)
    state = {k: v.expand(4, -1) for k, v in m.get_initial_state().items()}
    for t in range(5):
        o = m.compute_q_values({Columns.OBS: acting[1, t].unsqueeze(1), Columns.STATE_IN: state})
        state = o[Columns.STATE_OUT]
        assert torch.allclose(o["qf_preds"][:, 0], out["qf_preds"][1, t], atol=1e-5)


def test_message_passing_ablation_ignores_neighbours():
    obs = _acting_obs(_rand_compact(4), NBR)[:, None]                          # [N, 1, obs_dim]
    other = obs.clone()
    other[..., LAY.compact_dim + LAY.max_neighbors:] += 1.0                   # change neighbour blocks
    for mp, same in ((False, True), (True, False)):
        m = _module(mp=mp)
        q0, q1 = (m.compute_q_values({Columns.OBS: x})["qf_preds"] for x in (obs, other))
        assert torch.allclose(q0, q1) == same
    assert torch.allclose(q0[3], q1[3])                                       # agent 3 has no neighbours


def test_replay_compacts_and_returns_neighbour_index():
    N = 4
    ids = [f"oru_{i}" for i in range(N)]
    sp = gym.spaces.Box(-99, 99, (LAY.obs_dim,), np.float32)
    eps = MultiAgentEpisode(observation_space={a: sp for a in ids},
                            action_space={a: gym.spaces.MultiDiscrete([5] * 3) for a in ids})
    frames = [_rand_compact(N) for _ in range(11)]
    ob = [_acting_obs(f, NBR).numpy() for f in frames]
    eps.add_env_reset(observations={a: ob[0][i] for i, a in enumerate(ids)})
    for t in range(10):
        eps.add_env_step({a: ob[t + 1][i] for i, a in enumerate(ids)}, {a: np.array([t % 5] * 3) for a in ids},
                         {a: float(i) for i, a in enumerate(ids)}, truncateds={"__all__": t == 9})
    buf = JointEpisodeReplayBuffer(seed=0, obs_layout=LAY.to_dict())
    buf.add([eps])
    s = buf.sample(3, seq_len=4)
    assert s["obs"].shape == (3, 5, N, LAY.compact_dim)
    assert np.array_equal(s["nbr_idx"], np.broadcast_to(NBR, (3, N, 2)))
    assert np.allclose(s["agent_rewards"], np.arange(N)) and np.allclose(s["rewards"], 1.5)
    stacked = torch.stack(frames).numpy()
    assert any(np.allclose(s["obs"][0], stacked[t:t + 5]) for t in range(7))


@pytest.fixture(scope="module")
def ray_local():
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, object_store_memory=150_000_000,
             ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_iql_loss_matches_hand_computation(ray_local):
    from oran_asm_sim import ORanASMMultiAgentEnv
    cfg = (VDNConfig().environment(ORanASMMultiAgentEnv, env_config=SMALL).env_runners(num_env_runners=0)
           .training(gamma=0.9, td_error_loss_fn="mse", mixer="none"))
    algo = cfg.build_algo()
    learner = algo.learner_group._learner
    B, T, N, H, A = 2, 3, 4, 3, 5
    g = torch.Generator().manual_seed(0)
    q, qn, qt = (torch.rand(B, T, N, H, A, generator=g) for _ in range(3))
    batch = {"actions": torch.randint(A, (B, T, N, H), generator=g), "rewards": torch.zeros(B, T),
             "agent_rewards": torch.rand(B, T, N, generator=g),
             "terminateds": torch.zeros(B, T, dtype=torch.bool), "loss_mask": torch.ones(B, T, dtype=torch.bool)}
    batch["terminateds"][1, 2] = True
    batch["loss_mask"][0, 0] = False
    loss = learner.compute_loss_for_module(module_id=TEAM_POLICY, config=algo.config, batch=batch,
                                           fwd_out={"qf_preds": q, "qf_next_preds": qn, "qf_target_next_preds": qt})
    terms = []
    for b in range(B):
        for t in range(T):
            if not batch["loss_mask"][b, t]:
                continue
            for i in range(N):
                qi = sum(q[b, t, i, h, batch["actions"][b, t, i, h]] for h in range(H))
                nxt = sum(qt[b, t, i, h, qn[b, t, i, h].argmax()] for h in range(H))
                y = batch["agent_rewards"][b, t, i] + 0.9 * (0.0 if batch["terminateds"][b, t] else 1.0) * nxt
                terms.append((qi - y) ** 2)
    assert loss.item() == pytest.approx(torch.stack(terms).mean().item(), rel=1e-5)
    algo.stop()


@pytest.mark.parametrize("variant", [dict(), dict(message_passing=False), dict(mixer="none")])
def test_graph_variants_train_and_execute(ray_local, tmp_path, variant):
    env_cfg = {**SMALL, "share_latency_s": 0.2, "max_ues": 24}
    if variant.get("message_passing") is False:
        env_cfg["obs_sharing"] = False
    cfg = (VDNConfig().environment(ORanASMGraphEnv, env_config=env_cfg).env_runners(num_env_runners=0)
           .rl_module(model_config={"fc_dim": 16, "rnn_dim": 16, "max_seq_len": 8})
           .training(num_steps_sampled_before_learning_starts=16, target_network_update_freq=16, burn_in_len=2,
                     train_batch_size_per_learner=4, mixer=variant.get("mixer", "vdn"),
                     replay_buffer_config={"storage_dtype": "float16"})
           .graph(message_passing=variant.get("message_passing", True))
           .reporting(min_sample_timesteps_per_iteration=40, min_time_s_per_iteration=0))
    algo = cfg.build_algo()
    mod = algo.learner_group._learner.module[TEAM_POLICY]
    assert isinstance(mod, GraphDRQNModule) and mod.net.mp == variant.get("message_passing", True)
    assert algo.local_replay_buffer.layout is not None
    res = algo.train()
    assert np.isfinite(float(res["learners"][TEAM_POLICY]["qf_loss"]))
    path = algo.save_to_path(str(tmp_path / "ckpt"))
    algo.stop()
    algo2 = VDN.from_checkpoint(path)
    mod2 = algo2.learner_group._learner.module[TEAM_POLICY]
    assert all(torch.equal(v, mod2.state_dict()[k]) for k, v in mod.state_dict().items())
    algo2.stop()
    ma = ORanASMGraphEnv(env_cfg)
    assert len(run_policy(ma.core, DRQNPolicy(mod, ma), 3, seed=1)) == 3
