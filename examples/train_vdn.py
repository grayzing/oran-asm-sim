"""Train VDN with a recurrent deep Q-network (DRQN) on the per-O-RU ASM environment (Ray RLlib).

    pip install "ray[rllib]>=2.60" torch
    python examples/train_vdn.py --preset fast --iters 60 --runners 4
    python examples/train_vdn.py --eval-only --checkpoint results/vdn/checkpoint

One agent per O-RU (one per site), MultiDiscrete([5] * 3) sleep commands for its three sectors. All
O-RUs share one LSTM Q-network with one Q-head per sector; the team TD error is taken on
Q_tot = sum_i Q_i, with Q_i the sum of O-RU i's sector heads. After training, the greedy policy is compared with the
non-learning baselines on held-out seeds.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oran_asm_sim import ORanASMMultiAgentEnv  # noqa: E402
from oran_asm_sim.baselines import AlwaysOn, FixedASM, LoadThresholdSM4, run_policy  # noqa: E402
from oran_asm_sim.config import PRESETS  # noqa: E402

EVAL_KEYS = ("reward", "power_norm", "qos_violation_frac", "outage_frac", "drop_frac")


def env_config(args):
    cfg = {k: dict(v) for k, v in PRESETS[args.preset].items()}
    cfg.setdefault("env", {}).update(episode_steps=args.episode_steps, random_start=True, seed=args.seed)
    return cfg                                   # local_reward_mix = 0: pure team reward


def train(args, ckpt_dir):
    import ray
    from oran_asm_sim.vdn import TEAM_POLICY, VDNConfig

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    cfg = (VDNConfig()
           .environment(ORanASMMultiAgentEnv, env_config=env_config(args))
           .env_runners(num_env_runners=args.runners)
           .rl_module(model_config={"fc_dim": args.hidden, "rnn_dim": args.hidden, "max_seq_len": args.seq_len})
           .training(lr=args.lr, gamma=args.gamma, burn_in_len=args.burn_in, updates_per_round=args.updates,
                     train_batch_size_per_learner=args.batch, target_network_update_freq=args.target_freq,
                     epsilon=[(0, 1.0), (args.eps_steps, 0.05)],
                     replay_buffer_config={"type": "oran_asm_sim.vdn.replay.JointEpisodeReplayBuffer",
                                           "capacity": args.buffer, "seed": args.seed})
           .debugging(seed=args.seed))
    algo = cfg.build_algo()
    rows = []
    try:
        for i in range(args.iters):
            res = algo.train()
            er, lr = res.get("env_runners", {}), res.get("learners", {}).get(TEAM_POLICY, {})
            row = {"iter": i, "env_steps": int(er.get("num_env_steps_sampled_lifetime", 0))}
            for k in ("team_reward_mean", "power_norm", "qos_violation_frac", "outage_frac", "drop_frac"):
                row[k] = float(er.get(k, np.nan))
            for k in ("qf_loss", "qf_mean", "td_error_mean"):
                row[k] = float(lr.get(k, np.nan))
            rows.append(row)
            print("  ".join(f"{k}={v:.4g}" for k, v in row.items()), flush=True)
            with open(os.path.join(args.out, "training.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, row.keys())
                w.writeheader()
                w.writerows(rows)
    finally:
        algo.save_to_path(ckpt_dir)
        print("checkpoint:", ckpt_dir)
        algo.stop()
        ray.shutdown()


def evaluate(args, ckpt_dir):
    from ray.rllib.core.rl_module.rl_module import RLModule
    from oran_asm_sim.vdn import TEAM_POLICY, DRQNPolicy

    module = RLModule.from_checkpoint(os.path.join(ckpt_dir, "learner_group", "learner", "rl_module", TEAM_POLICY))
    ma = ORanASMMultiAgentEnv(env_config(args))
    policies = [AlwaysOn(), FixedASM(1), FixedASM(2), FixedASM(3), LoadThresholdSM4(), DRQNPolicy(module, ma)]
    print(f"\nheld-out evaluation: hours {args.eval_hours}, {args.eval_steps} steps each")
    print(f"{'policy':24s}" + "".join(f"{k:>20s}" for k in EVAL_KEYS))
    for p in policies:
        rows = []
        for j, h in enumerate(args.eval_hours):
            getattr(p, "reset", lambda: None)()
            rows += run_policy(ma.core, p, args.eval_steps, seed=10_000_000 + j, start_hour=h)
        print(f"{p.name:24s}" + "".join(f"{np.mean([r[k] for r in rows]):20.4f}" for k in EVAL_KEYS), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fast", choices=list(PRESETS))
    ap.add_argument("--iters", type=int, default=60, help="1 iteration = 1000 env steps")
    ap.add_argument("--runners", type=int, default=4)
    ap.add_argument("--episode-steps", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=24, help="replay sequence length incl. burn-in")
    ap.add_argument("--burn-in", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32, help="sequences per gradient step")
    ap.add_argument("--updates", type=int, default=4, help="gradient steps per sampling round")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--target-freq", type=int, default=500, help="env steps between target-net syncs")
    ap.add_argument("--eps-steps", type=int, default=20_000, help="env steps to anneal epsilon 1 -> 0.05")
    ap.add_argument("--buffer", type=int, default=50_000, help="replay capacity in env steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/vdn")
    ap.add_argument("--checkpoint", default=None, help="default: <out>/checkpoint")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--eval-hours", type=float, nargs="+", default=[3.0, 9.0, 13.0, 19.0])
    ap.add_argument("--eval-steps", type=int, default=120)
    args = ap.parse_args()
    ckpt = os.path.abspath(args.checkpoint or os.path.join(args.out, "checkpoint"))
    if not args.eval_only:
        os.makedirs(args.out, exist_ok=True)
        train(args, ckpt)
    evaluate(args, ckpt)


if __name__ == "__main__":
    main()
