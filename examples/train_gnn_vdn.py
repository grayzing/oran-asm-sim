"""Train the graph VDN model and its ablations on the per-O-RU ASM environment (Ray RLlib).

    python examples/train_gnn_vdn.py --variant gnn_vdn vdn_nomp gnn_iql --preset fast --iters 60
    python examples/train_gnn_vdn.py --variant gnn_vdn --latency-ms 40 --n-mini 25      # exact 40 ms
    python examples/train_gnn_vdn.py --variant gnn_vdn --eval-only

Model (per O-RU, parameters shared): DeepSet over the O-RU's UEs (DL buffer, CQI, throughput) +
O-RU energy efficiency / utilisation / state -> one GNN layer over the one-hop neighbours' shared
observations (stale by ``--latency-ms``, max-pooled) -> Lin -> LSTM -> one dueling Q-head per sector.

Variants:
    gnn_vdn   full model: message passing + value decomposition (Q_tot = sum_i Q_i, team reward)
    vdn_nomp  no observation sharing / message passing, value decomposition
    gnn_iql   message passing, independent Q-learning (per-agent TD error, no decomposition)
    iql_nomp  neither

Each variant writes <out>/<variant>/training.csv and a checkpoint; evaluation compares the greedy
policies with the non-learning baselines on held-out seeds.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oran_asm_sim.baselines import AlwaysOn, FixedASM, LoadThresholdSM4, run_policy  # noqa: E402
from oran_asm_sim.config import PRESETS  # noqa: E402
from oran_asm_sim.graph_env import ORanASMGraphEnv  # noqa: E402

VARIANTS = {  # name: (observation sharing + message passing, mixer)
    "gnn_vdn": (True, "vdn"),
    "vdn_nomp": (False, "vdn"),
    "gnn_iql": (True, "none"),
    "iql_nomp": (False, "none"),
}
EVAL_KEYS = ("reward", "power_norm", "qos_violation_frac", "outage_frac", "drop_frac")
LOG_KEYS = ("team_reward_mean", "power_norm", "qos_violation_frac", "outage_frac", "drop_frac")


def env_config(args, variant):
    cfg = {k: dict(v) for k, v in PRESETS[args.preset].items()}
    cfg.setdefault("env", {}).update(episode_steps=args.episode_steps, random_start=True, seed=args.seed)
    if args.n_mini:
        cfg["env"]["n_mini"] = args.n_mini
    cfg.update(obs_sharing=VARIANTS[variant][0], share_latency_s=args.latency_ms / 1000.0,
               max_ues=args.max_ues, max_neighbors=args.max_neighbors, neighbor_radius_m=args.neighbor_radius)
    return cfg


def train(args, variant, out, ckpt_dir):
    import ray
    from oran_asm_sim.vdn import TEAM_POLICY, VDNConfig

    sharing, mixer = VARIANTS[variant]
    ecfg = env_config(args, variant)
    probe = ORanASMGraphEnv(ecfg)
    lay = probe.obs_layout
    mb = args.buffer * probe.n_oru * lay.compact_dim * 2 / 1e6
    print(f"[{variant}] {probe.n_oru} O-RUs, obs_dim {lay.obs_dim}, neighbours {lay.max_neighbors}, "
          f"sharing staleness {probe.share_staleness_s * 1e3:.0f} ms (latency {args.latency_ms:g} ms, "
          f"mini-period {probe.core.sim.dt_mini * 1e3:.0f} ms); replay ~{mb:.0f} MB (float16)", flush=True)

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    cfg = (VDNConfig()
           .environment(ORanASMGraphEnv, env_config=ecfg)
           .env_runners(num_env_runners=args.runners)
           .rl_module(model_config={"enc_dim": args.hidden, "fc_dim": args.hidden, "rnn_dim": args.hidden,
                                    "max_seq_len": args.seq_len})
           .training(lr=args.lr, gamma=args.gamma, burn_in_len=args.burn_in, updates_per_round=args.updates,
                     train_batch_size_per_learner=args.batch, target_network_update_freq=args.target_freq,
                     epsilon=[(0, 1.0), (args.eps_steps, 0.05)], mixer=mixer,
                     replay_buffer_config={"type": "oran_asm_sim.vdn.replay.JointEpisodeReplayBuffer",
                                           "capacity": args.buffer, "seed": args.seed,
                                           "storage_dtype": "float16"})
           .graph(lay, message_passing=sharing)
           .debugging(seed=args.seed))
    algo = cfg.build_algo()
    rows = []
    try:
        for i in range(args.iters):
            res = algo.train()
            er, lr = res.get("env_runners", {}), res.get("learners", {}).get(TEAM_POLICY, {})
            row = {"iter": i, "env_steps": int(er.get("num_env_steps_sampled_lifetime", 0))}
            row.update({k: float(er.get(k, np.nan)) for k in LOG_KEYS})
            row.update({k: float(lr.get(k, np.nan)) for k in ("qf_loss", "qf_mean", "td_error_mean")})
            rows.append(row)
            print(f"[{variant}] " + "  ".join(f"{k}={v:.4g}" for k, v in row.items()), flush=True)
            with open(os.path.join(out, "training.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, row.keys())
                w.writeheader()
                w.writerows(rows)
    finally:
        algo.save_to_path(ckpt_dir)
        print("checkpoint:", ckpt_dir)
        algo.stop()
        ray.shutdown()


def evaluate(args, trained):
    from ray.rllib.core.rl_module.rl_module import RLModule
    from oran_asm_sim.vdn import TEAM_POLICY, DRQNPolicy

    base_env = ORanASMGraphEnv(env_config(args, "gnn_vdn"))
    entries = [(p, base_env) for p in (AlwaysOn(), FixedASM(1), FixedASM(2), FixedASM(3), LoadThresholdSM4())]
    for variant, ckpt in trained:
        module = RLModule.from_checkpoint(os.path.join(ckpt, "learner_group", "learner", "rl_module", TEAM_POLICY))
        ma = ORanASMGraphEnv(env_config(args, variant))
        pol = DRQNPolicy(module, ma)
        pol.name = variant
        entries.append((pol, ma))
    print(f"\nheld-out evaluation: hours {args.eval_hours}, {args.eval_steps} steps each")
    print(f"{'policy':24s}" + "".join(f"{k:>20s}" for k in EVAL_KEYS))
    summary = {}
    for p, ma in entries:
        rows = []
        for j, h in enumerate(args.eval_hours):
            getattr(p, "reset", lambda: None)()
            rows += run_policy(ma.core, p, args.eval_steps, seed=10_000_000 + j, start_hour=h)
        summary[p.name] = {k: float(np.mean([r[k] for r in rows])) for k in EVAL_KEYS}
        print(f"{p.name:24s}" + "".join(f"{summary[p.name][k]:20.4f}" for k in EVAL_KEYS), flush=True)
    with open(os.path.join(args.out, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="+", default=["gnn_vdn"], choices=list(VARIANTS))
    ap.add_argument("--preset", default="fast", choices=list(PRESETS))
    ap.add_argument("--iters", type=int, default=60, help="1 iteration = 1000 env steps")
    ap.add_argument("--runners", type=int, default=4)
    ap.add_argument("--episode-steps", type=int, default=300)
    # observation sharing
    ap.add_argument("--latency-ms", type=float, default=40.0, help="O-RU -> neighbour observation latency")
    ap.add_argument("--n-mini", type=int, default=None,
                    help="scheduling mini-periods per 1 s step (latency resolution = 1 s / n_mini)")
    ap.add_argument("--max-ues", type=int, default=48, help="padded UE-set size per O-RU")
    ap.add_argument("--max-neighbors", type=int, default=6)
    ap.add_argument("--neighbor-radius", type=float, default=None, help="metres; default: no limit")
    # learning
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
    ap.add_argument("--out", default="results/gnn_vdn")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--eval-hours", type=float, nargs="+", default=[3.0, 9.0, 13.0, 19.0])
    ap.add_argument("--eval-steps", type=int, default=120)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    trained = []
    for v in args.variant:
        out = os.path.join(args.out, v)
        ckpt = os.path.abspath(os.path.join(out, "checkpoint"))
        if not args.eval_only:
            os.makedirs(out, exist_ok=True)
            train(args, v, out, ckpt)
        trained.append((v, ckpt))
    evaluate(args, trained)


if __name__ == "__main__":
    main()
