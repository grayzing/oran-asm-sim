"""Train a per-sector ASM controller with Ray RLlib (PPO, new API stack).

    pip install "ray[rllib]>=2.40" gymnasium torch
    python examples/train_rllib.py --mode single --preset fast --iters 50
    python examples/train_rllib.py --mode multi  --preset dtla_19site --iters 200

single : one centralised agent, MultiDiscrete([5] * n_sectors) action
multi  : one agent per O-RU (one per site), MultiDiscrete([5] * 3) action (one command per
         sector it controls), one shared policy (parameter sharing)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oran_asm_sim import ORanASMEnv, ORanASMMultiAgentEnv  # noqa: E402
from oran_asm_sim.config import PRESETS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "multi"], default="single")
    ap.add_argument("--preset", default="fast", choices=list(PRESETS))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--runners", type=int, default=4)
    ap.add_argument("--episode-steps", type=int, default=300)
    ap.add_argument("--obs", choices=["pooled", "set"], default="pooled",
                    help="'set' needs a custom DeepSets/attention RLModule")
    args = ap.parse_args()

    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    env_config = {k: dict(v) for k, v in PRESETS[args.preset].items()}
    env_config.setdefault("env", {}).update(episode_steps=args.episode_steps, random_start=True, seed=0)

    ray.init(ignore_reinit_error=True)
    cfg = (PPOConfig()
           .env_runners(num_env_runners=args.runners, num_envs_per_env_runner=1,
                        rollout_fragment_length="auto")
           .training(lr=3e-4, gamma=0.99, lambda_=0.95, train_batch_size_per_learner=4000,
                     minibatch_size=256, num_epochs=10, entropy_coeff=0.01))
    if args.mode == "single":
        cfg = cfg.environment(ORanASMEnv, env_config=env_config)
    else:
        env_config["local_reward_mix"] = 0.2
        env_config["ma_obs_mode"] = args.obs
        cfg = (cfg.environment(ORanASMMultiAgentEnv, env_config=env_config)
                  .multi_agent(policies={"oru_policy"},
                               policy_mapping_fn=lambda agent_id, episode, **kw: "oru_policy"))

    algo = cfg.build_algo() if hasattr(cfg, "build_algo") else cfg.build()
    for i in range(args.iters):
        res = algo.train()
        er = res.get("env_runners", {})
        print(f"iter {i:4d}  return_mean={er.get('episode_return_mean', float('nan')):9.3f}  "
              f"len={er.get('episode_len_mean', float('nan'))}")
    path = algo.save()
    print("checkpoint:", path)


if __name__ == "__main__":
    main()
