"""Reference (non-learning) policies. Each maps the env to a command vector in
{0..4} for the controllable sectors (``env.ctrl``; the HetNet macro is always on);
useful as RL baselines and sanity checks."""
from __future__ import annotations

import numpy as np


class AlwaysOn:
    name = "always-on (no ASM)"

    def __call__(self, env, obs=None):
        return np.zeros(env.S, int)


class FixedASM:
    def __init__(self, level: int):
        self.level = level
        self.name = f"opportunistic SM{level}"

    def __call__(self, env, obs=None):
        return np.full(env.S, self.level, int)


class LoadThresholdSM4:
    """Opportunistic SM2 everywhere; put a sector in SM4 when at most
    ``max_pot_ues`` UEs would pick it as best server with all cells on and its
    expected offered load is below ``thr_bps``; wake it otherwise. A crude
    coverage-unaware heuristic (UEs of an SM4 sector rely on neighbours)."""

    def __init__(self, thr_bps: float = 20e6, max_pot_ues: int = 4, asm_level: int = 2):
        self.thr, self.k, self.lvl = thr_bps, max_pot_ues, asm_level
        self.name = f"threshold SM4 + SM{asm_level}"

    def __call__(self, env, obs=None):
        sim = env.sim
        best = sim.best_sector_if_all_on()
        pot = np.bincount(best, minlength=sim.S)[env.ctrl]
        lam = sim.traffic.mean_offered_bps(sim.t, sim.period_multiplier())
        pot_load = np.bincount(best, weights=lam, minlength=sim.S)[env.ctrl]
        off = (pot <= self.k) & (pot_load < self.thr)
        return np.where(off, 4, self.lvl).astype(int)


def run_policy(env, policy, n_steps: int, seed=None, start_hour=None):
    obs, info = env.reset(seed=seed, options={"start_hour": start_hour} if start_hour is not None else None)
    rows = []
    for _ in range(n_steps):
        a = policy(env, obs)
        # map command levels to action indices if action_set is a subset
        lv = list(env.action_levels)
        a_idx = np.array([lv.index(int(x)) if int(x) in lv else 0 for x in a])
        obs, r, term, trunc, info = env.step(a_idx)
        info = dict(info)
        info["reward"] = r
        rows.append(info)
        if term or trunc:
            break
    return rows
