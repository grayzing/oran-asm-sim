"""Gymnasium / RLlib import shims.

The simulator itself only needs numpy (+ scipy optional). If ``gymnasium`` is
installed (required for RLlib) its ``Env`` and ``spaces`` are used; otherwise
minimal stand-ins with the same interface are provided so the environment can
still be stepped and unit-tested."""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised when gymnasium is installed
    import gymnasium as gym
    from gymnasium import spaces
    HAVE_GYM = True
    Env = gym.Env
except Exception:  # minimal fallback
    HAVE_GYM = False

    class _Space:
        def seed(self, seed=None):
            self._rng = np.random.default_rng(seed)

        @property
        def np_random(self):
            if not hasattr(self, "_rng"):
                self._rng = np.random.default_rng()
            return self._rng

    class Box(_Space):
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.shape = tuple(shape) if shape is not None else np.shape(low)
            self.dtype = np.dtype(dtype)
            self.low = np.broadcast_to(np.asarray(low, self.dtype), self.shape)
            self.high = np.broadcast_to(np.asarray(high, self.dtype), self.shape)

        def sample(self):
            lo = np.where(np.isfinite(self.low), self.low, -1.0)
            hi = np.where(np.isfinite(self.high), self.high, 1.0)
            return self.np_random.uniform(lo, hi).astype(self.dtype)

        def contains(self, x):
            x = np.asarray(x)
            return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)

    class MultiDiscrete(_Space):
        def __init__(self, nvec, dtype=np.int64):
            self.nvec = np.asarray(nvec, np.int64)
            self.shape = self.nvec.shape
            self.dtype = np.dtype(dtype)

        def sample(self):
            return (self.np_random.random(self.nvec.shape) * self.nvec).astype(self.dtype)

        def contains(self, x):
            x = np.asarray(x)
            return x.shape == self.shape and np.all((x >= 0) & (x < self.nvec))

    class Discrete(_Space):
        def __init__(self, n):
            self.n = int(n)
            self.shape = ()
            self.dtype = np.dtype(np.int64)

        def sample(self):
            return int(self.np_random.integers(self.n))

        def contains(self, x):
            return 0 <= int(x) < self.n

    class Dict(_Space):
        def __init__(self, d):
            self.spaces = dict(d)

        def __getitem__(self, k):
            return self.spaces[k]

        def sample(self):
            return {k: v.sample() for k, v in self.spaces.items()}

        def contains(self, x):
            return all(self.spaces[k].contains(x[k]) for k in self.spaces)

    class _Spaces:
        pass

    spaces = _Spaces()
    spaces.Box, spaces.MultiDiscrete, spaces.Discrete, spaces.Dict = Box, MultiDiscrete, Discrete, Dict

    class Env:
        metadata = {}
        observation_space = None
        action_space = None

        def reset(self, *, seed=None, options=None):
            raise NotImplementedError

        def step(self, action):
            raise NotImplementedError

        def close(self):
            pass

        @property
        def unwrapped(self):
            return self

try:  # pragma: no cover
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    HAVE_RAY = True
except Exception:
    HAVE_RAY = False

    class MultiAgentEnv(Env):  # type: ignore[no-redef]
        def __init__(self):
            super().__init__()
