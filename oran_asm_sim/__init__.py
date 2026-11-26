"""oran_asm_sim - vectorised 3GPP-style system-level simulator for per-sector
O-RU Advanced Sleep Mode (ASM) control, packaged as Gymnasium / RLlib envs."""
from .config import (EnvConfig, GeometryConfig, HetNetConfig, MobilityConfig, PeriodConfig, PowerConfig,
                     RadioConfig, SimConfig, TierConfig, TrafficConfig)
from .env import ORanASMEnv, ORanASMMultiAgentEnv
from .power import ASM_TIMING, ORUPowerModel
from .simulator import ORANSystemSimulator

__all__ = ["SimConfig", "GeometryConfig", "RadioConfig", "MobilityConfig", "PeriodConfig",
           "TrafficConfig", "PowerConfig", "EnvConfig", "HetNetConfig", "TierConfig", "ORANSystemSimulator", "ORanASMEnv",
           "ORanASMMultiAgentEnv", "ORUPowerModel", "ASM_TIMING", "register"]
__version__ = "0.2.0"


def register():
    """Register with Gymnasium (``gym.make("ORanASM-v0")``) and RLlib's registry."""
    try:
        import gymnasium as gym
        if "ORanASM-v0" not in gym.envs.registry:
            gym.register(id="ORanASM-v0", entry_point="oran_asm_sim.env:ORanASMEnv")
    except ImportError:
        pass
    try:
        from ray.tune.registry import register_env
        register_env("oran_asm", lambda cfg: ORanASMEnv(cfg))
        register_env("oran_asm_ma", lambda cfg: ORanASMMultiAgentEnv(cfg))
    except ImportError:
        pass
