"""Value-Decomposition Networks (Sunehag et al., 2018) with a recurrent deep Q-network
for :class:`~oran_asm_sim.env.ORanASMMultiAgentEnv`, on RLlib's new API stack.

Q_tot(tau, a) = sum_i Q_i(tau_i, a_i): every O-RU acts greedily on its own recurrent Q_i
(decentralised execution); only the sum is trained, on the team reward (centralised training).
An O-RU's Q_i is the sum of one Q-head per sector it controls.

Graph variant (:class:`~oran_asm_sim.graph_env.ORanASMGraphEnv`, ``VDNConfig().graph()``):
DeepSet encoding of each O-RU's UE set, one round of message passing over neighbouring O-RUs'
(latency-delayed) shared observations, then the recurrent Q-network. Ablations:
``graph(message_passing=False)`` and ``training(mixer="none")`` (independent Q-learning).
"""
from .algorithm import TEAM_POLICY, VDN, ASMKpiCallback, VDNConfig
from .learner import VDNTorchLearner
from .gnn import DeepSetEncoder, GraphDRQNModule, GraphRecurrentQNet
from .module import DRQNModule, RecurrentQNet
from .policy import DRQNPolicy
from .replay import JointEpisodeReplayBuffer

__all__ = ["VDN", "VDNConfig", "VDNTorchLearner", "DRQNModule", "DRQNPolicy", "RecurrentQNet",
           "GraphDRQNModule", "GraphRecurrentQNet", "DeepSetEncoder",
           "JointEpisodeReplayBuffer", "ASMKpiCallback", "TEAM_POLICY"]
