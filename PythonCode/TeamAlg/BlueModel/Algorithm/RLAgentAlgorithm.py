from __future__ import annotations

from cssim.algorithms import RuleAlgorithm
from cssim.environment import TeamState
from cssim.protocol import Action


class AlgorithmConfig:
    """UE固定读取的默认蓝方配置。"""

    mode = "rule"


class ReinforceAgentAlgorithm(RuleAlgorithm):
    """未选择蓝方算法时使用的安全空闲策略。"""

    def decide(self, state: TeamState):
        """按战斗实体数量返回Idle，保证另一方算法可以正常运行。"""

        return [Action.idle() for _agent in state.agents]
