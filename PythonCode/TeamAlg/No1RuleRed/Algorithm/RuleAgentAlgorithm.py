from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cssim.algorithms import RuleAlgorithm
from cssim.environment import TeamState
from cssim.protocol import Action, ChangeParentAction


@dataclass(frozen=True, order=True)
class TargetScore:
    """用于稳定排序的目标评分；分数越小越优先。"""

    distance: float
    hp: float
    priority: int
    uid: int


class ReinforceAgentAlgorithm(RuleAlgorithm):
    """No.1 红方规则：接敌优先，未接敌时分组夺取蓝方战略目标。"""

    defending_team = 1

    def __init__(
        self,
        context,
        guard_radius: float = 900.0,
        enable_parent_assignment: bool = False,
    ):
        super().__init__(context)
        if guard_radius < 0:
            raise ValueError("guard_radius 不能为负数")
        self.guard_radius = float(guard_radius)
        self.enable_parent_assignment = bool(enable_parent_assignment)

    @staticmethod
    def _target_priority(entity) -> int:
        """让高价值/指挥类目标在同等距离下优先受击。"""

        name = f"{entity.entity_type} {entity.class_name}".lower()
        if any(token in name for token in ("commander", "command", "hq", "base")):
            return 0
        if any(token in name for token in ("uav", "helicopter", "air")):
            return 1
        return 2

    @classmethod
    def nearest_best_enemy(cls, state: TeamState, agent):
        visible = state.perceived_opponents(agent)
        return min(
            visible,
            key=lambda enemy: TargetScore(
                distance=float(np.linalg.norm(agent.position - enemy.position)),
                hp=max(float(enemy.hp), 0.0),
                priority=cls._target_priority(enemy),
                uid=int(enemy.uid),
            ),
            default=None,
        )

    def assigned_objective(self, state: TeamState, agent):
        objectives = tuple(sorted(
            state.key_objects_for_team(self.defending_team), key=lambda item: item.uid
        ))
        if not objectives:
            return None
        return objectives[agent.team_index % len(objectives)]

    def choose_action(self, state: TeamState, agent) -> Action | ChangeParentAction:
        if not agent.alive:
            return Action.idle()

        if self.enable_parent_assignment and agent.entity_type != "BP_BaseSoldier_C":
            parent = state.nearest_available_parent(agent)
            if parent is not None and agent.commander_uid() != parent.uid:
                return ChangeParentAction(parent, agent)

        enemy = self.nearest_best_enemy(state, agent)
        if enemy is not None:
            return Action.attack(enemy)

        objective = self.assigned_objective(state, agent)
        if objective is None:
            return Action.move("+X")

        distance = float(np.linalg.norm(agent.position - objective.position))
        if distance <= self.guard_radius:
            return Action.guard_position(objective.position)
        return Action.move_at(objective.position)

    def decide(self, state: TeamState):
        """严格按 state.agents 顺序返回一条动作。"""

        return [self.choose_action(state, agent) for agent in state.agents]


RuleAgentAlgorithm = ReinforceAgentAlgorithm
