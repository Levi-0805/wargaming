from __future__ import annotations

import math

from cssim.algorithms import RuleAlgorithm
from cssim.environment import TeamState
from cssim.protocol import Action, ChangeParentAction


class MultiObjectivePlan:
    """按Python的team_index动作行号，把兵力稳定分配给多个目标。"""

    @staticmethod
    def objectives(state: TeamState, defending_team: int):
        return tuple(sorted(
            state.key_objects_for_team(defending_team), key=lambda item: item.uid
        ))

    @classmethod
    def assignment(cls, state: TeamState, agent, defending_team: int):
        objectives = cls.objectives(state, defending_team)
        if not objectives:
            return None, 0, 1
        objective_index = agent.team_index % len(objectives)
        assigned_agents = tuple(
            item for item in state.agents
            if item.team_index % len(objectives) == objective_index
        )
        slot = next(
            index for index, item in enumerate(assigned_agents) if item.uid == agent.uid
        )
        return objectives[objective_index], slot, len(assigned_agents)


class ReinforceAgentAlgorithm(RuleAlgorithm):
    """红方规则模板：修改 decide 即可实现自己的进攻策略。"""

    defending_team = 1
    def __init__(self, context, enable_parent_assignment: bool = False):
        super().__init__(context)
        self.enable_parent_assignment = bool(enable_parent_assignment)

    @staticmethod
    def nearest_enemy(state: TeamState, agent):
        """只从当前Agent的局部感知中选择最近敌方。"""

        visible = state.perceived_opponents(agent)
        return min(
            visible,
            key=lambda enemy: float(sum((agent.position - enemy.position) ** 2)),
            default=None,
        )

    def formation_point(self, state: TeamState, agent):
        """把本方实体均匀分配到全部蓝方指挥所周围。"""

        objective, slot, assigned_count = MultiObjectivePlan.assignment(
            state, agent, self.defending_team
        )
        if objective is None:
            return None
        angle = 2.0 * math.pi * slot / assigned_count
        height = 1800.0 if agent.entity_type == "BP_Base_UAV_C" else 0.0
        return (
            float(objective.position[0]) + 350.0 * math.cos(angle),
            float(objective.position[1]) + 350.0 * math.sin(angle),
            float(objective.position[2]) + height,
        )

    def choose_action(self, state: TeamState, agent) -> Action | ChangeParentAction:
        """每个实体只选一个动作；指派上级同样占用本帧动作槽。"""

        if not agent.alive:
            return Action.idle()
        # 案例只自动调整无人装备，避免士兵之间按距离形成循环指挥关系。
        if self.enable_parent_assignment and agent.entity_type != "BP_BaseSoldier_C":
            parent = state.nearest_available_parent(agent)
            if parent is not None and agent.commander_uid() != parent.uid:
                return ChangeParentAction(parent, agent)
        target = self.nearest_enemy(state, agent)
        if target is not None:
            return Action.attack(target)
        point = self.formation_point(state, agent)
        return Action.move_to(point) if point is not None else Action.move("+X")

    def decide(self, state: TeamState):
        """必须严格按照state.agents顺序为每个实体返回一个动作。"""

        return [self.choose_action(state, agent) for agent in state.agents]


RuleAgentAlgorithm = ReinforceAgentAlgorithm
