from __future__ import annotations

import math

from cssim.algorithms import RuleAlgorithm
from cssim.environment import TeamState
from cssim.protocol import Action


class ReinforceAgentAlgorithm(RuleAlgorithm):
    """红方集中进攻：只在武器射程内开火，全队打同一个弱点和同一高价值目标。

    第一局回放红方击毁得分为 220（士兵 100 分、机器狗 20 分）。当时全队按
    占领点数量拆开，并且一进入感知范围就攻击，人还停在约 1000cm 的火力射程外。
    """

    _PRIORITY = {
        "BP_BaseSoldier_C": 0,
        "BP_RoboDog_C": 1,
        "BP_Base_UAV_C": 2,
        "BP_MNWS_Vehicle_Armored_C": 3,
        "BP_MNWS_Vehicle_6x6UGV_C": 4,
        "BP_Helicopter_C": 5,
    }

    def __init__(self, context, enable_parent_assignment: bool = False):
        super().__init__(context)
        self.enable_parent_assignment = bool(enable_parent_assignment)

    @classmethod
    def _priority(cls, enemy) -> int:
        return cls._PRIORITY.get(enemy.entity_type, 6)

    @staticmethod
    def _xyz(point) -> tuple[float, float, float]:
        return (float(point[0]), float(point[1]), float(point[2]))

    @classmethod
    def _horizontal(cls, origin, point) -> float:
        left = cls._xyz(origin)
        right = cls._xyz(point)
        return math.hypot(left[0] - right[0], left[1] - right[1])

    @staticmethod
    def _fire_range(agent) -> float:
        merged = getattr(agent, "properties", {}) or {}
        raw = getattr(agent, "raw", {}) or {}
        value = merged.get("fireRange", raw.get("fireRange", 1000.0))
        try:
            distance = float(value)
        except (TypeError, ValueError):
            distance = 1000.0
        if not math.isfinite(distance) or distance <= 0:
            return 1000.0
        return distance

    @staticmethod
    def _count(obj, name: str) -> float:
        try:
            return float((obj.raw or {}).get(name, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _held_by_red(cls, obj) -> bool:
        return (
            cls._count(obj, "blueteamNum") <= 0
            and cls._count(obj, "redTeamNum") > 0
            and cls._count(obj, "percent") >= 0.99
        )

    @staticmethod
    def _centroid(state: TeamState) -> tuple[float, float, float]:
        alive = [agent for agent in state.agents if agent.alive]
        group = alive or list(state.agents)
        count = float(len(group))
        return (
            sum(float(agent.position[0]) for agent in group) / count,
            sum(float(agent.position[1]) for agent in group) / count,
            sum(float(agent.position[2]) for agent in group) / count,
        )

    def _objective(self, state: TeamState):
        objectives = [item for item in state.key_objects if item.valid]
        if not objectives:
            return None
        center = self._centroid(state)
        return min(
            objectives,
            key=lambda item: (
                1 if self._held_by_red(item) else 0,
                self._count(item, "blueteamNum"),
                self._horizontal(center, item.position),
                int(item.uid),
            ),
        )

    def _team_target(self, state: TeamState):
        enemies = [enemy for enemy in state.visible_opponents if enemy.alive]
        if not enemies:
            return None
        center = self._centroid(state)
        return min(
            enemies,
            key=lambda enemy: (
                self._priority(enemy),
                self._horizontal(center, enemy.position),
                max(float(enemy.hp), 0.0),
                int(enemy.uid),
            ),
        )

    def _in_range(self, agent, enemy) -> bool:
        return self._horizontal(agent.position, enemy.position) <= self._fire_range(agent) * 0.95

    def _destination(self, agent, point, radius: float) -> tuple[float, float, float]:
        x, y, z = self._xyz(point)
        if radius > 0:
            angle = (int(agent.team_index) + 1) * 2.399963229728653
            x += radius * math.cos(angle)
            y += radius * math.sin(angle)
        if agent.entity_type == "BP_Base_UAV_C":
            z += 300.0
        return (x, y, z)

    def _attack_target(self, state: TeamState, agent, in_range: list):
        team_target = self._team_target(state)
        if team_target is not None and any(enemy.uid == team_target.uid for enemy in in_range):
            return team_target
        return min(
            in_range,
            key=lambda enemy: (
                self._priority(enemy),
                self._horizontal(agent.position, enemy.position),
                max(float(enemy.hp), 0.0),
                int(enemy.uid),
            ),
        )

    def choose_action(self, state: TeamState, agent) -> Action:
        if not agent.alive:
            return Action.idle()

        perceived = [enemy for enemy in state.perceived_opponents(agent) if enemy.alive]
        in_range = [enemy for enemy in perceived if self._in_range(agent, enemy)]
        if in_range:
            return Action.attack(self._attack_target(state, agent, in_range))

        team_target = self._team_target(state)
        if team_target is not None:
            return Action.move_at(self._destination(agent, team_target.position, 0.0))

        objective = self._objective(state)
        if objective is None:
            return Action.move("+X")
        if self._horizontal(agent.position, objective.position) <= 1200.0:
            return Action.guard_position(self._destination(agent, objective.position, 800.0))
        return Action.move_at(self._destination(agent, objective.position, 800.0))

    def decide(self, state: TeamState):
        return [self.choose_action(state, agent) for agent in state.agents]


RuleAgentAlgorithm = ReinforceAgentAlgorithm
