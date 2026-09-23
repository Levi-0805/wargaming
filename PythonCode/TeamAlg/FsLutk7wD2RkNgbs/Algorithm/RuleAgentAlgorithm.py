from __future__ import annotations

import math

from cssim.algorithms import RuleAlgorithm
from cssim.environment import TeamState
from cssim.protocol import Action


class ReinforceAgentAlgorithm(RuleAlgorithm):
    """地面部队沿右侧一路推进，无人机分三路侦察。

    占领点按“先右后左、由近及远”排成一条路线。步兵和机器狗以纵队中位为
    基准，拉得太开就等后面的单位，已经占领的点附近还有敌人时不继续往前。
    无人机按存活顺序分到三个据点上空，只给地面部队提供感知，不跟着地面追击。
    """

    _UAV = "BP_Base_UAV_C"
    # 山猫和运输直升机在本想定里不会位移，不能拿来当纵队队尾。
    _PACE_TYPES = frozenset({
        "BP_BaseSoldier_C",
        "BP_RoboDog_C",
        "BP_MNWS_Vehicle_Armored_C",
    })
    _PRIORITY = {
        "BP_BaseSoldier_C": 0,
        "BP_RoboDog_C": 1,
        "BP_Base_UAV_C": 2,
        "BP_MNWS_Vehicle_Armored_C": 3,
        "BP_MNWS_Vehicle_6x6UGV_C": 4,
        "BP_Helicopter_C": 5,
    }
    _LEASH = 10000.0
    _HOLD_RADIUS = 35000.0

    def __init__(self, context, enable_parent_assignment: bool = False):
        super().__init__(context)
        self.enable_parent_assignment = bool(enable_parent_assignment)
        self._march_uids: tuple[int, ...] | None = None

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

    @classmethod
    def _is_uav(cls, agent) -> bool:
        return agent.entity_type == cls._UAV

    @classmethod
    def _ground_units(cls, state: TeamState) -> list:
        return [agent for agent in state.agents if agent.alive and not cls._is_uav(agent)]

    @classmethod
    def _uavs(cls, state: TeamState) -> list:
        return [agent for agent in state.agents if agent.alive and cls._is_uav(agent)]

    @staticmethod
    def _median(units: list) -> tuple[float, float, float]:
        xs = sorted(float(unit.position[0]) for unit in units)
        ys = sorted(float(unit.position[1]) for unit in units)
        zs = [float(unit.position[2]) for unit in units]
        middle = len(xs) // 2
        return xs[middle], ys[middle], sum(zs) / float(len(zs))

    def _remember_march(self, state: TeamState) -> None:
        if self._march_uids:
            return
        objectives = [item for item in state.key_objects if item.valid]
        ground = self._ground_units(state)
        if not objectives or not ground:
            return
        ground_y = sum(float(unit.position[1]) for unit in ground) / float(len(ground))
        uavs = self._uavs(state)
        if uavs:
            uav_y = sum(float(unit.position[1]) for unit in uavs) / float(len(uavs))
            left_vector = uav_y - ground_y
        else:
            left_vector = 1.0
        if abs(left_vector) < 1.0:
            left_vector = 1.0
        origin = self._median(ground)

        def on_left(obj) -> bool:
            return (float(obj.position[1]) - ground_y) * left_vector > 0

        ordered = sorted(
            objectives,
            key=lambda item: (
                1 if on_left(item) else 0,
                self._horizontal(origin, item.position),
                int(item.uid),
            ),
        )
        self._march_uids = tuple(int(item.uid) for item in ordered)

    def _ordered_objectives(self, state: TeamState) -> list:
        self._remember_march(state)
        by_uid = {int(item.uid): item for item in state.key_objects if item.valid}
        if not self._march_uids:
            return list(by_uid.values())
        return [by_uid[uid] for uid in self._march_uids if uid in by_uid]

    def _threatened(self, state: TeamState, obj) -> bool:
        return any(
            enemy.alive and self._horizontal(enemy.position, obj.position) <= self._HOLD_RADIUS
            for enemy in state.visible_opponents
        )

    def _secured(self, state: TeamState, obj) -> bool:
        return self._held_by_red(obj) and not self._threatened(state, obj)

    def _current_objective(self, state: TeamState):
        ordered = self._ordered_objectives(state)
        if not ordered:
            return None
        for obj in ordered:
            if not self._secured(state, obj):
                return obj
        return ordered[-1]

    def _in_range(self, agent, enemy) -> bool:
        return self._horizontal(agent.position, enemy.position) <= self._fire_range(agent) * 0.95

    def _attack_target(self, agent, in_range: list):
        return min(
            in_range,
            key=lambda enemy: (
                self._priority(enemy),
                max(float(enemy.hp), 0.0),
                self._horizontal(agent.position, enemy.position),
                int(enemy.uid),
            ),
        )

    def _destination(self, agent, point, radius: float, height: float) -> tuple[float, float, float]:
        x, y, z = self._xyz(point)
        if radius > 0:
            angle = (int(agent.team_index) + 1) * 2.399963229728653
            x += radius * math.cos(angle)
            y += radius * math.sin(angle)
        return (x, y, z + height)

    def _pace_units(self, state: TeamState) -> list:
        mobile = [
            agent for agent in self._ground_units(state)
            if agent.entity_type in self._PACE_TYPES
        ]
        return mobile or self._ground_units(state)

    def _pace_point(self, units: list, objective) -> tuple[float, float, float]:
        """队尾前方一个缰绳长度处。全队只向这个点前进，不再退回营地。"""

        tail = max(units, key=lambda unit: self._horizontal(unit.position, objective.position))
        tail_point = self._xyz(tail.position)
        target = self._xyz(objective.position)
        distance = math.hypot(tail_point[0] - target[0], tail_point[1] - target[1])
        if distance <= self._LEASH:
            return target
        scale = self._LEASH / distance
        return (
            tail_point[0] + (target[0] - tail_point[0]) * scale,
            tail_point[1] + (target[1] - tail_point[1]) * scale,
            tail_point[2],
        )

    def _column_point(self, agent, units: list, objective):
        pace = self._pace_point(units, objective)
        pace_distance = self._horizontal(pace, objective.position)
        agent_distance = self._horizontal(agent.position, objective.position)
        if agent_distance + 500.0 < pace_distance:
            return self._xyz(agent.position)
        return pace

    def _scout_action(self, state: TeamState, agent) -> Action:
        perceived = [enemy for enemy in state.perceived_opponents(agent) if enemy.alive]
        in_range = [enemy for enemy in perceived if self._in_range(agent, enemy)]
        if in_range:
            return Action.attack(self._attack_target(agent, in_range))
        ordered = self._ordered_objectives(state)
        if not ordered:
            return Action.move("+X")
        uavs = self._uavs(state)
        slot = next((index for index, unit in enumerate(uavs) if unit.uid == agent.uid), 0)
        target = ordered[slot % len(ordered)]
        return Action.move_at(self._destination(agent, target.position, 1800.0, 800.0))

    def choose_action(self, state: TeamState, agent) -> Action:
        if not agent.alive:
            return Action.idle()
        if self._is_uav(agent):
            return self._scout_action(state, agent)

        perceived = [enemy for enemy in state.perceived_opponents(agent) if enemy.alive]
        in_range = [enemy for enemy in perceived if self._in_range(agent, enemy)]
        if in_range:
            return Action.attack(self._attack_target(agent, in_range))

        objective = self._current_objective(state)
        if objective is None:
            if perceived:
                nearest = min(
                    perceived,
                    key=lambda enemy: (self._horizontal(agent.position, enemy.position), int(enemy.uid)),
                )
                return Action.move_at(self._destination(agent, nearest.position, 0.0, 0.0))
            return Action.move("+X")

        ground = self._pace_units(state) or [agent]
        point = self._column_point(agent, ground, objective)
        if self._horizontal(agent.position, point) <= 1200.0:
            return Action.guard_position(self._destination(agent, point, 500.0, 0.0))
        return Action.move_at(self._destination(agent, point, 500.0, 0.0))

    def decide(self, state: TeamState):
        return [self.choose_action(state, agent) for agent in state.agents]


RuleAgentAlgorithm = ReinforceAgentAlgorithm
