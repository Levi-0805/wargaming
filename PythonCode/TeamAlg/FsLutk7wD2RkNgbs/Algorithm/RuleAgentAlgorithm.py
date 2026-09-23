from __future__ import annotations

import math

from cssim.algorithms import RuleAlgorithm
from cssim.environment import TeamState
from cssim.protocol import Action


class ReinforceAgentAlgorithm(RuleAlgorithm):
    """全体从南侧一条路压向蓝方所在据点，先头等队伍收拢后再进点。

    刚打的三局里，前后差距不到 50 米时等候没有生效，整队贴进指挥所被打掉，
    得分 500 和 200。差距大约 70 米的那局从南侧一起进点，击毁 2340 再加占领
    1600，合计 3940，结束时点里还剩 2 个蓝方。现在超前 25 米就停，进到 80 米
    内改冲据点中心，把剩下的人清掉。
    """

    _SOLDIER = "BP_BaseSoldier_C"
    _DOG = "BP_RoboDog_C"
    _UAV = "BP_Base_UAV_C"
    _GROUPS_PER_TEAM = 5
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
        self._road_uids: tuple[int, ...] | None = None

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

    def _alive(self, state: TeamState, entity_type: str | None = None) -> list:
        units = [agent for agent in state.agents if agent.alive]
        if entity_type is not None:
            units = [agent for agent in units if agent.entity_type == entity_type]
        return sorted(units, key=lambda agent: (int(agent.team_index), int(agent.uid)))

    def _groups(self, state: TeamState) -> list[list]:
        """每条机器狗配一名步兵；多出来的步兵单独成组。"""

        soldiers = self._alive(state, self._SOLDIER)
        dogs = self._alive(state, self._DOG)
        groups: list[list] = []
        for index, dog in enumerate(dogs):
            members = [dog]
            if index < len(soldiers):
                members.append(soldiers[index])
            groups.append(members)
        groups.extend([soldier] for soldier in soldiers[len(dogs):])
        return groups

    def _membership(self, state: TeamState, agent) -> tuple[int, int, int] | None:
        """返回 (队伍号, 队内编组号, 编组内序号)。不属于步兵/机器狗时返回 None。"""

        for group_index, members in enumerate(self._groups(state)):
            for pair_index, member in enumerate(members):
                if member.uid == agent.uid:
                    team_id = group_index // self._GROUPS_PER_TEAM
                    slot = group_index % self._GROUPS_PER_TEAM
                    return team_id, slot, pair_index
        return None

    def _remember_roads(self, state: TeamState) -> None:
        if self._road_uids:
            return
        objectives = [item for item in state.key_objects if item.valid]
        if not objectives:
            return
        ground = self._alive(state, self._SOLDIER) or [
            agent for agent in state.agents if agent.alive and agent.entity_type != self._UAV
        ]
        if not ground:
            self._road_uids = tuple(int(item.uid) for item in objectives)
            return
        ground_y = sum(float(unit.position[1]) for unit in ground) / float(len(ground))
        uavs = self._alive(state, self._UAV)
        if uavs:
            uav_y = sum(float(unit.position[1]) for unit in uavs) / float(len(uavs))
            left_vector = uav_y - ground_y
        else:
            left_vector = 1.0
        if abs(left_vector) < 1.0:
            left_vector = 1.0
        ordered = sorted(
            objectives,
            key=lambda item: (
                (float(item.position[1]) - ground_y) * left_vector,
                int(item.uid),
            ),
        )
        self._road_uids = tuple(int(item.uid) for item in ordered)

    def _roads(self, state: TeamState) -> list:
        self._remember_roads(state)
        by_uid = {int(item.uid): item for item in state.key_objects if item.valid}
        if not self._road_uids:
            return list(by_uid.values())
        return [by_uid[uid] for uid in self._road_uids if uid in by_uid]

    @staticmethod
    def _count(obj, name: str) -> float:
        try:
            return float((obj.raw or {}).get(name, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    def _battle_objective(self, state: TeamState):
        """蓝方人堆在哪个据点，队伍就攻哪个据点。"""

        roads = self._roads(state)
        if not roads:
            return None
        ground = self._alive(state, self._SOLDIER) or self._alive(state)
        if ground:
            origin = self._xyz(ground[0].position)
        else:
            origin = (0.0, 0.0, 0.0)
        return max(
            roads,
            key=lambda item: (
                self._count(item, "blueteamNum"),
                -self._horizontal(origin, item.position),
                -int(item.uid),
            ),
        )

    def _in_range(self, agent, enemy) -> bool:
        return self._horizontal(agent.position, enemy.position) <= self._fire_range(agent) * 0.95

    def _attack_target(self, agent, enemies: list):
        return min(
            enemies,
            key=lambda enemy: (
                self._priority(enemy),
                max(float(enemy.hp), 0.0),
                self._horizontal(agent.position, enemy.position),
                int(enemy.uid),
            ),
        )

    def _spread_point(self, objective, team_id: int, slot: int, pair_index: int, agent=None) -> tuple[float, float, float]:
        del team_id
        x, y, z = self._xyz(objective.position)
        close = agent is not None and self._horizontal(agent.position, objective.position) <= 8000.0
        lateral = (int(slot) - 2) * (400.0 if close else 700.0)
        lane = 0.0 if close else -8000.0
        stagger = int(pair_index) * (150.0 if close else 400.0)
        return (x + stagger, y + lane + lateral, z)

    def _should_wait(self, state: TeamState, agent) -> bool:
        """先头比队伍中位超前约 50 米、且队伍离据点还远时，原地等一等。"""

        if agent.entity_type not in {self._SOLDIER, self._DOG}:
            return False
        objective = self._battle_objective(state)
        if objective is None:
            return False
        troops = self._alive(state, self._SOLDIER) + self._alive(state, self._DOG)
        if len(troops) < 6:
            return False
        distances = sorted(
            self._horizontal(unit.position, objective.position) for unit in troops
        )
        median = distances[len(distances) // 2]
        if median <= 8000.0:
            return False
        tail = distances[min(len(distances) - 1, int(len(distances) * 0.85))]
        if tail - median > 20000.0:
            return False
        mine = self._horizontal(agent.position, objective.position)
        return mine + 2500.0 < median

    def _move_or_hold(self, agent, point, objective) -> Action:
        if objective is not None and self._count(objective, "blueteamNum") > 0:
            if self._horizontal(agent.position, objective.position) > 600.0:
                return Action.move_at(point)
        if self._horizontal(agent.position, point) <= 1500.0:
            return Action.guard_position(point)
        return Action.move_at(point)

    def _scout_action(self, state: TeamState, agent) -> Action:
        perceived = [enemy for enemy in state.perceived_opponents(agent) if enemy.alive]
        in_range = [enemy for enemy in perceived if self._in_range(agent, enemy)]
        if in_range:
            return Action.attack(self._attack_target(agent, in_range))
        road = self._battle_objective(state)
        if road is None:
            return Action.move("+X")
        uavs = self._alive(state, self._UAV)
        slot = next((index for index, unit in enumerate(uavs) if unit.uid == agent.uid), 0)
        x, y, z = self._spread_point(road, slot, slot % self._GROUPS_PER_TEAM, 0)
        return Action.move_at((x, y, z + 800.0))

    def choose_action(self, state: TeamState, agent) -> Action:
        if not agent.alive:
            return Action.idle()
        if agent.entity_type == self._UAV:
            return self._scout_action(state, agent)

        perceived = [enemy for enemy in state.perceived_opponents(agent) if enemy.alive]
        in_range = [enemy for enemy in perceived if self._in_range(agent, enemy)]
        if in_range:
            return Action.attack(self._attack_target(agent, in_range))
        if self._should_wait(state, agent):
            return Action.guard_position(self._xyz(agent.position))
        if perceived:
            return Action.move_at(self._xyz(self._attack_target(agent, perceived).position))
        visible = [enemy for enemy in state.visible_opponents if enemy.alive]
        if visible:
            return Action.move_at(self._xyz(self._attack_target(agent, visible).position))

        membership = self._membership(state, agent)
        if membership is None:
            team_id = int(agent.team_index) % 3
            slot = int(agent.team_index) % self._GROUPS_PER_TEAM
            pair_index = 0
        else:
            team_id, slot, pair_index = membership
        road = self._battle_objective(state)
        if road is None:
            return Action.move("+X")
        return self._move_or_hold(
            agent,
            self._spread_point(road, team_id, slot, pair_index, agent),
            road,
        )

    def decide(self, state: TeamState):
        return [self.choose_action(state, agent) for agent in state.agents]


RuleAgentAlgorithm = ReinforceAgentAlgorithm
