from __future__ import annotations

import math

from cssim.algorithms import RuleAlgorithm
from cssim.environment import TeamState
from cssim.protocol import Action


class ReinforceAgentAlgorithm(RuleAlgorithm):
    """步兵和机器狗走向无人机最近看见的蓝方，无人机在目标附近停下。

    上一局蓝方留在指挥所时，红方集中冲进去拿到了 2390。下一局蓝方离开据点分兵，
    无人机冲过目标飞出地图，步兵仍只挤指挥所，分数掉到 220。现在记下全队看见的
    敌人，按聚集点分给三队；无人机每次只朝侦察点走近一段，靠近后警戒，避免继续冲出去。
    山猫和运输机的坐标移动平台不执行，这两类改发方向移动。
    """

    _SOLDIER = "BP_BaseSoldier_C"
    _DOG = "BP_RoboDog_C"
    _UAV = "BP_Base_UAV_C"
    _ARMORED = "BP_MNWS_Vehicle_Armored_C"
    _LYNX = "BP_MNWS_Vehicle_6x6UGV_C"
    _HELI = "BP_Helicopter_C"
    _GROUPS_PER_TEAM = 5
    _CONTACT_STEPS = 40
    _CLUSTER_RADIUS = 9000.0
    _CLOSE_RANGE = 12000.0
    _UAV_STEP = 8000.0
    _UAV_HOLD = 2200.0
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
        self._seen: dict[int, tuple[tuple[float, float, float], int]] = {}
        self._motion: dict[int, list[tuple[int, float, float]]] = {}
        self._last_step: int | None = None

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

    def _observe(self, state: TeamState) -> None:
        step = int(state.step)
        if self._last_step is not None and step < self._last_step:
            self._seen.clear()
            self._motion.clear()
        self._last_step = step
        for enemy in state.visible_opponents:
            if not enemy.alive:
                continue
            self._seen[int(enemy.uid)] = (self._xyz(enemy.position), step)
        self._seen = {
            uid: item
            for uid, item in self._seen.items()
            if step - int(item[1]) <= self._CONTACT_STEPS
        }
        for agent in self._alive(state):
            history = self._motion.setdefault(int(agent.uid), [])
            history.append((step, float(agent.position[0]), float(agent.position[1])))
            if len(history) > 25:
                del history[:-25]

    def _clusters(self, state: TeamState) -> list[tuple[float, float, float]]:
        points = [item[0] for item in self._seen.values()]
        remaining = sorted(points, key=lambda point: (point[0], point[1], point[2]))
        clusters: list[tuple[float, float, float]] = []
        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            rest = []
            for point in remaining:
                if self._horizontal(seed, point) <= self._CLUSTER_RADIUS:
                    group.append(point)
                else:
                    rest.append(point)
            remaining = rest
            count = float(len(group))
            clusters.append((
                sum(point[0] for point in group) / count,
                sum(point[1] for point in group) / count,
                sum(point[2] for point in group) / count,
            ))
        return clusters

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

    def _formation_point(
        self,
        agent,
        anchor: tuple[float, float, float],
        team_id: int,
        slot: int,
        pair_index: int,
    ) -> tuple[tuple[float, float, float], bool]:
        close = self._horizontal(agent.position, anchor) <= self._CLOSE_RANGE
        lane = (int(team_id) % 3 - 1) * (500.0 if close else 8000.0)
        lateral = (int(slot) - 2) * (350.0 if close else 1600.0)
        stagger = int(pair_index) * (200.0 if close else 500.0)
        point = (
            anchor[0] + stagger,
            anchor[1] + lane + lateral,
            float(agent.position[2]) if agent.entity_type == self._ARMORED else anchor[2],
        )
        return point, close

    def _move_or_hold(self, agent, point, hold: float) -> Action:
        if self._horizontal(agent.position, point) <= hold:
            return Action.guard_position(point)
        return Action.move_at(point)

    def _direction_toward(self, agent, point: tuple[float, float, float]) -> Action:
        dx = point[0] - float(agent.position[0])
        dy = point[1] - float(agent.position[1])
        if math.hypot(dx, dy) <= 600.0:
            return Action.guard_position(point)
        horizontal = "+X" if dx > 400.0 else "-X" if dx < -400.0 else ""
        vertical = "+Y" if dy > 400.0 else "-Y" if dy < -400.0 else ""
        direction = f"{horizontal}{vertical}" or ("-X" if dx < 0 else "+X")
        return Action.move(direction)

    def _stuck(self, agent) -> bool:
        history = self._motion.get(int(agent.uid)) or []
        if len(history) < 16:
            return False
        old = history[-16]
        return self._horizontal((old[1], old[2], 0.0), agent.position) < 400.0

    def _anchor_for(self, state: TeamState, team_id: int) -> tuple[float, float, float] | None:
        clusters = self._clusters(state)
        if clusters:
            return clusters[int(team_id) % len(clusters)]
        objective = self._battle_objective(state)
        if objective is None:
            return None
        return self._xyz(objective.position)

    def _step_toward(
        self,
        origin: tuple[float, float, float],
        dest: tuple[float, float, float],
        limit: float,
    ) -> tuple[float, float, float]:
        dx = dest[0] - origin[0]
        dy = dest[1] - origin[1]
        dz = dest[2] - origin[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance <= limit or distance <= 1.0:
            return dest
        scale = limit / distance
        return (origin[0] + dx * scale, origin[1] + dy * scale, origin[2] + dz * scale)

    def _loiter_point(self, state: TeamState, slot: int) -> tuple[float, float, float] | None:
        clusters = self._clusters(state)
        if clusters:
            base = clusters[slot % len(clusters)]
        else:
            roads = self._roads(state)
            if not roads:
                return None
            battle = self._battle_objective(state)
            others = [
                road for road in roads
                if battle is None or int(road.uid) != int(battle.uid)
            ]
            if slot < 3 or not others or battle is None:
                chosen = battle or roads[slot % len(roads)]
                base = self._xyz(chosen.position)
            else:
                base = self._xyz(others[(slot - 3) % len(others)].position)
        lane = (int(slot) % 3 - 1) * 4000.0
        return (base[0], base[1] + lane, base[2] + 1200.0)

    def _scout_action(self, state: TeamState, agent) -> Action:
        perceived = [enemy for enemy in state.perceived_opponents(agent) if enemy.alive]
        in_range = [enemy for enemy in perceived if self._in_range(agent, enemy)]
        if in_range:
            return Action.attack(self._attack_target(agent, in_range))
        uavs = self._alive(state, self._UAV)
        slot = next((index for index, unit in enumerate(uavs) if unit.uid == agent.uid), 0)
        loiter = self._loiter_point(state, slot)
        if loiter is None:
            return Action.move("-X")
        if self._horizontal(agent.position, loiter) <= self._UAV_HOLD:
            return Action.guard_position(loiter)
        return Action.move_at(self._step_toward(self._xyz(agent.position), loiter, self._UAV_STEP))

    def _ground_action(self, state: TeamState, agent) -> Action:
        perceived = [enemy for enemy in state.perceived_opponents(agent) if enemy.alive]
        in_range = [enemy for enemy in perceived if self._in_range(agent, enemy)]
        if in_range:
            return Action.attack(self._attack_target(agent, in_range))
        if perceived:
            return Action.move_at(self._xyz(self._attack_target(agent, perceived).position))

        membership = self._membership(state, agent)
        if membership is None:
            same_type = self._alive(state, agent.entity_type)
            index = next(
                (item for item, unit in enumerate(same_type) if unit.uid == agent.uid),
                int(agent.team_index),
            )
            team_id = index % 3
            slot = index % self._GROUPS_PER_TEAM
            pair_index = 0
        else:
            team_id, slot, pair_index = membership
        anchor = self._anchor_for(state, team_id)
        if anchor is None:
            return Action.move("-X")
        point, close = self._formation_point(agent, anchor, team_id, slot, pair_index)
        if agent.entity_type in {self._LYNX, self._HELI} or (
            agent.entity_type == self._ARMORED and self._stuck(agent)
        ):
            return self._direction_toward(agent, point)
        return self._move_or_hold(agent, point, 600.0 if close else 1500.0)

    def choose_action(self, state: TeamState, agent) -> Action:
        if not agent.alive:
            return Action.idle()
        if agent.entity_type == self._UAV:
            return self._scout_action(state, agent)
        return self._ground_action(state, agent)

    def decide(self, state: TeamState):
        self._observe(state)
        return [self.choose_action(state, agent) for agent in state.agents]


RuleAgentAlgorithm = ReinforceAgentAlgorithm
