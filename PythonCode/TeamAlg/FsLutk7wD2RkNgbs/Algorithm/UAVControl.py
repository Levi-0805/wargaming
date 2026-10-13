from __future__ import annotations

import logging
import math

from cssim.protocol import Action
from .AssaultRoute import AssaultRoute

logger = logging.getLogger(__name__)


class UAVControl:
    """Advance while gaining height, approach a visible ground target, dive.

    The approach radius is a policy choice, not an API precondition. Long
    low-altitude strike approaches were shot down in the recorded match.
    """

    STRIKE_STEPS = 12
    CRUISE_HEIGHT = 8000.0
    STRIKE_HORIZONTAL = 2000.0
    CLUSTER_RADIUS = 1200.0  # Scenario payload blast radius, cm; scoring heuristic.
    VALUES = {"BP_BaseSoldier_C": 100, "BP_RoboDog_C": 20,
              "BP_MNWS_Vehicle_Armored_C": 90, "BP_MNWS_Vehicle_6x6UGV_C": 50,
              "BP_Base_UAV_C": 10, "BP_Helicopter_C": 50}

    def __init__(self):
        self._frame = None
        self._missions = {}
        self._flight = {}
        self._retry_after = {}
        self._altitude = {}
        self._approaches = {}
        self._bypass = {}

    def _begin(self, state):
        if self._frame is None or self._frame[0] != state.episode or state.step < self._frame[1]:
            self._missions.clear()
            self._flight.clear()
            self._retry_after.clear()
            self._altitude.clear()
            self._approaches.clear()
            self._bypass.clear()
        self._frame = (state.episode, state.step)
        alive = {agent.uid for agent in state.agents if agent.alive}
        self._missions = {uid: mission for uid, mission in self._missions.items() if uid in alive}
        self._approaches = {uid: value for uid, value in self._approaches.items() if uid in alive}

    def _target(self, state, agent, enemies):
        reserved = [mission[1] for uid, mission in self._missions.items() if uid != agent.uid]
        available = [enemy for enemy in enemies if enemy.alive
                     and enemy.entity_type not in {"BP_Base_UAV_C", "BP_Helicopter_C"}
                     and self._retry_after.get((agent.uid, enemy.uid), -1) <= state.step
                     and all(math.dist(tuple(enemy.position), point) > self.CLUSTER_RADIUS for point in reserved)]
        if not available:
            return None

        def score(enemy):
            # Prefer ground clusters and reduce overlap with our own troops.
            worth = sum(self.VALUES.get(other.entity_type, 10) for other in enemies
                        if other.alive and math.dist(tuple(other.position), tuple(enemy.position)) <= self.CLUSTER_RADIUS)
            friendly = sum(1 for unit in state.agents if unit.alive and unit.uid != agent.uid
                           and math.dist(tuple(unit.position), tuple(enemy.position)) <= self.CLUSTER_RADIUS)
            ground = enemy.entity_type not in {"BP_Base_UAV_C", "BP_Helicopter_C"}
            return (ground, worth - 100 * friendly,
                    -AssaultRoute.distance(agent.position, enemy.position), -enemy.uid)
        return max(available, key=score)

    def choose(self, state, agent, objective):
        self._begin(state)
        if not agent.alive:
            return Action.idle()
        contacts = [enemy for enemy in state.perceived_opponents(agent) if enemy.alive]
        mission = self._missions.get(agent.uid)
        if mission is not None:
            uid, point, started = mission
            target = state.entity_by_uid(uid)
            if target is not None and not target.alive:
                del self._missions[agent.uid]
            elif state.step - started < self.STRIKE_STEPS:
                return Action.self_destruct(point)
            else:
                # Alive after the deadline is not proof of a successful hit.
                del self._missions[agent.uid]
                self._retry_after[(agent.uid, uid)] = state.step + 8
                logger.warning("无人机自爆未确认: uid=%d target=%d step=%d", agent.uid, uid, state.step)
                if target is not None and any(enemy.uid == uid for enemy in contacts):
                    return Action.attack(target)
        target = self._target(state, agent, contacts)
        ground_height = float(objective.position[2]) if objective is not None else 0.0
        base_height = max(float(agent.properties.get("initLocation", {}).get("z", 0)), ground_height)
        altitude = self._altitude.setdefault(agent.uid, base_height + self.CRUISE_HEIGHT)
        if (target is not None and agent.communication_ok() is not False
                and AssaultRoute.distance(agent.position, target.position) <= self.STRIKE_HORIZONTAL):
            point = tuple(float(v) for v in target.position)
            self._missions[agent.uid] = (target.uid, point, state.step)
            logger.info("无人机锁定自爆: uid=%d target=%d point=%s step=%d", agent.uid, target.uid, point, state.step)
            return Action.self_destruct(point)

        # Team reports can guide reconnaissance, but launching requires a
        # locally confirmed target. Never aim at an unseen objective occupant.
        locked = self._approaches.get(agent.uid)
        known = state.entity_by_uid(locked[0]) if locked else None
        if known is not None and known.alive:
            point = tuple(float(v) for v in known.position)
            self._approaches[agent.uid] = (known.uid, point, state.step)
        elif locked and known is None and state.step - locked[2] <= 6:
            point = locked[1]  # Last reported position, never an unseen update.
        else:
            self._approaches.pop(agent.uid, None)
            shared = target or self._target(state, agent, state.visible_opponents)
            point = shared.position if shared is not None else objective.position if objective is not None else None
            if shared is not None:
                self._approaches[agent.uid] = (shared.uid, tuple(float(v) for v in point), state.step)
        if point is None:
            return Action.guard_position(agent.position)
        point = (float(point[0]), float(point[1]), altitude)
        ground = [unit for unit in state.agents if unit.alive and unit.entity_type == "BP_BaseSoldier_C"]
        if ground:
            support = min(ground, key=lambda unit: AssaultRoute.distance(unit.position, point))
            distance = AssaultRoute.distance(support.position, point)
            if distance > 60000:
                ratio = 60000 / distance
                point = (float(support.position[0]) + (point[0] - support.position[0]) * ratio,
                         float(support.position[1]) + (point[1] - support.position[1]) * ratio,
                         altitude)
        # The platform's +Z direction order gained only ~36m in 250 frames.
        # Height is a destination component, never a prerequisite for XY travel.
        distance = AssaultRoute.distance(agent.position, point)
        if distance <= 1500 and abs(float(agent.position[2]) - altitude) <= 1500:
            return Action.guard_position(point)
        bypass = self._bypass.get(agent.uid)
        if bypass is not None:
            if state.step < bypass[1] and AssaultRoute.distance(agent.position, bypass[0]) > 800:
                return Action.move_at(bypass[0])
            self._bypass.pop(agent.uid, None)
            self._flight.pop(agent.uid, None)
        previous = self._flight.get(agent.uid)
        if previous is None or math.dist(point, previous[0]) > 10000 or distance < previous[2] - 200:
            previous = (point, state.step, distance)
            self._flight[agent.uid] = previous
        if state.step - previous[1] >= 4:
            self._altitude[agent.uid] = min(altitude + 2000, base_height + 12000)
            self._flight.pop(agent.uid, None)
            dx, dy = point[0] - agent.position[0], point[1] - agent.position[1]
            length = max(math.hypot(dx, dy), 1)
            side = 1 if agent.team_index % 2 else -1
            bypass_point = (float(agent.position[0]) - dy / length * side * 3000,
                            float(agent.position[1]) + dx / length * side * 3000,
                            self._altitude[agent.uid])
            self._bypass[agent.uid] = (bypass_point, state.step + 3)
            logger.info("无人机短绕障: uid=%d step=%d target=%s", agent.uid, state.step, point)
            return Action.move_at(bypass_point)
        return Action.move_at(point)
