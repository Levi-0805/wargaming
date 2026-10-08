from __future__ import annotations

import logging
import math

from cssim.protocol import Action
from .AssaultRoute import AssaultRoute
from .MotionRecovery import MotionRecovery

logger = logging.getLogger(__name__)


class UAVControl:
    """Scout with bounded destinations; commit the documented strike action.

    SelfDestruct takes the observed target's xyz. There is no documented
    25m launch precondition. Keep issuing the same mission briefly instead
    of cancelling it with ordinary Moving on the following frame.
    """

    STRIKE_STEPS = 12
    CLUSTER_RADIUS = 1200.0  # Scenario payload blast radius, cm; scoring heuristic.
    VALUES = {"BP_BaseSoldier_C": 100, "BP_RoboDog_C": 20,
              "BP_MNWS_Vehicle_Armored_C": 90, "BP_MNWS_Vehicle_6x6UGV_C": 50,
              "BP_Base_UAV_C": 10, "BP_Helicopter_C": 50}

    def __init__(self):
        self._frame = None
        self._missions = {}
        self._flight = {}
        self._retry_after = {}

    def _begin(self, state):
        if self._frame is None or self._frame[0] != state.episode or state.step < self._frame[1]:
            self._missions.clear()
            self._flight.clear()
            self._retry_after.clear()
        self._frame = (state.episode, state.step)
        alive = {agent.uid for agent in state.agents if agent.alive}
        self._missions = {uid: mission for uid, mission in self._missions.items() if uid in alive}

    def _target(self, state, agent, enemies):
        reserved = [mission[1] for uid, mission in self._missions.items() if uid != agent.uid]
        available = [enemy for enemy in enemies if enemy.alive
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
        if target is not None and agent.communication_ok() is not False:
            point = tuple(float(v) for v in target.position)
            self._missions[agent.uid] = (target.uid, point, state.step)
            logger.info("无人机锁定自爆: uid=%d target=%d point=%s step=%d", agent.uid, target.uid, point, state.step)
            return Action.self_destruct(point)

        # Team reports can guide reconnaissance, but launching requires a
        # locally confirmed target. Never aim at an unseen objective occupant.
        shared = self._target(state, agent, state.visible_opponents)
        point = shared.position if shared is not None else objective.position if objective is not None else None
        if point is None:
            return Action.guard_position(agent.position)
        point = (float(point[0]), float(point[1]), float(point[2]) + 2500)
        ground = [unit for unit in state.agents if unit.alive and unit.entity_type == "BP_BaseSoldier_C"]
        if ground:
            support = min(ground, key=lambda unit: AssaultRoute.distance(unit.position, point))
            distance = AssaultRoute.distance(support.position, point)
            if distance > 60000:
                ratio = 60000 / distance
                point = (float(support.position[0]) + (point[0] - support.position[0]) * ratio,
                         float(support.position[1]) + (point[1] - support.position[1]) * ratio,
                         float(support.position[2]) + 2500)
        distance = math.dist(tuple(agent.position), point)
        if distance <= 1500:
            return Action.guard_position(point)
        previous = self._flight.get(agent.uid)
        if previous is None or math.dist(point, previous[0]) > 1500 or distance < previous[2] - 200:
            previous = (point, state.step, distance)
            self._flight[agent.uid] = previous
        if state.step - previous[1] >= 6:
            # Directly correct toward the destination, never rotate away by
            # the infantry's obstacle-escape angles or follow stale road nodes.
            vector = tuple(point[i] - float(agent.position[i]) for i in range(3))
            return Action.move(MotionRecovery._direction(vector, 0, airborne=True))
        return Action.move_at(point)
