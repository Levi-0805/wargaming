from __future__ import annotations

import logging
import math

from cssim.protocol import Action
from .AssaultRoute import AssaultRoute
from .FlightNavigation import FlightNavigation

logger = logging.getLogger(__name__)


class UAVControl:
    """Advance while gaining height, approach a visible ground target, dive.

    The approach radius is a policy choice, not an API precondition. Long
    low-altitude strike approaches were shot down in the recorded match.
    """

    STRIKE_STEPS = 12
    CRUISE_HEIGHT = 8000.0
    STRIKE_HORIZONTAL = 12000.0  # Policy limit; local confirmation is still mandatory.
    CLUSTER_RADIUS = 1200.0  # Scenario payload blast radius, cm; scoring heuristic.
    VALUES = {"BP_BaseSoldier_C": 100, "BP_RoboDog_C": 20,
              "BP_MNWS_Vehicle_Armored_C": 90, "BP_MNWS_Vehicle_6x6UGV_C": 50,
              "BP_Base_UAV_C": 10, "BP_Helicopter_C": 50}

    def __init__(self):
        self._frame = None
        self._missions = {}
        self._navigation = FlightNavigation()
        self._retry_after = {}
        self._altitude = {}
        self._approaches = {}
        self._slots = {}
        self._bearing = (1.0, 0.0)
        self._objective_uid = None
        self._telemetry = {}

    def _begin(self, state):
        if self._frame is None or self._frame[0] != state.episode or state.step < self._frame[1]:
            self._missions.clear()
            self._navigation = FlightNavigation()
            self._retry_after.clear()
            self._altitude.clear()
            self._approaches.clear()
            uavs = [unit for unit in state.agents if unit.entity_type == "BP_Base_UAV_C"]
            offsets = (0, -1, 1, -2, 2)
            self._slots = {unit.uid: offsets[i % len(offsets)] for i, unit in enumerate(uavs)}
            self._bearing = None
            self._objective_uid = None
            self._telemetry.clear()
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
        objective_uid = objective.uid if objective is not None else None
        if objective_uid != self._objective_uid:
            self._approaches.clear()
            self._bearing = None
            self._objective_uid = objective_uid
        if not agent.alive:
            return Action.idle()
        contacts = [enemy for enemy in state.perceived_opponents(agent) if enemy.alive]
        if state.step >= 0 and state.step % 40 == 0 and self._telemetry.get(agent.uid) != state.step:
            self._telemetry[agent.uid] = state.step
            logger.info('无人机任务观测: uid=%d step=%d objective=%s distance_m=%.1f contacts=%d ground_contacts=%d link=%s',
                        agent.uid, state.step, objective_uid,
                        AssaultRoute.distance(agent.position, objective.position) / 100 if objective else -1,
                        len(contacts), sum(e.entity_type not in {'BP_Base_UAV_C', 'BP_Helicopter_C'} for e in contacts),
                        agent.communication_ok())
        mission = self._missions.get(agent.uid)
        if mission is not None:
            self._navigation.cancel(agent)
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
            self._navigation.cancel(agent)
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
            reports = [enemy for enemy in state.visible_opponents if objective is None
                       or AssaultRoute.distance(enemy.position, objective.position) <= 20000]
            shared = target or self._target(state, agent, reports)
            point = shared.position if shared is not None else objective.position if objective is not None else None
            if shared is not None:
                self._approaches[agent.uid] = (shared.uid, tuple(float(v) for v in point), state.step)
        if point is None:
            self._navigation.cancel(agent)
            return Action.guard_position(agent.position)
        point = (float(point[0]), float(point[1]), altitude)
        # Fixed roster slots around a common approach axis. Losses never
        # reshuffle survivors and a lagging wing never stops the other UAVs.
        if self._bearing is None:
            dx, dy = point[0] - agent.position[0], point[1] - agent.position[1]
            length = max(math.hypot(dx, dy), 1)
            self._bearing = (dx / length, dy / length)
        offset = self._slots.get(agent.uid, 0) * 3000
        point = (point[0] - self._bearing[1] * offset,
                 point[1] + self._bearing[0] * offset, altitude)
        # The former 600m soldier clamp created another changing rear waypoint.
        # Fly to the actual bounded objective/report. CommandLinks handles the
        # observed communication state and reassigns a live parent when needed.
        return self._navigation.apply(state, agent, point, objective_uid)
