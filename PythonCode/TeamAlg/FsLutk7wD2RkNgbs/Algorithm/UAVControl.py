from __future__ import annotations

import logging
import math

from cssim.protocol import Action
from .AssaultRoute import AssaultRoute
from .FlightNavigation import FlightNavigation

logger = logging.getLogger(__name__)


class UAVControl:
    """Use public contacts; a report or a single detection is not a dive lock."""

    STRIKE_STEPS = 8
    CRUISE_HEIGHT = 8000.0
    # Policy bounds, not platform guarantees of range, visibility or blast size.
    STRIKE_HORIZONTAL = 3500.0
    STRIKE_DISTANCE = 9000.0
    LOCK_FRAMES = 3
    LOCK_DRIFT = 300.0
    CLUSTER_RADIUS = 1200.0
    FRIENDLY_CLEARANCE = 2000.0
    MAX_LAUNCH_WAIT = 50
    TARGET_DWELL = 8
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
        self._observations = {}
        self._launched = set()
        self._launch_start = 0
        self._fire_watch = {}
        self._stations = {}
        self._focus = {}
        self._body_frame = None
        self._body = []

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
            self._observations.clear()
            self._launched.clear()
            self._launch_start = max(state.step, 0)
            self._fire_watch.clear()
            self._stations.clear()
            self._focus.clear()
            self._body_frame = None
            self._body = []
        self._frame = (state.episode, state.step)
        alive = {agent.uid for agent in state.agents if agent.alive}
        self._missions = {uid: mission for uid, mission in self._missions.items() if uid in alive}
        self._approaches = {uid: value for uid, value in self._approaches.items() if uid in alive}

    def _target(self, state, agent, enemies):
        reserved = [mission for uid, mission in self._missions.items() if uid != agent.uid]
        available = [enemy for enemy in enemies if enemy.alive
                     and enemy.entity_type not in {"BP_Base_UAV_C", "BP_Helicopter_C"}
                     and all(enemy.uid != mission[0] and math.dist(tuple(enemy.position), mission[1]) > self.CLUSTER_RADIUS
                             for mission in reserved)]
        if not available:
            self._focus.pop(agent.uid, None)
            return None
        focus = self._focus.get(agent.uid)
        if focus and state.step - focus[1] < self.TARGET_DWELL:
            retained = next((enemy for enemy in available if enemy.uid == focus[0]), None)
            if retained is not None:
                return retained

        def score(enemy):
            # Prefer ground clusters and reduce overlap with our own troops.
            worth = sum(self.VALUES.get(other.entity_type, 10) for other in enemies
                        if other.alive and math.dist(tuple(other.position), tuple(enemy.position)) <= self.CLUSTER_RADIUS)
            friendly = sum(1 for unit in state.agents if unit.alive and unit.uid != agent.uid
                           and math.dist(tuple(unit.position), tuple(enemy.position)) <= self.CLUSTER_RADIUS)
            ground = enemy.entity_type not in {"BP_Base_UAV_C", "BP_Helicopter_C"}
            return (ground, worth - 100 * friendly,
                    -AssaultRoute.distance(agent.position, enemy.position), -enemy.uid)
        target = max(available, key=score)
        self._focus[agent.uid] = (target.uid, state.step)
        return target

    def _support_body(self, state, objective, route):
        key = (state.episode, state.step, objective.uid if objective else None)
        if self._body_frame != key:
            self._body_frame = key
            self._body = [unit for unit in state.agents if unit.alive
                          and unit.entity_type == 'BP_BaseSoldier_C'
                          and unit.communication_ok() is not False
                          and (route is None or unit.uid not in route._garrison)]
            if objective is not None:
                self._body.sort(key=lambda unit: (AssaultRoute.distance(unit.position, objective.position), unit.uid))
        return self._body

    @staticmethod
    def _escort_enabled(state, objective):
        return objective is not None and any(u.entity_type == 'BP_BaseSoldier_C' for u in state.agents)

    def _strike_supported(self, state, target, objective, route):
        if not self._escort_enabled(state, objective):
            return True
        body = self._support_body(state, objective, route)
        initial = sum(u.entity_type == 'BP_BaseSoldier_C' for u in state.agents)
        required = max(min(4, initial), math.ceil(.6 * len(body)))
        return sum(AssaultRoute.distance(u.position, target.position) <= 22000 for u in body) >= required

    def _escort_point(self, state, agent, objective, altitude, route):
        """One real observation destination per objective, set before takeoff.

        In 20261015_173759, aircraft kept flying around their first rear goal
        even after receiving new coordinates. Time the launch instead of
        relying on unverified in-flight replacement of intermediate goals.
        """
        old = self._stations.get(agent.uid)
        if old is not None:
            return old[1], (objective.uid, 'escort', old[2])
        body = self._support_body(state, objective, route)
        anchor = body[max(0, math.ceil(.6 * len(body)) - 1)] if body else agent
        if self._bearing is None:
            dx, dy = objective.position[0] - anchor.position[0], objective.position[1] - anchor.position[1]
            length = max(math.hypot(dx, dy), 1)
            self._bearing = (dx / length, dy / length) if length > 1 else (1.0, 0.0)
        scout = self._slots.get(agent.uid, 0) == 0
        station_distance = 10000 if scout else 12000
        offset = self._slots.get(agent.uid, 0) * 1500
        bx, by = self._bearing
        point = (float(objective.position[0] - bx * station_distance - by * offset),
                 float(objective.position[1] - by * station_distance + bx * offset), altitude)
        old = (station_distance, point, 0)
        self._stations[agent.uid] = old
        logger.info('无人机固定前沿观察点: uid=%d step=%d objective=%d stand_off_m=%.1f soldiers=%d',
                    agent.uid, state.step, objective.uid, station_distance / 100, len(body))
        return old[1], (objective.uid, 'escort', old[2])

    def _observe(self, state, agent, contacts):
        # Only this UAV's current local detections can advance a lock. Missing
        # frames and movement outside the small anchor region reset it.
        seen = {enemy.uid for enemy in contacts}
        for key in list(self._observations):
            if key[0] == agent.uid and key[1] not in seen:
                del self._observations[key]
        for enemy in contacts:
            key = (agent.uid, enemy.uid)
            point = tuple(float(v) for v in enemy.position)
            old = self._observations.get(key)
            if old and old[0] == state.step:
                continue
            continuous = old and old[0] == state.step - 1 and math.dist(old[1], point) <= self.LOCK_DRIFT
            self._observations[key] = (state.step, old[1] if continuous else point, old[2] + 1 if continuous else 1)

    def _clear(self, state, agent, target):
        return not any(unit.alive and unit.uid != agent.uid
                       and math.dist(tuple(unit.position), tuple(target.position)) <= self.FRIENDLY_CLEARANCE
                       for unit in state.agents)

    @staticmethod
    def _payload(agent):
        try:
            ammo = agent.raw.get("interaction", ()) or ()
            return not ammo or any(float(v) > 0 for v in ammo)
        except (TypeError, ValueError):
            return True

    def _locked(self, state, agent, target):
        observation = self._observations.get((agent.uid, target.uid))
        return (observation is not None and observation[0] == state.step
                and observation[2] >= self.LOCK_FRAMES
                and self._clear(state, agent, target) and self._payload(agent))

    def _launch(self, state, agent, objective, contacts, route=None):
        if agent.uid in self._launched:
            return True
        if self._escort_enabled(state, objective):
            body = self._support_body(state, objective, route)
            limit = 65000 if self._slots.get(agent.uid, 0) == 0 else 60000
            # Match departure to the main body's advance. A timer alone must
            # not launch a direct flight while infantry is still at spawn.
            ready = bool(contacts) or bool(body) and AssaultRoute.distance(
                body[max(0, math.ceil(.6 * len(body)) - 1)].position, objective.position) <= limit
            if ready:
                self._launched.add(agent.uid)
                logger.info('主力接近，放飞至固定前沿: uid=%d step=%d', agent.uid, state.step)
            return ready
        ground = [unit for unit in state.agents if unit.alive and unit.entity_type == "BP_BaseSoldier_C"
                  and unit.communication_ok() is not False]
        distances = sorted(AssaultRoute.distance(unit.position, objective.position) for unit in ground) if objective else []
        # One scout leaves first; the rest depart when a small ground group is
        # approaching. This is an opening latch, never a moving rear waypoint
        # or a requirement for ground troops to wait for aircraft.
        limit = 75000 if self._slots.get(agent.uid, 0) == 0 else 60000
        ready = (not distances or contacts or distances[min(4, len(distances) - 1)] <= limit
                 or state.step - self._launch_start >= self.MAX_LAUNCH_WAIT)
        if ready:
            self._launched.add(agent.uid)
            logger.info('无人机出发: uid=%d step=%d local_contacts=%d', agent.uid, state.step, len(contacts))
        return bool(ready)

    def _normal_attack(self, state, agent, target):
        # Give UE its UID-based attack request for a moving ground contact.
        # Bound requests with neither ammunition use nor damage, then resume
        # navigation. Sending this command is not evidence of a successful hit.
        old = self._fire_watch.get(agent.uid)
        ammo = tuple(agent.raw.get("interaction", ()) or ())
        if (old is None or old[0] != target.uid or old[1] != ammo or target.hp < old[2]
                or state.step - old[3] >= 7):
            old = (target.uid, ammo, float(target.hp), state.step)
            self._fire_watch[agent.uid] = old
        if state.step - old[3] < 3:
            self._navigation.cancel(agent)
            return Action.attack(target)
        return None

    def choose(self, state, agent, objective, route=None):
        self._begin(state)
        objective_uid = objective.uid if objective is not None else None
        if objective_uid != self._objective_uid:
            self._approaches.clear()
            self._stations.clear()
            self._focus.clear()
            self._bearing = None
            self._objective_uid = objective_uid
        if not agent.alive:
            return Action.idle()
        contacts = [enemy for enemy in state.perceived_opponents(agent) if enemy.alive]
        self._observe(state, agent, contacts)
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
            target = next((enemy for enemy in contacts if enemy.uid == uid), None)
            if (target is not None and agent.communication_ok() is not False
                    and self._locked(state, agent, target) and math.dist(point, tuple(target.position)) <= self.LOCK_DRIFT
                    and self._strike_supported(state, target, objective, route)
                    and state.step - started < self.STRIKE_STEPS):
                point = tuple(float(v) for v in target.position)
                self._missions[agent.uid] = (uid, point, started)
                return Action.self_destruct(point)
            del self._missions[agent.uid]
            self._retry_after[(agent.uid, uid)] = state.step + 16
            logger.warning("无人机停止续发自爆（目标失效或超时，未确认命中）: uid=%d target=%d step=%d", agent.uid, uid, state.step)
            # Request a command transition; whether UE can cancel an already
            # started dive is unverified. The strict pre-launch check is primary.
            return Action.guard_position(agent.position)
        if agent.communication_ok() is False:
            self._navigation.cancel(agent)
            return Action.idle()
        target = self._target(state, agent, contacts)
        ground_height = float(objective.position[2]) if objective is not None else 0.0
        base_height = max(float(agent.properties.get("initLocation", {}).get("z", 0)), ground_height)
        altitude = self._altitude.setdefault(agent.uid, base_height + self.CRUISE_HEIGHT)
        if (target is not None and self._locked(state, agent, target)
                and self._strike_supported(state, target, objective, route)
                and self._retry_after.get((agent.uid, target.uid), -1) <= state.step
                and AssaultRoute.distance(agent.position, target.position) <= self.STRIKE_HORIZONTAL
                and math.dist(tuple(agent.position), tuple(target.position)) <= self.STRIKE_DISTANCE):
            point = tuple(float(v) for v in target.position)
            self._missions[agent.uid] = (target.uid, point, state.step)
            self._navigation.cancel(agent)
            logger.info("无人机锁定自爆: uid=%d target=%d point=%s step=%d", agent.uid, target.uid, point, state.step)
            return Action.self_destruct(point)

        if target is not None and self._clear(state, agent, target) and self._payload(agent):
            attack = self._normal_attack(state, agent, target)
            if attack is not None:
                self._launched.add(agent.uid)
                return attack
        if not self._launch(state, agent, objective, contacts, route):
            self._navigation.cancel(agent)
            return Action.guard_position(agent.position)

        if self._escort_enabled(state, objective):
            point, station = self._escort_point(state, agent, objective, altitude, route)
            return self._navigation.apply(state, agent, point, station)

        # Team reports can guide reconnaissance, but launching requires a
        # locally confirmed target. Never aim at an unseen objective occupant.
        if target is not None:
            self._approaches[agent.uid] = (target.uid, tuple(float(v) for v in target.position), state.step)
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
