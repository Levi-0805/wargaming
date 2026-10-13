from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


class AssaultRoute:
    """One objective sequence, with independent, monotonic route cursors.

    Coordinates are cm. LOWER samples actual ground trajectories in
    Output_20261008_230451 (136/176). HQ_B samples the ground path in
    Output_20260924_150433 episode 2 (671). Only map geometry is reused;
    runtime decisions never read replay enemies. Other maps use UE navigation.
    """

    LOWER = (
        (77387, 36261, 7394), (65185, 36303, 6082),
        (59734, 36261, 5646), (49548, 34208, 5747),
        (38220, 32771, 5998), (31881, 31436, 6411),
        (20366, 30043, 6408), (9200, 28897, 6408),
        (2534, 28097, 6408), (-6470, 30114, 6425),
        (-17959, 27728, 6425), (-29782, 25170, 6426),
    )
    HQ_B = (
        (-37782, 24389, 6408), (-46037, 15885, 6425),
        (-50063, 5399, 6425), (-49816, -6016, 5943),
        (-53046, -18007, 5880), (-47310, -27482, 5617),
    )
    ANCHORS = ((-41810, 32275), (-45450, -30945), (-3000, -27530))
    ARRIVAL = 1600.0
    CAPTURE_CONFIRMATIONS = 4
    GARRISON_CORE = 2500.0
    GARRISON_MIN = 4
    GARRISON_MARGIN = 3

    def __init__(self):
        self._episode = None
        self._step = None
        self.order = ()
        self.stage = 0
        self._confirmed = 0
        self._cursor = {}
        self._routes = ()
        self.known_map = False
        self._garrison = {}
        self._guard_slots = {}
        self._lost = 0

    @staticmethod
    def distance(left, right):
        return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))

    @staticmethod
    def count(obj, field):
        try:
            return float(obj.raw.get(field, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    def _initialize(self, state):
        self.order, self.stage, self._confirmed = (), 0, 0
        self._cursor.clear()
        self._garrison.clear()
        self._guard_slots.clear()
        self._lost = 0
        self._routes = ()
        self.known_map = False
        objects = [obj for obj in state.key_objects if obj.valid]
        if not objects:
            return
        matched = [min(objects, key=lambda obj: self.distance(obj.position, point)) for point in self.ANCHORS]
        self.known_map = len({obj.uid for obj in matched}) == 3 and all(
            self.distance(obj.position, point) < 500 for obj, point in zip(matched, self.ANCHORS)
        )
        if self.known_map:
            ordered = matched  # HQ -> B -> A, left to right in the supplied view.
        else:
            hq = next((obj for obj in objects if any(
                word in str(obj.raw.get("rSVD1", "")).lower()
                for word in ("指挥", "headquarter", "hq")
            )), objects[0])
            ordered = [hq]
            remaining = [obj for obj in objects if obj.uid != hq.uid]
            while remaining:
                target = min(remaining, key=lambda obj: self.distance(ordered[-1].position, obj.position))
                ordered.append(target)
                remaining = [obj for obj in remaining if obj.uid != target.uid]
        self.order = tuple(obj.uid for obj in ordered)
        self._routes = tuple(
            tuple(prefix) + (tuple(float(v) for v in obj.position),)
            for obj, prefix in zip(ordered, (self.LOWER, self.HQ_B, ()) if self.known_map else [()] * len(ordered))
        )
        logger.info("统一推进目标顺序: %s 下路坐标匹配=%s", self.order, self.known_map)

    def update(self, state):
        if self._episode != state.episode or (self._step is not None and state.step < self._step):
            self._initialize(state)
            self._step = None
        elif not self.order:
            self._initialize(state)
        self._episode = state.episode
        if self._step == state.step:
            return
        self._step = state.step
        # A full progress bar can be reversed immediately by surviving defenders.
        # Revisit a lost earlier objective instead of leaving vehicles en route to B.
        lost = next((i for i, uid in enumerate(self.order[:self.stage])
                     if (previous := state.key_object_by_uid(uid)) is not None
                     and (previous.team != state.team if previous.team is not None
                          else self.count(previous, "percent") < 1)), None)
        self._lost = self._lost + 1 if lost is not None else 0
        if self._lost >= 2:
            self.stage, self._confirmed, self._lost = lost, 0, 0
            route = self._routes[self.stage]
            for unit in state.agents:
                cursor = self._cursor.get(unit.uid)
                if cursor is not None and cursor[0] > self.stage:
                    self._cursor[unit.uid] = (self.stage, min(
                        range(len(route)), key=lambda i: self.distance(unit.position, route[i])))
            logger.warning("已占目标失守，停止转场: step=%d uid=%d", state.step, self.order[self.stage])
        obj = self.objective(state)
        if obj is None:
            return
        friendly_field, enemy_field = (("redTeamNum", "blueteamNum") if state.team == 0
                                       else ("blueteamNum", "redTeamNum"))
        ours = obj.team == state.team if obj.team is not None else (
            self.count(obj, "percent") >= 1.0
            and self.count(obj, friendly_field) > self.count(obj, enemy_field))
        self._confirmed = self._confirmed + 1 if ours else 0
        # Reserve defenders before departure, not after everybody has started
        # moving to B. A label is not evidence that a unit is inside the point.
        self._assign_garrison(state, include_current=ours)
        guards = [unit for unit in state.agents if self._garrison.get(unit.uid) == obj.uid]
        ready = len(guards) >= self.garrison_needed(state, obj) and all(
            self.inside_garrison(obj, unit.position) for unit in guards)
        if (self._confirmed >= self.CAPTURE_CONFIRMATIONS and ready
                and self.stage + 1 < len(self.order)):
            self.stage += 1
            self._confirmed = 0
            logger.info("据点占领已确认，主力转向下一目标: step=%d uid=%d", state.step, self.order[self.stage])

    def garrison_needed(self, state, obj):
        enemy_field = "blueteamNum" if state.team == 0 else "redTeamNum"
        return max(self.GARRISON_MIN, int(self.count(obj, enemy_field)) + self.GARRISON_MARGIN)

    def _assign_garrison(self, state, include_current=False):
        previous, self._garrison = self._garrison, {}
        for uid in self.order[:self.stage + int(include_current)]:
            obj = state.key_object_by_uid(uid)
            if obj is None or not obj.valid:
                continue
            nearby = [unit for unit in state.agents if unit.alive
                      and unit.entity_type in {"BP_BaseSoldier_C", "BP_RoboDog_C"}
                      and unit.communication_ok() is not False
                      and unit.uid not in self._garrison
                      and self.distance(unit.position, obj.position) <= 22000]
            needed = self.garrison_needed(state, obj)
            nearby.sort(key=lambda unit: (
                not self.inside_garrison(obj, unit.position),
                previous.get(unit.uid) != uid,
                self.distance(unit.position, obj.position), unit.uid))
            self._garrison.update({unit.uid: uid for unit in nearby[:needed]})
        self._guard_slots = {key: point for key, point in self._guard_slots.items()
                             if self._garrison.get(key[0]) == key[1]}

    @staticmethod
    def _footprint(obj):
        # UE supplies box extents in this scenario (HQ: 77m x 65m).
        # Keep a 4m inset. Never assume the building centre is walkable.
        scale = obj.raw.get("scale", {})
        if isinstance(scale, dict) and float(scale.get("x", 0)) > 400 and float(scale.get("y", 0)) > 400:
            return (float(scale["x"]) - 400, float(scale["y"]) - 400)
        return None

    @staticmethod
    def _local_point(obj, point):
        yaw = math.radians(float(obj.raw.get("rotation", {}).get("yaw", 0)))
        dx, dy = float(point[0]) - obj.position[0], float(point[1]) - obj.position[1]
        return dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw), yaw

    def inside_garrison(self, obj, point):
        extent = self._footprint(obj)
        if extent:
            x, y, _ = self._local_point(obj, point)
            return abs(x) <= extent[0] and abs(y) <= extent[1]
        return self.distance(obj.position, point) <= self.GARRISON_CORE

    def guard_slot(self, agent, obj):
        key = (agent.uid, obj.uid)
        if self._footprint(obj):
            # An actual reached position is a better hold point than an
            # unverified central waypoint inside a building.
            return self._guard_slots.setdefault(key, self.clamp_guard_point(obj, agent.position))
        # Small fixed interior slots, independent of casualties and target
        # movement. Do not send defenders chasing a contact outside the point.
        slot = agent.team_index % 9
        return (float(obj.position[0]) + (slot % 3 - 1) * 700,
                float(obj.position[1]) + (slot // 3 - 1) * 700, float(obj.position[2]))

    def clamp_guard_point(self, obj, point):
        extent = self._footprint(obj)
        if extent:
            x, y, yaw = self._local_point(obj, point)
            x, y = max(-extent[0], min(extent[0], x)), max(-extent[1], min(extent[1], y))
            return (float(obj.position[0]) + x * math.cos(yaw) - y * math.sin(yaw),
                    float(obj.position[1]) + x * math.sin(yaw) + y * math.cos(yaw), float(point[2]))
        distance = self.distance(obj.position, point)
        ratio = min(1.0, (self.GARRISON_CORE - 400) / max(distance, 1.0))
        return (float(obj.position[0]) + (float(point[0]) - obj.position[0]) * ratio,
                float(obj.position[1]) + (float(point[1]) - obj.position[1]) * ratio, float(obj.position[2]))

    def garrison_for(self, state, agent):
        uid = self._garrison.get(agent.uid)
        return state.key_object_by_uid(uid) if uid is not None else None

    def departing(self, state, agent):
        return self.stage > 0 and agent.uid not in self._garrison and any(
            obj is not None and self.distance(agent.position, obj.position) <= 25000
            for obj in (state.key_object_by_uid(uid) for uid in self.order[:self.stage])
        )

    def objective(self, state):
        return state.key_object_by_uid(self.order[self.stage]) if self.order else None

    def point(self, state, agent):
        """Rear units finish the approach even after the front captures HQ."""
        if not self.order:
            return None
        garrison = self.garrison_for(state, agent)
        if garrison is not None:
            return tuple(float(v) for v in garrison.position)
        position = agent.position
        if agent.uid not in self._cursor:
            route = self._routes[0]
            if self.known_map:
                # Join at the observed road entrance, not west through the
                # buildings adjacent to the x~60000,y~12000 spawn.
                index = 1 if position[1] < 26000 and position[0] > 50000 else min(
                    range(len(route)), key=lambda i: self.distance(position, route[i])
                )
            else:
                index = 0
            self._cursor[agent.uid] = (0, index)
        leg, index = self._cursor[agent.uid]
        route = self._routes[leg]
        # Combat may carry a unit past a waypoint. Rejoin at an actually
        # reached later point instead of ordering it back down the road.
        later = [i for i in range(index + 1, len(route))
                 if self.distance(position, route[i]) <= self.ARRIVAL]
        if later:
            index = max(later)
        if leg < self.stage and self.distance(position, route[-1]) <= 15000:
            leg, index = leg + 1, 0
            route = self._routes[leg]
        while self.distance(position, route[index]) <= self.ARRIVAL:
            if index + 1 < len(route):
                index += 1
            elif leg < self.stage:
                leg, index = leg + 1, 0
                route = self._routes[leg]
            else:
                break
        self._cursor[agent.uid] = (leg, index)
        return route[index]
