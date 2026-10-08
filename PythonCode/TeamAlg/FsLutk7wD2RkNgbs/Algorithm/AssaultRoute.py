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

    def __init__(self):
        self._episode = None
        self._step = None
        self.order = ()
        self.stage = 0
        self._confirmed = 0
        self._cursor = {}
        self._routes = ()
        self.known_map = False

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
        obj = self.objective(state)
        if obj is None:
            return
        ours = obj.team == state.team if obj.team is not None else self.count(obj, "percent") >= 1.0
        enemy_field = "blueteamNum" if state.team == 0 else "redTeamNum"
        clear = ours and self.count(obj, enemy_field) == 0
        self._confirmed = self._confirmed + 1 if clear else 0
        if self._confirmed >= 2 and self.stage + 1 < len(self.order):
            self.stage += 1
            self._confirmed = 0
            logger.info("据点已占领且清敌，转向下一目标: step=%d uid=%d", state.step, self.order[self.stage])

    def objective(self, state):
        return state.key_object_by_uid(self.order[self.stage]) if self.order else None

    def point(self, state, agent):
        """Rear units finish the approach even after the front captures HQ."""
        if not self.order:
            return None
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
