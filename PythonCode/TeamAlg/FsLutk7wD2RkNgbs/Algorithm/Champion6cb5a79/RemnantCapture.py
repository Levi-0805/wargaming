from __future__ import annotations

import heapq
import logging
import math

logger = logging.getLogger(__name__)


class RemnantCapture:
    """A public-observation fallback for a collapsed, unsupported B assault.

    This changes navigation requests, not the engine's navigation mesh. The
    detour excludes the public B box with a margin; it cannot certify cover or
    traversability between waypoints.
    """

    FOOT = {'BP_BaseSoldier_C', 'BP_RoboDog_C'}
    CARS = {'BP_MNWS_Vehicle_Armored_C', 'BP_MNWS_Vehicle_6x6UGV_C'}
    CLEAR_FRAMES = 3
    MARGIN = 3000.0

    def __init__(self):
        self.active = False
        self.source_uid = None
        self.target_uid = None
        self.started = None
        self.reason = None
        self.ready = False
        self._frame = None
        self._clear = 0
        self._peak = 0
        self._approached = False
        self._paths = {}
        self._cursor = {}
        self._safe = {}
        self._watch = {}

    @classmethod
    def capturers(cls, state):
        return [u for u in state.agents if u.alive and u.hp > 0 and u.entity_type in cls.FOOT]

    @classmethod
    def armed_support(cls, state, route=None):
        base_b = state.key_object_by_uid(route._map_order[1]) if route is not None else None
        foot = cls.capturers(state)
        for unit in state.agents:
            if (not unit.alive or unit.hp <= 0 or unit.entity_type not in cls.CARS
                    or unit.communication_ok() is False):
                continue
            if (route is not None and base_b is not None
                    and route.distance(unit.position, base_b.position) > 35000
                    and not any(route.distance(unit.position, u.position) <= 15000 for u in foot)):
                # A fully armed car stranded a kilometre behind the assault is
                # not available support for the remaining capture units.
                continue
            try:
                ammo = [float(v) for v in unit.raw.get('interaction', ())]
            except (TypeError, ValueError):
                return True
            # Unknown ammunition cannot prove the supporting vehicle is empty.
            if not ammo or any(not math.isfinite(v) or v != 0 for v in ammo):
                return True
        return False

    def update(self, state, route):
        """Return True when this fallback owns objective selection."""
        if not route.known_map or len(route._map_order) != 3:
            return False
        hq, base_b, base_a = (state.key_object_by_uid(uid) for uid in route._map_order)
        if any(obj is None or not obj.valid for obj in (hq, base_b, base_a)):
            self.ready = False
            return self.active
        strategy = route._strategy
        strategy.observe(state)
        frame = (state.episode, state.step)
        if frame != self._frame:
            consecutive = self._frame is not None and frame == (self._frame[0], self._frame[1] + 1)
            clear = strategy.occupants(state, base_a) == 0 and strategy.pressure(state, route, base_a) == 0
            self._clear = (self._clear + 1 if consecutive else 1) if clear else 0
            self._frame = frame
        hq_clear = (route.hq_secured and strategy.occupants(state, hq) == 0
                    and route.count(hq, 'percent') >= 1
                    and (hq.team is None or hq.team == state.team))
        if self.active:
            self.ready = hq_clear and self._clear >= self.CLEAR_FRAMES
            return True  # Never bounce back into B when A changes or is taken.
        if not hq_clear or not route.is_base_b(route.objective(state)):
            return False
        foot = self.capturers(state)
        self._peak = max(self._peak, len(foot))
        mobile = [u for u in foot if u.uid not in route._garrison]
        count = strategy.occupants(state, base_b)
        self._approached |= any(route.distance(u.position, base_b.position) <= 35000 for u in mobile)
        collapsed = (self._approached and len(foot) * 2 <= self._peak
                     and count is not None and count >= max(4, len(mobile)))
        if (not foot or (len(foot) > 4 and not collapsed) or self.armed_support(state, route)
                or count is None or count <= 0 or self._clear < self.CLEAR_FRAMES
                or route.owned(state, base_a)):
            return False
        self.active, self.ready = True, True
        self.source_uid, self.target_uid = base_b.uid, base_a.uid
        self.started = state.step
        self.reason = 'last_capturers' if len(foot) <= 4 else 'unsupported_collapse'
        suffix = [uid for uid in route.order[route.stage:] if uid != base_a.uid]
        route._reorder(state, (*route.order[:route.stage], base_a.uid, *suffix))
        # An entire surviving human/dog pair must not be trapped by a reserve
        # chosen for the original full-size force. Keep them on the same goal.
        route._garrison.clear()
        route._guard_slots.clear()
        logger.warning('B进攻止损改道A: step=%d reason=%s survivors=%d peak=%d B=%s',
                       state.step, self.reason, len(foot), self._peak, count)
        return True

    @classmethod
    def bounds(cls, route, obj):
        extent = route._footprint(obj) or (route.GARRISON_CORE, route.GARRISON_CORE)
        yaw = math.radians(float(obj.raw.get('rotation', {}).get('yaw', 0)))
        # World-aligned enclosing box covers the rotated capture box as well.
        x = abs(math.cos(yaw)) * extent[0] + abs(math.sin(yaw)) * extent[1] + cls.MARGIN
        y = abs(math.sin(yaw)) * extent[0] + abs(math.cos(yaw)) * extent[1] + cls.MARGIN
        return obj.position[0] - x, obj.position[0] + x, obj.position[1] - y, obj.position[1] + y

    @staticmethod
    def inside(point, bounds):
        x0, x1, y0, y1 = bounds
        return x0 < point[0] < x1 and y0 < point[1] < y1

    @staticmethod
    def crosses(left, right, bounds):
        # Slab intersection, including tiny corner crossings missed by sampling.
        lo, hi = 0.0, 1.0
        for axis, lower, upper in ((0, bounds[0], bounds[1]), (1, bounds[2], bounds[3])):
            delta = right[axis] - left[axis]
            if abs(delta) < 1e-9:
                if not lower < left[axis] < upper:
                    return False
            else:
                a, b = sorted(((lower - left[axis]) / delta, (upper - left[axis]) / delta))
                lo, hi = max(lo, a), min(hi, b)
                if hi <= lo:
                    return False
        return hi > lo and hi > 0 and lo < 1

    def detour(self, route, origin, target, base_b):
        bounds = self.bounds(route, base_b)
        origin, target = tuple(origin), tuple(target)
        x0, x1, y0, y1 = bounds
        # Use a recorded ground elevation for the B-side requests. These are
        # geometry-derived navigation requests, not sampled traversed roads.
        z = route.B_A[-1][2]
        corners = [(x0 - 100, y0 - 100, z), (x0 - 100, y1 + 100, z),
                   (x1 + 100, y0 - 100, z), (x1 + 100, y1 + 100, z)]
        prefix = []
        if self.inside(origin, bounds):
            # A unit already caught inside B first exits through its closest
            # boundary; all subsequent segments must stay outside the box.
            exits = [(x0 - 100, origin[1], z), (x1 + 100, origin[1], z),
                     (origin[0], y0 - 100, z), (origin[0], y1 + 100, z)]
            exit_point = min(exits, key=lambda p: route.distance(origin, p))
            prefix.append(exit_point)
            origin = exit_point
        nodes = [origin, target, *corners]
        heap, seen = [(0.0, 0, ())], set()
        while heap:
            cost, index, path = heapq.heappop(heap)
            if index in seen:
                continue
            if index == 1:
                return tuple(prefix) + tuple(nodes[i] for i in path)
            seen.add(index)
            for j, point in enumerate(nodes):
                if j not in seen and not self.crosses(nodes[index], point, bounds):
                    heapq.heappush(heap, (cost + route.distance(nodes[index], point), j, (*path, j)))
        return ()

    def point(self, state, agent, route):
        if (not self.active or not agent.alive or agent.entity_type not in self.FOOT | self.CARS
                or agent.uid in route._garrison):
            return None
        target = state.key_object_by_uid(self.target_uid)
        base_b = state.key_object_by_uid(self.source_uid)
        if target is None or base_b is None:
            return tuple(agent.position)
        if self.ready and route.owned(state, target) and route.inside_garrison(target, agent.position):
            # Being inside the public box alone did not produce occupation in
            # the recorded A approach. Stop only after actual capture evidence.
            return tuple(agent.position)
        if agent.uid not in self._paths:
            origin = tuple(agent.position)
            hq = state.key_object_by_uid(route.hq_uid)
            prefix = ()
            # Exit along the known front entrance instead of cutting from the
            # HQ courtyard across a building facade to the new destination.
            if hq is not None and route.inside_garrison(hq, origin):
                reverse = tuple(reversed(route.HQ_FRONT))
                nearest = min(range(len(reverse)), key=lambda i: route.distance(origin, reverse[i]))
                prefix = reverse[nearest:] + (route.LOWER[-1],)
            start = prefix[-1] if prefix else origin
            path = self.detour(route, start, target.position, base_b)
            self._paths[agent.uid] = prefix + path if path else (origin,)
            self._cursor[agent.uid] = 0
            self._safe[agent.uid] = (next((p for p in self._paths[agent.uid]
                                         if not self.inside(p, self.bounds(route, base_b))
                                         and route.distance(p, target.position) > 20000), origin)
                                     if self.inside(origin, self.bounds(route, base_b)) else origin)
        path, index = self._paths[agent.uid], self._cursor[agent.uid]
        if not self.ready:
            return self._safe[agent.uid]
        if route.inside_garrison(target, agent.position):
            # Reaching A by a different engine path must not recall a unit to
            # an old HQ/B waypoint. Continue to the capture goal until owned.
            self._cursor[agent.uid] = len(path) - 1
            return tuple(target.position)
        while index < len(path) - 1 and route.distance(agent.position, path[index]) <= 650:
            index += 1
        # A one-way bounded retry only relaxes a nearly reached waypoint; it
        # never skips a detour corner if that would cut across the B exclusion.
        distance = route.distance(agent.position, path[index])
        old = self._watch.get(agent.uid)
        if old is None or old[0] != index or distance < old[1] - 200 or state.step - old[3] > 2:
            old = (index, distance, 0, state.step)
        elif state.step != old[3]:
            old = (index, old[1], old[2] + 1, state.step)
        bounds = self.bounds(route, base_b)
        leaving_box = (index == 0 and self.inside(agent.position, bounds)
                       and not self.inside(path[index], bounds)
                       and index + 1 < len(path) and not self.inside(path[index + 1], bounds))
        if (old[2] >= 20 and distance <= 1600 and index < len(path) - 1
                and (leaving_box or not self.crosses(agent.position, path[index + 1], bounds))):
            index += 1
            old = (index, route.distance(agent.position, path[index]), 0, state.step)
        self._watch[agent.uid], self._cursor[agent.uid] = old, index
        if route.distance(agent.position, target.position) > 20000 and not self.inside(agent.position, self.bounds(route, base_b)):
            self._safe[agent.uid] = tuple(agent.position)
        return path[index]
