from __future__ import annotations

import math
from cssim.protocol import Action
from .AssaultRoute import AssaultRoute
from .HQEntry import HQEntry


class ColumnAdvance:
    """Keep the moving main body together without tethering it to stuck units.

    Only forward navigation is gated. Trailing movement, local fire, commander
    repair and retreats remain available. Progress is measured along the road,
    so a corner does not make a nearby unit appear to lead the column.
    """

    FOOT = {'BP_BaseSoldier_C', 'BP_RoboDog_C'}
    VEHICLES = HQEntry.VEHICLES
    FOOT_LEAD = 3500.0
    VEHICLE_LEAD = 5000.0
    RESERVE_LAG = 1500.0
    STALLED_MOVES = 24

    def __init__(self):
        self._frame = None
        self._objective = None
        self._watch = {}
        self._moved = set()
        self.progress = {}
        self.excluded = set()
        self.held = set()
        self.foot = []
        self.anchor = None
        self.path = ()
        self._hp = {}
        self.under_fire = set()

    @staticmethod
    def project(point, path):
        return ColumnAdvance.locate(point, path)[0]

    @staticmethod
    def locate(point, path):
        if len(path) < 2:
            return 0.0, 0.0
        distance, progress, arc = float('inf'), 0.0, 0.0
        for a, b in zip(path, path[1:]):
            dx, dy = b[0]-a[0], b[1]-a[1]
            length = math.hypot(dx, dy)
            if length < 1:
                continue
            fraction = max(0., min(1., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/(length*length)))
            offset = math.hypot(point[0]-a[0]-fraction*dx, point[1]-a[1]-fraction*dy)
            if offset < distance:
                distance, progress = offset, arc + fraction*length
            arc += length
        return progress, distance

    def begin(self, state, route):
        frame = (state.episode, state.step)
        if self._frame == frame:
            return
        obj = route.objective(state)
        uid = obj.uid if obj else None
        if (self._frame is None or self._frame[0] != state.episode or state.step < self._frame[1]
                or self._objective != uid):
            self.__init__()
        consecutive = self._frame is not None and state.step == self._frame[1] + 1
        self._frame, self._objective = frame, uid
        self.held.clear()
        self.under_fire = {u.uid for u in state.agents if u.uid in self._hp and u.hp < self._hp[u.uid] - .01}
        self._hp = {u.uid: float(u.hp) for u in state.agents if u.alive}
        if obj is None:
            self.path = ()
            self.progress.clear()
            self.excluded.clear()
            self.foot = []
            self.anchor = None
            self._watch.clear()
            self._moved.clear()
            return
        units = [u for u in state.agents if u.alive and u.entity_type in self.FOOT | self.VEHICLES
                 and u.communication_ok() is not False
                 and (u.uid not in route._garrison or route._garrison[u.uid] == uid)]
        if not self.path:
            self.path = self._path(route, obj, units)
        measured = {u.uid: self.locate(u.position, self.path) if len(self.path) >= 2 else
                    (-AssaultRoute.distance(u.position, obj.position), 0.0) for u in units}
        self.progress = {uid: value[0] for uid, value in measured.items()}
        self.excluded = set()
        for unit in units:
            value = self.progress[unit.uid]
            offset = measured[unit.uid][1]
            old = self._watch.get(unit.uid)
            # Joining the road can temporarily run opposite its forward arc.
            # Getting closer to it is real progress, not a reason to discard
            # an approaching tail from the formation reference.
            if old is None or not consecutive or value > old[0] + 200 or offset < old[2] - 200:
                watch = (value, 0, offset)
            else:
                watch = (old[0], old[1] + int(unit.uid in self._moved), old[2])
            self._watch[unit.uid] = watch
            if watch[1] >= self.STALLED_MOVES:
                self.excluded.add(unit.uid)
        self._moved.clear()
        self.foot = [u for u in units if u.entity_type in self.FOOT and u.uid not in self.excluded]
        values = sorted(self.progress[u.uid] for u in self.foot)
        self.anchor = values[max(0, math.ceil(.35*len(values))-1)] if values else None

    @staticmethod
    def _path(route, obj, units):
        if not route.known_map:
            return route._routes[route.stage]
        if route.is_hq(obj):
            return route.LOWER + HQEntry.PATH
        # Replanning stores only the destination in _routes. Preserve a shared
        # road reference on secondary legs too, including an A -> B return.
        corridor = route.LOWER + route.HQ_B + route.B_A
        end = (len(route.LOWER)-1, len(route.LOWER)+len(route.HQ_B)-1,
               len(corridor)-1)[route._map_order.index(obj.uid)]
        body = [u for u in units if u.entity_type in ColumnAdvance.FOOT] or units
        starts = sorted(min(range(len(corridor)), key=lambda i: AssaultRoute.distance(u.position, corridor[i]))
                        for u in body)
        start = starts[len(starts)//2] if starts else end
        indices = range(start, end+1) if start <= end else range(start, end-1, -1)
        return tuple(corridor[i] for i in indices) + (tuple(obj.position),)

    def forward(self, agent, action):
        if action.command != 'Moving':
            return False
        if action.points is None:
            # Supported short directional obstruction escapes stay available.
            return False
        current = self.progress.get(agent.uid)
        if current is None:
            return False
        if len(self.path) >= 2:
            return self.project(action.points[:3], self.path) > current + 150
        return False

    def engaged(self, state, agent):
        return agent.uid in self.under_fire or any(
            enemy.alive and enemy.entity_type in self.FOOT | self.VEHICLES
            for enemy in state.perceived_opponents(agent))

    def apply(self, state, agent, action, route, formation, released=()):
        anchor = self.anchor
        # The following wave intentionally holds outside the corner. It must
        # not tether the wave already released through it to that same spot.
        if agent.uid in released or agent.uid in formation.front:
            cohort = sorted(self.progress[u.uid] for u in self.foot if u.uid in released)
            if cohort:
                anchor = cohort[max(0, math.ceil(.35*len(cohort))-1)]
        obj = route.objective(state)
        supporting_entry = (obj is not None and agent.uid in formation.front
                            and AssaultRoute.distance(agent.position, obj.position) <= 18000
                            and (bool(released) or any(route.inside_garrison(obj, u.position) for u in self.foot)))
        if (self.anchor is not None and agent.uid in self.progress
                and self.forward(agent, action)
                and not self.engaged(state, agent) and not supporting_entry
                and not route.inside_garrison(obj, agent.position)):
            if agent.entity_type in self.FOOT:
                limit = anchor + self.FOOT_LEAD
            elif agent.uid in formation.reserve and formation.reserve[agent.uid] != self._objective:
                limit = anchor - self.RESERVE_LAG
            else:
                limit = anchor + self.VEHICLE_LEAD
            if self.progress[agent.uid] > limit:
                self.held.add(agent.uid)
                return Action.guard_position(agent.position)
        if action.command == 'Moving' and agent.uid in self.progress:
            self._moved.add(agent.uid)
        return action
