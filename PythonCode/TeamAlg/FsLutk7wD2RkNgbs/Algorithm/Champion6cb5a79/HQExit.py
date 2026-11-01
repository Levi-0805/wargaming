from __future__ import annotations

from cssim.protocol import Action
from .AssaultRoute import AssaultRoute


class HQExit:
    """Recover observed stalled departures along the reached front corridor.

    Ordinary successful departures keep their existing route. Navigation
    attempts, rather than time spent firing or waiting, start this fallback.
    """

    FOOT = {'BP_BaseSoldier_C', 'BP_RoboDog_C'}
    PATH = tuple(reversed(AssaultRoute.HQ_FRONT[3:]))
    ALTERNATE = tuple(reversed(AssaultRoute.HQ_FRONT_ALTERNATE[2:]))
    STALL = 8
    WAYPOINT_TRIALS = 16

    def __init__(self):
        self._frame = None
        self._probe = {}
        self._paths = {}
        self._cursor = {}
        self._watch = {}
        self._alternate = set()
        self._finished = set()

    def begin(self, state, route):
        frame = (state.episode, state.step)
        if (self._frame is None or frame[0] != self._frame[0] or frame[1] < self._frame[1]
                or not route.hq_secured):
            self.__init__()
        self._frame = frame

    def choose(self, state, agent, route, ordinary_point):
        self.begin(state, route)
        hq = state.key_object_by_uid(route.hq_uid) if route.hq_uid is not None else None
        if (not route.known_map or not route.hq_secured or hq is None
                or route.is_hq(route.objective(state)) or agent.entity_type not in self.FOOT
                or not agent.alive or agent.communication_ok() is False
                or agent.uid in route._garrison or route._remnant.active):
            self._probe.pop(agent.uid, None)
            return None
        uid, position = agent.uid, tuple(agent.position)
        if uid not in self._paths:
            if uid in self._finished or not route.inside_garrison(hq, position) or ordinary_point is None:
                return None
            # Only navigation out of the HQ qualifies. Guard-slot movement or
            # a unit still following an HQ ingress leg is not a failed exit.
            if route.distance(ordinary_point, hq.position) < 8000:
                return None
            old = self._probe.get(uid)
            if (old is None or state.step - old[2] > 2
                    or route.distance(position, old[0]) > 200):
                old = (position, 0, state.step)
            elif state.step != old[2]:
                old = (old[0], old[1] + 1, state.step)
            self._probe[uid] = old
            if old[1] < self.STALL:
                return None
            path = self.PATH
            self._paths[uid] = path
            self._cursor[uid] = min(range(len(path)), key=lambda i: route.distance(position, path[i]))
        path, cursor = self._paths[uid], self._cursor[uid]
        while route.distance(position, path[cursor]) <= 650:
            if cursor == len(path) - 1:
                return self._finish(agent, route)
            cursor += 1
        distance = route.distance(position, path[cursor])
        old = self._watch.get(uid)
        if (old is None or old[0] != cursor or state.step - old[3] > 2
                or distance < old[1] - 200):
            old = (cursor, distance, 0, state.step)
        elif state.step != old[3]:
            old = (cursor, old[1], old[2] + 1, state.step)
        self._watch[uid] = old
        if old[2] >= self.WAYPOINT_TRIALS:
            self._watch.pop(uid, None)
            if uid not in self._alternate:
                blocked = path[cursor]
                path = self.ALTERNATE
                self._alternate.add(uid)
                self._paths[uid] = path
                cursor = min(range(len(path)), key=lambda i: route.distance(blocked, path[i]))
            elif cursor < len(path) - 1:
                cursor += 1  # Try one next recorded corner, never skip the facade.
            else:
                return self._finish(agent, route)
        self._cursor[uid] = cursor
        return Action.move_at(path[cursor])

    def _finish(self, agent, route):
        self._paths.pop(agent.uid, None)
        self._watch.pop(agent.uid, None)
        self._finished.add(agent.uid)
        # Resume the ordinary departure leg, not the old courtyard cursor.
        leg = min(route.stage, route.order.index(route.hq_uid) + 1)
        route._cursor[agent.uid] = (leg, 0)
        route._waypoint_watch.pop(agent.uid, None)
        return None
