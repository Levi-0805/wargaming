from __future__ import annotations

import logging
import math
from cssim.protocol import Action
from .AssaultRoute import AssaultRoute
from .ColumnAdvance import ColumnAdvance

logger = logging.getLogger(__name__)


class EntryWaves:
    """Assemble at the reached road before the HQ corner, then release groups.

    Never recall units already beyond assembly, override fire, or require a
    stalled/disconnected tail to join. Waiting units keep separate reached
    positions rather than being ordered to a single coordinate.
    """

    RADIUS = 5500.0
    FOLLOW_MIN = 8
    FOLLOW_PROGRESS = 2500.0
    BLOCKED_WAVE_STEPS = 24

    def __init__(self):
        self._frame = None
        self.released = set()
        self.waves = []
        self._last = None
        self._waiting = set()
        self.held = set()

    def begin(self, state, route, column):
        frame = (state.episode, state.step)
        if frame == self._frame:
            return
        if self._frame is None or frame[0] != self._frame[0] or frame[1] < self._frame[1]:
            self.__init__()
        self._frame = frame
        self._waiting = set()
        self.held.clear()
        obj = route.objective(state)
        if not route.known_map or not route.is_hq(obj) or route.hq_secured:
            self.released.clear()
            self.waves.clear()
            self._last = None
            return
        rally = route.LOWER[-1]
        rally_progress = column.project(rally, column.path)
        units = [u for u in column.foot if u.uid not in self.released]
        # A late-start observation or a unit already across the entrance must
        # never be called back to form a new wave.
        passed = [u for u in units if route.inside_garrison(obj, u.position)
                  or column.progress[u.uid] > rally_progress + self.RADIUS]
        self.released.update(u.uid for u in passed)
        units = [u for u in units if u.uid not in self.released]
        ready = [u for u in units if AssaultRoute.distance(u.position, rally) <= self.RADIUS]
        self._waiting = {u.uid for u in ready}
        if not ready:
            return
        first = not self.waves
        needed = (min(len(units), max(3, math.ceil(.65*len(units)))) if first
                  else min(self.FOLLOW_MIN, len(units)))
        if len(ready) < needed:
            return  # The tail still has unrestricted movement toward assembly.
        if self._last is not None:
            prior = [u for u in state.agents if u.alive and u.uid in self._last[1]
                     and u.communication_ok() is not False]
            advanced = sum(route.inside_garrison(obj, u.position)
                           or column.progress.get(u.uid, 0) >= self._last[1][u.uid] + self.FOLLOW_PROGRESS
                           for u in prior)
            if prior and advanced < math.ceil(.6*len(prior)) and state.step - self._last[0] < self.BLOCKED_WAVE_STEPS:
                return
        # Mix soldiers and dogs within each release rather than sending every
        # faster dog ahead of its human controllers.
        soldiers = sorted((u for u in ready if u.entity_type == 'BP_BaseSoldier_C'),
                          key=lambda u: (-column.progress[u.uid], u.uid))
        dogs = sorted((u for u in ready if u.entity_type == 'BP_RoboDog_C'),
                      key=lambda u: (-column.progress[u.uid], u.uid))
        mixed = []
        while soldiers or dogs:
            mixed.extend(soldiers[:2])
            soldiers = soldiers[2:]
            mixed.extend(dogs[:1])
            dogs = dogs[1:]
        # The body has already assembled. Splitting those ready units into
        # eight-unit packets recreated the fragmented assault we meant to fix.
        wave = mixed
        ids = {u.uid for u in wave}
        self.released.update(ids)
        self._waiting.difference_update(ids)
        self._last = (state.step, {u.uid: column.progress[u.uid] for u in wave})
        self.waves.append((state.step, tuple(sorted(ids))))
        logger.info('指挥所入口成组放行: step=%d units=%s waiting=%d', state.step, sorted(ids), len(self._waiting))

    def apply(self, state, agent, action, route, column):
        if (agent.uid in self._waiting and column.forward(agent, action)
                and not column.engaged(state, agent)):
            self.held.add(agent.uid)
            return Action.guard_position(agent.position)
        return action
