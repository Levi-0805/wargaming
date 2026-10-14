from __future__ import annotations

import logging

from cssim.protocol import Action
from .AssaultRoute import AssaultRoute
from .UrbanContact import UrbanContact

logger = logging.getLogger(__name__)


class HQEntry:
    """Escalate facade oscillation to an observed, ordered courtyard entrance.

    Static geometry only: 20261009_003121 round 1, friendly dog 346 steps
    108-120 and friendly soldier 101 steps 151-155. Both actually reached the
    courtyard. Enemy locations/UIDs from that recording are never reused.
    """

    PATH = ((-36335, 30318, 6425), (-37026, 31271, 6425),
            (-37200, 32593, 6425), (-37334, 33617, 6425),
            (-37497, 34856, 6425), (-37649, 36093, 6425),
            (-38820, 36237, 6429), (-40107, 36086, 6425),
            (-40591, 34959, 6425), (-41108, 33820, 6425),
            (-41625, 32682, 6425), (-40105, 32287, 6429),
            (-39043, 31926, 6429))
    STALL_STEPS = 18
    ARRIVAL = 650

    def __init__(self):
        self._frame = None
        self._watch = {}
        self._routes = {}
        self._trials = {}
        self._completed = set()

    def choose(self, state, agent, route, urban, contacts):
        frame = (state.episode, state.step)
        if self._frame is None or frame[0] != self._frame[0] or frame[1] < self._frame[1]:
            self._watch.clear()
            self._routes.clear()
            self._trials.clear()
            self._completed.clear()
        self._frame = frame
        hq = state.key_object_by_uid(route.order[0]) if route.order else None
        guard = route.garrison_for(state, agent)
        eligible = (hq is not None and route.known_map and agent.entity_type in UrbanContact.GROUND
                    and agent.alive and agent.communication_ok() is not False
                    and (route.stage == 0 or guard is not None and guard.uid == hq.uid)
                    and AssaultRoute.distance(agent.position, hq.position) < 18000
                    and (route.enemy_count(state, hq) > 0 or route.count(hq, "percent") < 1))
        if not eligible:
            self._watch.pop(agent.uid, None)
            self._routes.pop(agent.uid, None)
            return None
        ground = [enemy for enemy in contacts if enemy.entity_type in UrbanContact.GROUND]
        effective = any(urban.effective(state, agent, enemy) for enemy in ground)
        distance = AssaultRoute.distance(agent.position, hq.position)
        watch = self._watch.get(agent.uid)
        if watch is None or effective or distance < watch[0] - 1500:
            watch = (distance, state.step)
            self._watch[agent.uid] = watch
        if effective:
            return None  # Effective fire interrupts travel without rewinding it.
        cursor = self._routes.get(agent.uid)
        if cursor is None:
            if agent.uid in self._completed or state.step - watch[1] < self.STALL_STEPS:
                return None
            reached = [i for i, point in enumerate(self.PATH)
                       if AssaultRoute.distance(agent.position, point) <= self.ARRIVAL]
            cursor = max(reached) if reached else 0
            self._routes[agent.uid] = cursor
            urban.cancel(agent)
            logger.info("外墙接敌无进展，改走已通行入口: uid=%d step=%d waypoint=%d",
                        agent.uid, state.step, cursor)
        # A fresh local contact gets a bounded firing trial, rather than every
        # shared report cancelling the entrance route again.
        local = {enemy.uid for enemy in state.perceived_opponents(agent)}
        ammo = urban.ammo(agent)
        if ground and (ammo is None or ammo > 0) and any(e.uid in local for e in ground):
            trial = self._trials.get(agent.uid, -100)
            if state.step - trial >= 12:
                trial = state.step
                self._trials[agent.uid] = trial
            if state.step - trial < 3:
                return None
        while AssaultRoute.distance(agent.position, self.PATH[cursor]) <= self.ARRIVAL:
            if cursor == len(self.PATH) - 1:
                self._completed.add(agent.uid)
                self._routes.pop(agent.uid, None)
                urban.cancel(agent)
                if guard is not None:
                    route._guard_slots[(agent.uid, guard.uid)] = route.clamp_guard_point(guard, agent.position)
                return None
            cursor += 1
        self._routes[agent.uid] = cursor
        # A full, fixed waypoint lets UE navigation and MotionRecovery finish
        # the corner. Target changes and short local circles cannot reset it.
        return Action.move_at(self.PATH[cursor])
