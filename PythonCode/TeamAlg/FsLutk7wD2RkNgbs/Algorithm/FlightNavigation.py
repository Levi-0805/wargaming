from __future__ import annotations

import logging

from cssim.protocol import Action
from .AssaultRoute import AssaultRoute

logger = logging.getLogger(__name__)


class FlightNavigation:
    """Own a fixed flight leg and explicitly transition before replacing it.

    The 20261014_164549 aircraft hovered near their first requested waypoint
    while later Moving coordinates/directions had no observed effect. Idle is
    a supported action transition, not a verified UE task-cancellation API.
    Physical effectiveness must be checked in a fresh simulation.
    """

    LEG_LENGTH = 25000.0
    ARRIVAL = 3000.0
    STALL_STEPS = 8

    def __init__(self):
        self._frame = None
        self._legs = {}
        self._pending = {}
        self.resets = {}

    def cancel(self, agent):
        self._legs.pop(agent.uid, None)
        self._pending.pop(agent.uid, None)

    def _point(self, agent, goal):
        distance = AssaultRoute.distance(agent.position, goal)
        ratio = min(1.0, self.LEG_LENGTH / max(distance, 1))
        return (float(agent.position[0]) + (goal[0] - agent.position[0]) * ratio,
                float(agent.position[1]) + (goal[1] - agent.position[1]) * ratio, float(goal[2]))

    def _start(self, state, agent, point, objective):
        distance = AssaultRoute.distance(agent.position, point)
        self._legs[agent.uid] = [point, objective, distance, state.step]
        return Action.move_at(point)

    def _replace(self, state, agent, goal, objective, reason):
        self._legs.pop(agent.uid, None)
        self._pending[agent.uid] = (state.step, self._point(agent, goal), objective)
        self.resets[agent.uid] = self.resets.get(agent.uid, 0) + 1
        logger.info('无人机航段切换: uid=%d step=%d reason=%s reset=%d',
                    agent.uid, state.step, reason, self.resets[agent.uid])
        return Action.idle()  # Exactly this frame; never wait for another unit.

    def apply(self, state, agent, goal, objective):
        frame = (state.episode, state.step)
        if self._frame is None or frame[0] != self._frame[0] or frame[1] < self._frame[1]:
            self._legs.clear()
            self._pending.clear()
            self.resets.clear()
        self._frame = frame
        if not agent.alive or agent.communication_ok() is False:
            self.cancel(agent)
            return Action.idle()
        pending = self._pending.get(agent.uid)
        if pending is not None:
            if state.step == pending[0]:
                return Action.idle()
            self._pending.pop(agent.uid)
            point = pending[1] if pending[2] == objective else self._point(agent, goal)
            return self._start(state, agent, point, objective)
        leg = self._legs.get(agent.uid)
        if leg is None:
            if AssaultRoute.distance(agent.position, goal) <= self.ARRIVAL:
                return Action.guard_position((float(agent.position[0]), float(agent.position[1]), goal[2]))
            return self._start(state, agent, self._point(agent, goal), objective)
        point, old_objective, best, since = leg
        if objective != old_objective:
            return self._replace(state, agent, goal, objective, 'objective')
        distance = AssaultRoute.distance(agent.position, point)
        if distance <= self.ARRIVAL:
            if AssaultRoute.distance(agent.position, goal) > self.ARRIVAL:
                return self._replace(state, agent, goal, objective, 'arrival')
            self.cancel(agent)
            return Action.guard_position((float(agent.position[0]), float(agent.position[1]), goal[2]))
        if distance < best - 200:
            leg[2:] = distance, state.step
        elif state.step - since >= self.STALL_STEPS:
            return self._replace(state, agent, goal, objective, 'stalled')
        return Action.move_at(point)
