from __future__ import annotations

import logging
import math

from cssim.protocol import Action
from .AssaultRoute import AssaultRoute

logger = logging.getLogger(__name__)


class FlightNavigation:
    """Send the actual destination first, without artificial hover waypoints.

    The 20261014_211207 aircraft reached the first 250m waypoint but mostly
    failed to adopt later goals, despite 34-41 Idle resets per round. Avoid
    those intermediate goals; a short directional Guard is a different,
    supported action transition when the objective changes or progress stops.
    Its physical effect still requires verification in a fresh simulation.
    """

    ARRIVAL = 3000.0
    STALL_STEPS = 8
    RESET_STEPS = 2

    def __init__(self):
        self._frame = None
        self._legs = {}
        self._pending = {}
        self.resets = {}
        self._stall_attempts = {}
        self._held = {}

    def cancel(self, agent):
        self._legs.pop(agent.uid, None)
        self._pending.pop(agent.uid, None)

    def _point(self, agent, goal):
        return tuple(float(value) for value in goal)

    @staticmethod
    def _face(agent, point):
        angle = math.atan2(point[1] - agent.position[1], point[0] - agent.position[0])
        directions = ('+X', '+X+Y', '+Y', '-X+Y', '-X', '-X-Y', '-Y', '+X-Y')
        return Action.guard(directions[round(angle / (math.pi / 4)) % 8])

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
        return self._face(agent, goal)

    def apply(self, state, agent, goal, objective, max_stall_resets=None):
        frame = (state.episode, state.step)
        if self._frame is None or frame[0] != self._frame[0] or frame[1] < self._frame[1]:
            self._legs.clear()
            self._pending.clear()
            self.resets.clear()
            self._stall_attempts.clear()
            self._held.clear()
        self._frame = frame
        if not agent.alive or agent.communication_ok() is False:
            self.cancel(agent)
            return Action.idle()
        if agent.uid in self._held and self._held[agent.uid] == objective:
            return Action.guard_position(agent.position)
        self._held.pop(agent.uid, None)
        if agent.uid not in self._stall_attempts or self._stall_attempts[agent.uid][0] != objective:
            self._stall_attempts[agent.uid] = (objective, 0)
        pending = self._pending.get(agent.uid)
        if pending is not None:
            if state.step - pending[0] < self.RESET_STEPS:
                return self._face(agent, pending[1])
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
            _, attempts = self._stall_attempts[agent.uid]
            if max_stall_resets is not None and attempts >= max_stall_resets:
                self.cancel(agent)
                self._held[agent.uid] = objective
                logger.warning('无人机观察点暂不可达，原地观察等待下个前进点: uid=%d step=%d', agent.uid, state.step)
                return Action.guard_position(agent.position)
            self._stall_attempts[agent.uid] = (objective, attempts + 1)
            return self._replace(state, agent, goal, objective, 'stalled')
        return Action.move_at(point)
