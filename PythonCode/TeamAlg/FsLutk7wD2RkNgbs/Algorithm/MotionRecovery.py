from __future__ import annotations

import logging
import math
import re

from cssim.protocol import Action


logger = logging.getLogger(__name__)


class MotionRecovery:
    """Detect failed movement from actual positions, then try direction control.

    Deliberate waiting, attacking and parent changes are not path failures.
    This is a bounded fallback, not a substitute for the platform's navigation.
    """

    STALL_STEPS = 8
    GROUND_STEP = 2000.0
    ANGLES = (0, 60, -60, 120, -120, 180)

    def __init__(self):
        self._episode = None
        self._step = None
        self._previous = {}
        self._still = {}
        self._escape = {}
        self._attempts = {}

    def _begin(self, state):
        if state.episode != self._episode or (self._step is not None and state.step < self._step):
            self._previous.clear()
            self._still.clear()
            self._escape.clear()
            self._attempts.clear()
        self._episode, self._step = state.episode, state.step

    def pause(self, state, agent):
        self._begin(state)
        # A special action deliberately consumes this frame. Preserve a past
        # motion failure, but do not count this pause as another failed move.
        self._previous[agent.uid] = (state.step, tuple(agent.position), False)

    @staticmethod
    def _vector(agent, action):
        if action.points is not None:
            return tuple(action.points[i] - float(agent.position[i]) for i in range(3))
        vector = [0.0, 0.0, 0.0]
        for sign, axis in re.findall(r"([+-])([XYZ])", action.direction or ""):
            vector["XYZ".index(axis)] = 1000.0 if sign == "+" else -1000.0
        return tuple(vector)

    @staticmethod
    def _direction(vector, angle, airborne):
        dx, dy, dz = vector
        radians = math.radians(angle)
        x, y = dx * math.cos(radians) - dy * math.sin(radians), dx * math.sin(radians) + dy * math.cos(radians)
        threshold = max(abs(x), abs(y)) * 0.35
        direction = ""
        if abs(x) > max(0.01, threshold):
            direction += "+X" if x > 0 else "-X"
        if abs(y) > max(0.01, threshold):
            direction += "+Y" if y > 0 else "-Y"
        if airborne and abs(dz) > 400:
            direction += "+Z" if dz > 0 else "-Z"
        return direction or "+X"

    def apply(self, state, agent, action):
        self._begin(state)
        previous = self._previous.get(agent.uid)
        position = tuple(float(value) for value in agent.position)
        vector = self._vector(agent, action) if action.command == "Moving" else (0, 0, 0)
        intended = action.command == "Moving" and (action.points is None or math.dist(position, action.points) > 300)
        if previous and previous[0] < state.step:
            if previous[2] and intended and math.dist(position, previous[1]) < 50:
                self._still[agent.uid] = self._still.get(agent.uid, 0) + 1
            elif math.dist(position, previous[1]) >= 50 or not intended:
                self._still[agent.uid] = 0
        if not intended or not agent.alive or agent.communication_ok() is False:
            self._previous[agent.uid] = (state.step, position, intended)
            return action

        airborne = agent.entity_type in {"BP_Base_UAV_C", "BP_Helicopter_C"}
        escape = self._escape.get(agent.uid)
        if escape and state.step < escape[0]:
            action = Action.move(escape[1])
        elif self._still.get(agent.uid, 0) >= self.STALL_STEPS:
            attempt = self._attempts.get(agent.uid, 0)
            angle = self.ANGLES[attempt % len(self.ANGLES)]
            direction = self._direction(vector, angle, airborne)
            self._attempts[agent.uid] = attempt + 1
            self._escape[agent.uid] = (state.step + (2 if airborne else 6), direction)
            self._still[agent.uid] = 0
            action = Action.move(direction)
            logger.info("移动无进展，切换方向: uid=%d step=%d attempt=%d direction=%s",
                        agent.uid, state.step, attempt + 1, direction)
        elif not airborne and action.points is not None:
            # Keep ground waypoints close and at the actor's own elevation.
            # The navigation mesh still determines the path between waypoints.
            dx, dy, _ = vector
            distance = math.hypot(dx, dy)
            if distance > self.GROUND_STEP:
                scale = self.GROUND_STEP / distance
                action = Action.move_at((position[0] + dx * scale, position[1] + dy * scale, position[2]))
        self._previous[agent.uid] = (state.step, position, intended)
        return action
