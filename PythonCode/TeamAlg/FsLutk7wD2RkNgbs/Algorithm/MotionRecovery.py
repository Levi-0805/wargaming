from __future__ import annotations

import logging
import math
import re

from cssim.protocol import Action

logger = logging.getLogger(__name__)


class MotionRecovery:
    """Measure progress toward a stable waypoint, not just distance walked.

    Preserve UE navigation goals and their map elevation. A short bounded
    direction escape is only a fallback; return to the waypoint after two
    frames so an aircraft cannot fly indefinitely beyond the map.
    """

    STALL_STEPS = 8
    ANGLES = (60, -60, 90, -90)

    def __init__(self):
        self._episode = None
        self._step = None
        self._previous = {}
        self._progress = {}
        self._escape = {}
        self._attempts = {}
        self._vehicle_escape = {}

    def _begin(self, state):
        if state.episode != self._episode or (self._step is not None and state.step < self._step):
            self._previous.clear()
            self._progress.clear()
            self._escape.clear()
            self._attempts.clear()
            self._vehicle_escape.clear()
        self._episode, self._step = state.episode, state.step

    def pause(self, state, agent):
        self._begin(state)
        self._previous[agent.uid] = (state.step, tuple(agent.position), False)
        self._progress.pop(agent.uid, None)
        self._escape.pop(agent.uid, None)
        self._vehicle_escape.pop(agent.uid, None)

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
        position = tuple(float(v) for v in agent.position)
        previous = self._previous.get(agent.uid)
        goal = tuple(action.points[:3]) if action.points is not None else None
        airborne = agent.entity_type in {"BP_Base_UAV_C", "BP_Helicopter_C"}
        distance = math.dist(position, goal) if goal is not None else None
        intended = action.command == "Moving" and (distance is None or distance > 300)
        self._previous[agent.uid] = (state.step, position, intended)
        if not intended or not agent.alive or agent.communication_ok() is False:
            self._progress.pop(agent.uid, None)
            self._escape.pop(agent.uid, None)
            self._vehicle_escape.pop(agent.uid, None)
            return action
        if previous and previous[0] == state.step:
            return action
        tracked = self._progress.get(agent.uid)
        if goal is not None:
            if tracked is None or tracked[0] is None or math.dist(goal, tracked[0]) > 600:
                tracked = (goal, state.step, distance)
            elif distance < tracked[2] - 200:
                tracked = (goal, state.step, distance)
            # Going away and returning to the same distance is not progress.
        elif tracked is None or (previous and math.dist(position, previous[1]) >= 50):
            tracked = (None, state.step, 0)
        self._progress[agent.uid] = tracked
        vehicle_escape = self._vehicle_escape.get(agent.uid)
        if vehicle_escape:
            start, reverse, side_point, original_goal = vehicle_escape
            if goal != original_goal:
                del self._vehicle_escape[agent.uid]
            elif state.step < start + 2:
                return Action.move(reverse)
            elif state.step < start + 5:
                return Action.move_at(side_point)
            else:
                del self._vehicle_escape[agent.uid]
                self._progress[agent.uid] = (goal, state.step, distance or 0)
                return action
        escape = self._escape.get(agent.uid)
        if escape and state.step < escape[0]:
            return Action.move(escape[1])
        if state.step - tracked[1] >= self.STALL_STEPS:
            attempt = self._attempts.get(agent.uid, 0)
            if agent.entity_type == "BP_MNWS_Vehicle_Armored_C":
                # Wheeled actors can be nose-first against an obstacle. A
                # sideways heading alone did not move them in the latest run.
                # Request retreat relative to measured yaw before a bounded
                # 25m side waypoint, then resume the unchanged route.
                yaw = math.radians(float(agent.yaw))
                forward = (math.cos(yaw), math.sin(yaw))
                reverse = self._direction((-forward[0], -forward[1], 0), 0, False)
                side = 1 if attempt % 2 == 0 else -1
                point = (position[0] - forward[0] * 2000 - forward[1] * side * 2500,
                         position[1] - forward[1] * 2000 + forward[0] * side * 2500, position[2])
                self._vehicle_escape[agent.uid] = (state.step, reverse, point, goal)
                self._attempts[agent.uid] = attempt + 1
                logger.warning("轮式车无进展，后退再绕障: uid=%d step=%d attempt=%d direction=%s",
                               agent.uid, state.step, attempt + 1, reverse)
                return Action.move(reverse)
            direction = self._direction(self._vector(agent, action), self.ANGLES[attempt % len(self.ANGLES)], airborne)
            self._attempts[agent.uid] = attempt + 1
            self._escape[agent.uid] = (state.step + 2, direction)
            self._progress[agent.uid] = (goal, state.step + 2, distance or 0)
            logger.warning("航点无净进展: uid=%d step=%d distance_m=%.1f attempt=%d escape=%s",
                           agent.uid, state.step, (distance or 0) / 100, attempt + 1, direction)
            return Action.move(direction)
        return action
