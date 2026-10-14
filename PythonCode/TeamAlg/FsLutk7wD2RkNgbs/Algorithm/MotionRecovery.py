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
        self._turns = {}
        self._turn_retry = {}
        self._stationary = {}

    def _begin(self, state):
        if state.episode != self._episode or (self._step is not None and state.step < self._step):
            self._previous.clear()
            self._progress.clear()
            self._escape.clear()
            self._attempts.clear()
            self._vehicle_escape.clear()
            self._turns.clear()
            self._turn_retry.clear()
            self._stationary.clear()
        self._episode, self._step = state.episode, state.step

    def pause(self, state, agent):
        self._begin(state)
        self._previous[agent.uid] = (state.step, tuple(agent.position), False)
        self._progress.pop(agent.uid, None)
        self._escape.pop(agent.uid, None)
        self._vehicle_escape.pop(agent.uid, None)
        self._turns.pop(agent.uid, None)
        self._stationary.pop(agent.uid, None)

    def _align_vehicle(self, state, agent, goal):
        """Request a bounded heading change without inventing lateral roads.

        The old 25m arcs repeatedly replaced the road goal while cars reversed
        at ~2m/frame. Try the supported Guard direction for two frames, then
        leave the actual road waypoint alone for at least eighteen frames.
        """
        if agent.entity_type != "BP_MNWS_Vehicle_Armored_C" or goal is None:
            return None
        dx, dy = goal[0] - agent.position[0], goal[1] - agent.position[1]
        error = (math.degrees(math.atan2(dy, dx)) - agent.yaw + 180) % 360 - 180
        turn = self._turns.get(agent.uid)
        if math.hypot(dx, dy) < 1000 or abs(error) < 70:
            self._turns.pop(agent.uid, None)
            return None
        if turn and (state.step - turn[0] >= 2 or math.dist(goal, turn[1]) > 600):
            self._turns.pop(agent.uid, None)
            return None
        if turn is None and (abs(error) < 105 or state.step < self._turn_retry.get(agent.uid, -1)):
            return None
        if turn is None:
            direction = self._direction((dx, dy, 0), 0, airborne=False)
            turn = (state.step, goal, direction)
            self._turns[agent.uid] = turn
            self._turn_retry[agent.uid] = state.step + 20
            logger.info('轮式车短时校正朝向: uid=%d step=%d direction=%s', agent.uid, state.step, direction)
        return Action.guard(turn[2])

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
        # Turning and changing short combat destinations must not erase proof
        # that a vehicle has failed to move. Count movement requests, not time
        # spent deliberately firing or guarding.
        stationary = self._stationary.get(agent.uid)
        if not agent.alive or agent.communication_ok() is False:
            self._stationary.pop(agent.uid, None)
        elif agent.entity_type == "BP_MNWS_Vehicle_Armored_C":
            if stationary is None or math.dist(position, stationary[0]) > 200:
                stationary = (position, 0)
            elif intended and (previous is None or previous[0] != state.step):
                stationary = (stationary[0], stationary[1] + 1)
            self._stationary[agent.uid] = stationary
        if not intended or not agent.alive or agent.communication_ok() is False:
            self._progress.pop(agent.uid, None)
            self._escape.pop(agent.uid, None)
            self._vehicle_escape.pop(agent.uid, None)
            self._turns.pop(agent.uid, None)
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
            if ((goal is None) != (original_goal is None)
                    or (goal is not None and math.dist(goal, original_goal) > 600)):
                del self._vehicle_escape[agent.uid]
            elif state.step < start + 2:
                return Action.move(reverse)
            elif state.step < start + 5:
                return Action.move_at(side_point)
            else:
                del self._vehicle_escape[agent.uid]
                self._progress[agent.uid] = (goal, state.step, distance or 0)
                self._stationary[agent.uid] = (position, 0)
                return action
        escape = self._escape.get(agent.uid)
        if escape and state.step < escape[0]:
            return Action.move(escape[1])
        if (state.step - tracked[1] >= self.STALL_STEPS
                or (stationary is not None and stationary[1] >= self.STALL_STEPS)):
            attempt = self._attempts.get(agent.uid, 0)
            if agent.entity_type == "BP_MNWS_Vehicle_Armored_C":
                # One short retreat is allowed for a true stationary obstacle,
                # followed by forward-heading alignment on the route. Slow
                # reverse travel is handled above even when progress is nonzero.
                yaw = math.radians(float(agent.yaw))
                forward = (math.cos(yaw), math.sin(yaw))
                reverse = self._direction((-forward[0], -forward[1], 0), 0, False)
                side = 1 if attempt % 2 == 0 else -1
                point = (position[0] - forward[0] * 2000 - forward[1] * side * 2500,
                         position[1] - forward[1] * 2000 + forward[0] * side * 2500, position[2])
                self._vehicle_escape[agent.uid] = (state.step, reverse, point, goal)
                self._attempts[agent.uid] = attempt + 1
                self._turns.pop(agent.uid, None)
                self._turn_retry[agent.uid] = state.step + 8
                self._stationary[agent.uid] = (position, 0)
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
        # Alignment is subordinate to recovery and never resets its clock.
        # Once UE is moving the vehicle briskly, avoid interrupting its path
        # just because the reported nose angle still lags behind the goal.
        moving_well = previous is not None and math.dist(position, previous[1]) > 600
        if moving_well:
            self._turns.pop(agent.uid, None)
        else:
            aligned = self._align_vehicle(state, agent, goal)
            if aligned is not None:
                return aligned
        return action
