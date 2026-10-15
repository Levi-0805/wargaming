from __future__ import annotations

import logging
import math

from cssim.protocol import Action
from .AssaultRoute import AssaultRoute

logger = logging.getLogger(__name__)


class ApproachCohesion:
    """One bounded rendezvous on the opening road, never a recurring tether.

    Early arrivals wait where they actually arrived. Trailing units keep their
    navigation; no artificial assembly coordinate or march back is generated.
    All ranges are policy distances in cm, not claimed safe weapons ranges.
    """

    SOLDIER = 'BP_BaseSoldier_C'
    VEHICLES = {'BP_MNWS_Vehicle_Armored_C', 'BP_MNWS_Vehicle_6x6UGV_C'}
    GROUND = VEHICLES | {SOLDIER, 'BP_RoboDog_C'}
    HOLD_RANGE = 32000.0
    JOIN_RANGE = 36000.0
    MAX_WAIT = 28

    def __init__(self):
        self._frame = None
        self.started = None
        self.released = False
        self.reason = None
        self._hp = {}

    def begin(self, state, route):
        frame = (state.episode, state.step)
        if self._frame == frame:
            return
        if self._frame is None or frame[0] != self._frame[0] or frame[1] < self._frame[1]:
            self.started = None
            self.released = False
            self.reason = None
            self._hp.clear()
        self._frame = frame
        obj = route.objective(state)
        if self.released or not route.known_map or obj is None:
            return
        units = [u for u in state.agents if u.alive and u.entity_type in self.GROUND
                 and u.communication_ok() is not False and u.uid not in route._garrison]
        distances = {u.uid: AssaultRoute.distance(u.position, obj.position) for u in units}
        soldiers = [u for u in units if u.entity_type == self.SOLDIER]
        vehicles = [u for u in units if u.entity_type in self.VEHICLES]
        damaged = any(u.uid in self._hp and u.hp < self._hp[u.uid] - .01
                      and distances[u.uid] <= self.HOLD_RANGE for u in units)
        self._hp = {u.uid: float(u.hp) for u in units}
        # Never recall troops already fighting inside or reactivate after HQ
        # capture, even when that point is subsequently lost.
        if route.stage > 0 or any(route.inside_garrison(obj, u.position) for u in units):
            reason = 'already_engaged'
        elif len(soldiers) < 8:
            reason = 'small_force'
        else:
            reason = None
            if self.started is None and any(distances[u.uid] <= self.HOLD_RANGE for u in soldiers):
                self.started = state.step
                logger.info('接敌前短暂集结: step=%d deadline=%d', state.step, state.step + self.MAX_WAIT)
            if self.started is not None:
                infantry_ready = sum(distances[u.uid] <= self.JOIN_RANGE for u in soldiers) >= math.ceil(.7 * len(soldiers))
                vehicles_ready = sum(distances[u.uid] <= 38000 for u in vehicles) >= min(2, len(vehicles))
                if damaged:
                    reason = 'under_fire'
                elif infantry_ready and vehicles_ready:
                    reason = 'main_body_ready'
                elif state.step - self.started >= self.MAX_WAIT:
                    reason = 'deadline'
        if reason is not None:
            self.released, self.reason = True, reason
            logger.info('集结结束，不再等待后队: step=%d reason=%s', state.step, reason)

    def choose(self, state, agent, route):
        if (self.released or self.started is None or agent.entity_type not in self.GROUND
                or not agent.alive or agent.communication_ok() is False or agent.uid in route._garrison):
            return None
        obj = route.objective(state)
        distance = AssaultRoute.distance(agent.position, obj.position) if obj else 0
        if not 18000 < distance <= self.HOLD_RANGE:
            return None
        if any(enemy.alive for enemy in state.perceived_opponents(agent)):
            return None  # A real local contact can be fought during assembly.
        return Action.guard_position(agent.position)
