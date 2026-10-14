from __future__ import annotations

from collections import deque
import logging

from cssim.protocol import Action
from .AssaultRoute import AssaultRoute
from .UrbanContact import UrbanContact

logger = logging.getLogger(__name__)


class CombatReadiness:
    """Protect depleted forces using public occupants and reached fallback points.

    No opening assembly gate. A weak later assault, or a rapid first-objective
    collapse, stops at the last actually reached approach position. Scouts and
    effective firing units continue working; strength is reassessed each frame.
    """

    def __init__(self):
        self._frame = None
        self._objective = None
        self._history = deque(maxlen=9)
        self._fallback = {}
        self._probes = {}
        self.weak = False

    def begin(self, state, route):
        frame = (state.episode, state.step)
        if self._frame == frame:
            return
        obj = route.objective(state)
        key = (state.episode, obj.uid if obj else None)
        if self._objective != key or self._frame is not None and state.step < self._frame[1]:
            self._history.clear()
            self._fallback.clear()
            self._probes.clear()
            self.weak = False
        self._frame, self._objective = frame, key
        if obj is None:
            self.weak = False
            return
        fighters = [unit for unit in state.agents if unit.alive and unit.entity_type in UrbanContact.GROUND
                    and unit.communication_ok() is not False and unit.uid not in route._garrison]
        power = 0.0
        for unit in fighters:
            ammo = UrbanContact.ammo(unit)
            if ammo is None or ammo > 0:
                weight = .5 if unit.entity_type == 'BP_RoboDog_C' else 2 if 'Vehicle' in unit.entity_type else 1
                power += weight * max(0.0, min(1.0, float(unit.hp) / max(float(unit.initial_hp), 1)))
            if AssaultRoute.distance(unit.position, obj.position) >= 24000:
                self._fallback[unit.uid] = tuple(float(v) for v in unit.position)
        # Use actual infantry/dog casualties, not shifts in guard assignment or
        # arriving vehicles, to detect a collapse during the first assault.
        alive = sum(unit.alive and unit.entity_type in {'BP_BaseSoldier_C', 'BP_RoboDog_C'} for unit in state.agents)
        self._history.append((state.step, alive))
        peak = max((count for step, count in self._history if state.step - step <= 8), default=alive)
        collapse = peak >= 8 and peak - alive >= max(4, peak * .25)
        enemy = route.enemy_count(state, obj)
        depleted = enemy > 0 and power < max(3, .8 * enemy)
        weak = depleted and (route.stage > 0 or collapse or self.weak)
        if weak != self.weak:
            logger.info('进攻能力重估: step=%d objective=%d preserve=%s armed_power=%.1f enemy_occupants=%d',
                        state.step, obj.uid, weak, power, enemy)
        self.weak = weak

    def choose(self, state, agent, route, urban, contacts):
        obj = route.objective(state)
        if (not self.weak or obj is None or agent.entity_type not in UrbanContact.GROUND
                or agent.uid in route._garrison or agent.communication_ok() is False
                or AssaultRoute.distance(agent.position, obj.position) > 26000):
            return None
        if any(enemy.entity_type in UrbanContact.GROUND and urban.effective(state, agent, enemy)
               for enemy in contacts):
            return None
        # A newly exposed local target still gets a short firing trial. Without
        # this, preserving a unit would prevent the very shot that can establish
        # effective fire and release it from fallback. Shared reports alone do
        # not cancel preservation, and target switching cannot reset the timer.
        local = {enemy.uid for enemy in state.perceived_opponents(agent)}
        ammo = urban.ammo(agent)
        if (ammo is None or ammo > 0) and any(
                enemy.uid in local and enemy.entity_type in UrbanContact.GROUND for enemy in contacts):
            started = self._probes.get(agent.uid, -100)
            if state.step - started >= 8:
                started = state.step
                self._probes[agent.uid] = started
            if state.step - started < 2:
                return None
        point = self._fallback.get(agent.uid)
        if point is None:
            return None  # No fabricated safe waypoint across unknown geometry.
        if AssaultRoute.distance(agent.position, point) > 500:
            return Action.move_at(point)
        return Action.guard_position(point)
