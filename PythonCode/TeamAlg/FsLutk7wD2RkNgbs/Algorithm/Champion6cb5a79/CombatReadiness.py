from __future__ import annotations

from collections import deque
import logging

from cssim.protocol import Action
from .AssaultRoute import AssaultRoute
from .UrbanContact import UrbanContact

logger = logging.getLogger(__name__)


class CombatReadiness:
    """Assess support and allow a bounded scan, never an indefinite retreat.

    In the 20261014_211207 recording, healthy armed soldiers stayed at the
    240m fallback for over 200 steps. A strength deficit cannot improve while
    all capture units are held there. Pause locally, then resume navigation.
    """

    CAPTURERS = {'BP_BaseSoldier_C', 'BP_RoboDog_C'}
    INFANTRY_SUPPORT = 30000.0
    VEHICLE_SUPPORT = 18000.0
    PAUSE_STEPS = 6
    ADVANCE_STEPS = 24

    def __init__(self):
        self._frame = None
        self._objective = None
        self._history = deque(maxlen=21)
        self._pauses = {}
        self._resume_until = {}
        self._probes = {}
        self.weak = False
        self.front_power = 0.0
        self.rear_power = 0.0
        self.enemy_pressure = 0.0
        self._recovery_samples = 0

    def begin(self, state, route):
        frame = (state.episode, state.step)
        if self._frame == frame:
            return
        obj = route.objective(state)
        key = (state.episode, obj.uid if obj else None)
        if self._objective != key or self._frame is not None and state.step < self._frame[1]:
            self._history.clear()
            self._pauses.clear()
            self._resume_until.clear()
            self._probes.clear()
            self.weak = False
            self._recovery_samples = 0
        self._frame, self._objective = frame, key
        if obj is None:
            self.weak = False
            return
        fighters = [unit for unit in state.agents if unit.alive and unit.entity_type in UrbanContact.GROUND
                    and unit.communication_ok() is not False
                    and route._garrison.get(unit.uid, obj.uid) == obj.uid]
        power = rear_power = 0.0
        front_capturers = 0
        for unit in fighters:
            distance = AssaultRoute.distance(unit.position, obj.position)
            capturer = unit.entity_type in self.CAPTURERS
            front = distance <= (self.INFANTRY_SUPPORT if capturer else self.VEHICLE_SUPPORT)
            front_capturers += capturer and front
            ammo = UrbanContact.ammo(unit)
            if ammo is None or ammo > 0:
                weight = .5 if unit.entity_type == 'BP_RoboDog_C' else 2 if 'Vehicle' in unit.entity_type else 1
                value = weight * max(0.0, min(1.0, float(unit.hp) / max(float(unit.initial_hp), 1)))
                if front:
                    power += value
                else:
                    rear_power += value
        # Use actual infantry/dog casualties, not shifts in guard assignment or
        # arriving vehicles, to detect a collapse during the first assault.
        alive = sum(unit.alive and unit.entity_type in self.CAPTURERS for unit in state.agents)
        initial = sum(unit.entity_type in self.CAPTURERS for unit in state.agents)
        self._history.append((state.step, alive))
        peak = max((count for step, count in self._history if state.step - step <= 20), default=alive)
        collapse = peak >= 8 and peak - alive >= max(3, peak * .20)
        critical = initial >= 8 and alive <= initial * .4
        # Public capture counts omit guns outside the capture box. Include
        # currently reported ground threats nearby without reading hidden units.
        reported = sum(2 if 'Vehicle' in unit.entity_type else .5 if unit.entity_type == 'BP_RoboDog_C' else 1
                       for unit in state.visible_opponents if unit.alive and unit.entity_type in UrbanContact.GROUND
                       and AssaultRoute.distance(unit.position, obj.position) <= 20000)
        enemy = max(route.enemy_count(state, obj), reported)
        capture_depleted = (critical and front_capturers <= max(3, enemy * .5))
        depleted = enemy > 0 and (power < max(3, .8 * enemy) or capture_depleted)
        weak = depleted and (route.stage > 0 or collapse or critical or self.weak)
        if self.weak and not weak and enemy > 0:
            self._recovery_samples += 1
            weak = self._recovery_samples < 3
        else:
            self._recovery_samples = 0
        self.front_power, self.rear_power, self.enemy_pressure = power, rear_power, enemy
        if weak != self.weak:
            logger.info('进攻能力重估: step=%d objective=%d preserve=%s front_power=%.1f rear_power=%.1f enemy_pressure=%.1f capturers=%d',
                        state.step, obj.uid, weak, power, rear_power, enemy, front_capturers)
        self.weak = weak

    def choose(self, state, agent, route, urban, contacts):
        obj = route.objective(state)
        if (not self.weak or obj is None or agent.entity_type not in UrbanContact.GROUND
                or agent.uid in route._garrison or agent.communication_ok() is False
                or route.inside_garrison(obj, agent.position)
                or AssaultRoute.distance(agent.position, obj.position) > 26000):
            return None
        # Vehicles continue their task, including after ammunition is spent.
        # Do not strand them at an arbitrary ring outside the objective.
        if agent.entity_type not in self.CAPTURERS:
            return None
        if state.step < self._resume_until.get(agent.uid, -1):
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
        started, point = self._pauses.setdefault(
            agent.uid, (state.step, tuple(float(v) for v in agent.position)))
        if state.step - started >= self.PAUSE_STEPS or AssaultRoute.distance(agent.position, point) > 600:
            self._pauses.pop(agent.uid, None)
            self._resume_until[agent.uid] = state.step + self.ADVANCE_STEPS
            logger.info('短暂停顿结束，恢复换位推进: uid=%d step=%d', agent.uid, state.step)
            return None
        # Guard the current reached position; never retrace a distant old point.
        return Action.guard_position(agent.position)
