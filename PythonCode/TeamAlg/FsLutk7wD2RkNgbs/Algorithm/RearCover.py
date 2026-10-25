from __future__ import annotations

import math

from cssim.protocol import Action

from .AssaultRoute import AssaultRoute
from .VehicleFireControl import VehicleFireControl


class RearCover:
    """Bounded HQ rear cover and reactions to damage without a visible target.

    Selection uses only current friendly positions, ammunition and the public
    occupation state. Scanning and damage reactions never authorize a shot.
    VehicleFireControl must run first so a real local firing opportunity wins.
    """

    FOOT = {"BP_BaseSoldier_C", "BP_RoboDog_C"}
    WINDOW = 32
    HQ_RANGE = 18000.0
    GUARD_RANGE = 8000.0
    HOLD_RADIUS = 1200.0
    REACTION_STEPS = 6
    REACTION_DISTANCE = 1400.0
    ARRIVAL = 400.0
    THREAT_MEMORY = 4
    SUPPORT_HP = 25.0
    DIRECTIONS = ("+X", "+X+Y", "+Y", "-X+Y", "-X", "-X-Y", "-Y", "+X-Y")

    def __init__(self):
        self._frame = None
        self._hp = {}
        self._damaged = set()
        self._secured = False
        self._hq_uid = None
        self._deadline = None
        self._cover_uid = None
        self._cover_point = None
        self._plans = {}
        self._attempts = {}
        self._finished = {}
        self._scan_started = {}
        self._rear_threat_step = None

    @staticmethod
    def _position(unit):
        return tuple(float(value) for value in unit.position)

    def begin(self, state, route):
        frame = (state.episode, state.step)
        if frame == self._frame:
            return
        if self._frame is None or frame[0] != self._frame[0] or frame[1] < self._frame[1]:
            self.__init__()
        self._frame = frame
        self._damaged = {u.uid for u in state.agents if u.alive and u.uid in self._hp
                         and float(u.hp) < self._hp[u.uid] - .01}
        self._hp = {u.uid: float(u.hp) for u in state.agents if u.alive}
        live = set(self._hp)
        self._plans = {uid: plan for uid, plan in self._plans.items() if uid in live}
        self._scan_started = {key: start for key, start in self._scan_started.items() if key[0] in live}
        hq = state.key_object_by_uid(route.hq_uid) if route.hq_uid is not None else None
        objective = route.objective(state)
        secured = (hq is not None and route.hq_secured and route.owned(state, hq)
                   and objective is not None and objective.uid != hq.uid)
        self._hq_uid = route.hq_uid
        if secured and not self._secured:
            self._deadline = state.step + self.WINDOW
        elif not secured and self._secured:
            self._deadline = None
            self._rear_threat_step = None
            self._plans.clear()
            self._scan_started.clear()
        self._secured = secured
        if not secured or self._deadline is None or state.step >= self._deadline:
            self._cover_uid = self._cover_point = None
            return
        guards = [u for u in state.agents if u.alive and u.entity_type in self.FOOT
                  and u.communication_ok() is not False
                  and route._garrison.get(u.uid) == hq.uid
                  and route.inside_garrison(hq, u.position)]
        contested_b = route.is_base_b(objective) and route.enemy_count(state, objective) > 0
        if contested_b:
            # B needs its remaining gun vehicles together. A blind fixed HQ
            # scan must not consume a scarce assault vehicle for 32 frames.
            # Local fire still runs before this selector, even with no spare.
            threatened = any(g.uid in self._damaged or any(
                e.alive and e.uid in g.perceived_uids() and e.entity_type not in VehicleFireControl.AIR
                and AssaultRoute.distance(g.position, e.position) <= 15000
                for e in state.perceived_opponents(g)) for g in guards)
            if threatened:
                self._rear_threat_step = state.step
            if (self._rear_threat_step is None
                    or state.step - self._rear_threat_step > self.THREAT_MEMORY):
                self._cover_uid = self._cover_point = None
                return
        support = [u for u in state.agents if u.alive and u.hp >= self.SUPPORT_HP
                   and u.entity_type in VehicleFireControl.VEHICLES
                   and u.communication_ok() is not False
                   and (VehicleFireControl.ammo(u) or 0) > 0]
        candidates = []
        for unit in state.agents:
            ammo = VehicleFireControl.ammo(unit)
            if (unit.alive and unit.hp > 0 and unit.entity_type in VehicleFireControl.VEHICLES
                    and unit.communication_ok() is not False and ammo is not None and ammo > 0
                    and AssaultRoute.distance(unit.position, hq.position) <= self.HQ_RANGE
                    and (not contested_b or sum(u.uid != unit.uid for u in support) >= 2)
                    and any(AssaultRoute.distance(unit.position, guard.position) <= self.GUARD_RANGE
                            for guard in guards)):
                candidates.append(unit)
        current = next((u for u in candidates if u.uid == self._cover_uid), None)
        if current is None:
            self._cover_uid = self._cover_point = None
            if candidates:
                current = min(candidates, key=lambda u: (
                    contested_b and u.entity_type == 'BP_MNWS_Vehicle_6x6UGV_C',
                    min(AssaultRoute.distance(u.position, g.position) for g in guards), u.uid))
                self._cover_uid, self._cover_point = current.uid, self._position(current)

    def choose_vehicle(self, state, agent, route):
        self.begin(state, route)
        if agent.uid != self._cover_uid:
            return None
        if AssaultRoute.distance(agent.position, self._cover_point) > self.HOLD_RADIUS:
            return Action.move_at(self._cover_point)
        hq = state.key_object_by_uid(self._hq_uid)
        return self.scan(state, agent, hq)

    def scan(self, state, agent, obj):
        """Observe sectors from a reached position; never infer an enemy UID."""
        if obj is None or obj.uid != self._hq_uid or not self._secured:
            return Action.guard_position(agent.position)
        start = self._scan_started.setdefault((agent.uid, obj.uid), state.step)
        elapsed = state.step - start
        if elapsed < 4:
            return Action.guard_position(agent.position)
        sector = (agent.team_index + (elapsed - 4) // 4) % len(self.DIRECTIONS)
        return Action.guard(self.DIRECTIONS[sector])

    def _reaction_point(self, agent, obj, route):
        attempt = self._attempts.get(agent.uid, 0)
        self._attempts[agent.uid] = attempt + 1
        dx, dy = obj.position[:2] - agent.position[:2]
        length = math.hypot(dx, dy)
        if length > self.ARRIVAL:
            dx, dy = dx / length, dy / length
            side = 1 if (agent.team_index + attempt) % 2 else -1
            dx, dy = .6 * dx - .8 * side * dy, .6 * dy + .8 * side * dx
        else:
            angle = (agent.team_index + 3 * attempt) * math.pi / 4
            dx, dy = math.cos(angle), math.sin(angle)
        point = route.clamp_guard_point(obj, (
            agent.position[0] + dx * self.REACTION_DISTANCE,
            agent.position[1] + dy * self.REACTION_DISTANCE, agent.position[2]))
        # A corner clamp can erase a lateral move. Try other short inward
        # directions using only the public box; no unseen attacker is guessed.
        if AssaultRoute.distance(agent.position, point) < 800:
            alternatives = [route.clamp_guard_point(obj, (
                agent.position[0] + math.cos(i * math.pi / 4) * self.REACTION_DISTANCE,
                agent.position[1] + math.sin(i * math.pi / 4) * self.REACTION_DISTANCE,
                agent.position[2])) for i in range(8)]
            point = max(alternatives, key=lambda p: AssaultRoute.distance(agent.position, p))
        return point

    def guard_reaction(self, state, agent, obj, route):
        """Call only after available local combat has been considered.

        Damage is evidence of danger, not an enemy location. Keep one short,
        fixed movement request for at most six steps, then hold the reached
        position rather than returning to the old position under fire.
        """
        self.begin(state, route)
        if (not agent.alive or agent.communication_ok() is False or agent.entity_type not in self.FOOT
                or not route.inside_garrison(obj, agent.position)):
            self._plans.pop(agent.uid, None)
            return None
        plan = self._plans.get(agent.uid)
        if plan is not None and plan[0] != obj.uid:
            self._plans.pop(agent.uid, None)
            plan = None
        if plan is not None:
            if state.step >= plan[2] or AssaultRoute.distance(agent.position, plan[1]) <= self.ARRIVAL:
                route._guard_slots[(agent.uid, obj.uid)] = route.clamp_guard_point(obj, agent.position)
                self._plans.pop(agent.uid, None)
                self._finished[agent.uid] = state.step
                return None
            return Action.move_at(plan[1])
        if agent.uid not in self._damaged or self._finished.get(agent.uid) == state.step:
            return None
        point = self._reaction_point(agent, obj, route)
        self._plans[agent.uid] = (obj.uid, point, state.step + self.REACTION_STEPS)
        return Action.move_at(point)
