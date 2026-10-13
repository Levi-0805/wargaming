from __future__ import annotations

import math

from cssim.protocol import Action
from .AssaultRoute import AssaultRoute
from .MotionRecovery import MotionRecovery


class UrbanContact:
    """Seek a useful view using public contacts and reached friendly positions.

    No collision mesh is exposed by this scenario. A waypoint is a request to
    UE navigation, not a claim that a straight line through a wall is clear.
    """

    GROUND = {"BP_BaseSoldier_C", "BP_RoboDog_C", "BP_MNWS_Vehicle_Armored_C", "BP_MNWS_Vehicle_6x6UGV_C"}

    def __init__(self):
        self._frame = None
        self._shots = {}
        self._effective = {}
        self._probes = {}
        self._plans = {}
        self._arrived = {}
        self._attempts = {}
        self._positions = {}
        self._used = {}

    @staticmethod
    def ammo(agent):
        values = agent.raw.get("interaction", ()) or ()
        try:
            return sum(float(value) for value in values) if values else None
        except (TypeError, ValueError):
            return None

    def begin(self, state):
        frame = (state.episode, state.step)
        if self._frame == frame:
            return
        if self._frame is None or frame[0] != self._frame[0] or frame[1] < self._frame[1]:
            for mapping in (self._shots, self._effective, self._probes, self._plans,
                            self._arrived, self._attempts, self._positions, self._used):
                mapping.clear()
        self._frame = frame
        for uid, (step, target_uid, ammo, hp) in list(self._shots.items()):
            own, enemy = state.entity_by_uid(uid), state.entity_by_uid(target_uid)
            current_ammo = self.ammo(own) if own is not None else None
            if (state.step - step == 1 and own is not None and own.alive and enemy is not None
                    and ammo is not None and current_ammo is not None and current_ammo < ammo
                    and float(enemy.hp) < hp):
                self._effective[uid] = (target_uid, state.step + 2)
            del self._shots[uid]
        for unit in state.agents:
            if unit.alive and unit.entity_type in self.GROUND and unit.communication_ok() is not False:
                p = tuple(float(v) for v in unit.position)
                key = tuple(round(v / 800) for v in p)
                self._positions[key] = (p, state.step)
        self._positions = {key: value for key, value in self._positions.items() if state.step - value[1] <= 40}
        self._used = {key: step for key, step in self._used.items() if state.step - step <= 40}

    def record(self, state, agent, action):
        if action.command == "NormalAttacking":
            enemy = state.entity_by_uid(action.target_uid)
            if enemy is not None:
                self._shots[agent.uid] = (state.step, enemy.uid, self.ammo(agent), float(enemy.hp))
            self.cancel(agent)

    def effective(self, state, agent, target):
        entry = self._effective.get(agent.uid)
        return entry is not None and entry[0] == target.uid and state.step <= entry[1]

    def shared_fire_allowed(self, state, agent, target):
        if self.effective(state, agent, target):
            return True
        # One trial per unit, not one per target: cycling shared targets must
        # not bypass the cooldown. Ammo expenditure alone is not confirmation.
        if state.step - self._probes.get(agent.uid, -100) >= 8:
            self._probes[agent.uid] = state.step
            return True
        return False

    def cancel(self, agent):
        self._plans.pop(agent.uid, None)
        self._arrived.pop(agent.uid, None)

    def planned_target(self, state, agent, contacts):
        plan = self._plans.get(agent.uid)
        if plan and plan[0][0] == "contact" and state.step < plan[2]:
            return next((target for target in contacts if target.uid == plan[0][1]), None)
        return None

    def _existing(self, state, agent, key, look_at):
        plan = self._plans.get(agent.uid)
        if plan and plan[0] == key and state.step < plan[2]:
            if AssaultRoute.distance(agent.position, plan[1]) > 400:
                return Action.move_at(plan[1])
            arrived = self._arrived.setdefault(agent.uid, state.step)
            if state.step - arrived < 2:
                vector = tuple(look_at[i] - agent.position[i] for i in range(3))
                return Action.guard(MotionRecovery._direction(vector, 0, airborne=False))
        self.cancel(agent)
        return None

    def _set(self, state, agent, key, point, duration):
        point = tuple(float(v) for v in point)
        self._arrived.pop(agent.uid, None)
        self._plans[agent.uid] = (key, point, state.step + duration)
        return Action.move_at(point)

    def _space_available(self, state, agent, point):
        return sum(uid != agent.uid and state.step < plan[2]
                   and AssaultRoute.distance(point, plan[1]) < 600
                   for uid, plan in self._plans.items()) < 2

    def constrain(self, agent, action, objective, route):
        if action.command != "Moving" or action.points is None:
            return action
        point = route.clamp_guard_point(objective, action.points)
        plan = self._plans.get(agent.uid)
        if plan and plan[1] == action.points:
            self._plans[agent.uid] = (plan[0], point, plan[2])
        return Action.move_at(point)

    def reposition(self, state, agent, target, duration=6):
        key = ("contact", target.uid)
        existing = self._existing(state, agent, key, target.position)
        if existing is not None:
            return existing
        observers = [unit for unit in state.agents if unit.uid != agent.uid and unit.alive
                     and unit.entity_type in self.GROUND and unit.communication_ok() is not False
                     and abs(float(unit.position[2] - agent.position[2])) < 600
                     and any(enemy.uid == target.uid for enemy in state.perceived_opponents(unit))
                     and self._space_available(state, agent, unit.position)
                     and 1200 < AssaultRoute.distance(unit.position, agent.position) < 8000]
        # Prefer an already occupied observation position, with separate nearby
        # destinations to keep every blocked unit from choosing the same spot.
        observers.sort(key=lambda unit: (not self.effective(state, unit, target),
                                        AssaultRoute.distance(unit.position, agent.position), unit.uid))
        if observers:
            unit = observers[agent.team_index % min(3, len(observers))]
            point = tuple(float(v) for v in unit.position)
        else:
            attempt = self._attempts.get(agent.uid, 0)
            self._attempts[agent.uid] = attempt + 1
            dx, dy = target.position[:2] - agent.position[:2]
            if math.hypot(dx, dy) < 1:
                dx, dy = 1.0, 0.0
            length = max(math.hypot(dx, dy), 1)
            side = (1 if agent.team_index % 2 else -1) * (1 if attempt % 2 == 0 else -1)
            # Fixed short destination; retry the other side after a bounded
            # attempt. Never continuously move the goal ahead of the unit.
            lateral = 800 if duration <= 2 else 1600
            advance = 1000 if length > 8000 else 0
            point = (agent.position[0] + (dx * advance - dy * side * lateral) / length,
                     agent.position[1] + (dy * advance + dx * side * lateral) / length,
                     agent.position[2])
        return self._set(state, agent, key, point, duration)

    def search_point(self, state, agent, objective, route):
        key = ("clear", objective.uid)
        existing = self._existing(state, agent, key, objective.position)
        if existing is not None:
            return existing
        candidates = [(k, p) for k, (p, _) in self._positions.items()
                      if route.inside_garrison(objective, p)
                      and abs(p[2] - agent.position[2]) < 600
                      and 1200 < AssaultRoute.distance(agent.position, p) < 8000
                      and self._space_available(state, agent, p)
                      and state.step - self._used.get((agent.uid, k), -100) > 20]
        candidates.sort(key=lambda row: (AssaultRoute.distance(agent.position, row[1]), row[0]))
        if candidates:
            k, point = candidates[agent.team_index % min(3, len(candidates))]
            self._used[(agent.uid, k)] = state.step
            return self._set(state, agent, key, point, 6)
        # The point's public occupant count authorizes searching, but gives no
        # hidden enemy coordinates. A local sideways look is the fallback.
        action = self.reposition(state, agent, objective, duration=6)
        plan = self._plans.get(agent.uid)
        if plan:
            self._plans[agent.uid] = (key, plan[1], plan[2])
        return action
