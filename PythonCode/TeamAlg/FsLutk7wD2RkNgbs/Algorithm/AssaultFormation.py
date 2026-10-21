from __future__ import annotations

import math

from cssim.protocol import Action
from .AssaultRoute import AssaultRoute


class AssaultFormation:
    """Stable vehicle waves and bounded, observation-driven capture handoff.

    Uses reached positions and the existing navigation route. Distances below
    are formation spacing, not claims about weapon effectiveness or safety.
    Local combat always takes precedence over waiting for the formation.
    """

    VEHICLES = {'BP_MNWS_Vehicle_Armored_C', 'BP_MNWS_Vehicle_6x6UGV_C'}
    FOOT = {'BP_BaseSoldier_C', 'BP_RoboDog_C'}
    APPROACH = 35000.0
    RESERVE_LINE = 12000.0
    SCREEN_GAP = 3000.0
    RECON_WAIT = 18
    HANDOFF_WAIT = 28

    def __init__(self):
        self._frame = None
        self.front = set()
        self.reserve = {}
        self._gates = {}
        self._hp = {}
        self._under_fire = set()
        self._active = []
        self._current = None
        self._pending_fire = {}

    @staticmethod
    def _ammo(unit):
        try:
            values = [float(v) for v in unit.raw.get('interaction', ())]
            return sum(values) if values and all(math.isfinite(v) and v >= 0 for v in values) else None
        except (TypeError, ValueError):
            return None

    def begin(self, state, route, recon_ready=False):
        frame = (state.episode, state.step)
        if self._frame == frame:
            return
        if self._frame is None or frame[0] != self._frame[0] or frame[1] < self._frame[1]:
            self.__init__()
            # Both available Lynx support the first wave. Keeping one of only
            # two in reserve had halved the initial Lynx support for infantry.
            lynx = sorted((u for u in state.agents if u.entity_type == 'BP_MNWS_Vehicle_6x6UGV_C'), key=lambda u: u.uid)
            wheeled = sorted((u for u in state.agents if u.entity_type == 'BP_MNWS_Vehicle_Armored_C'), key=lambda u: u.uid)
            vehicles = lynx[:2] + wheeled + lynx[2:]
            count = max(math.ceil(len(vehicles) / 2), len(vehicles) - 4)
            self.front = {u.uid for u in vehicles[:count]}
            spare = [u for u in vehicles if u.uid not in self.front]
            if len(spare) > 4:
                self.front.update(u.uid for u in spare[4:])
                spare = spare[:4]
            secondaries = [uid for uid in route.order if uid != route.hq_uid]
            if secondaries:
                self.reserve = {u.uid: secondaries[i % len(secondaries)]
                                for i, u in enumerate(spare)}
        self._frame = frame
        self._under_fire = {u.uid for u in state.agents if u.uid in self._hp and u.hp < self._hp[u.uid] - .01}
        self._hp = {u.uid: float(u.hp) for u in state.agents if u.alive}
        obj = route.objective(state)
        self._current = obj.uid if obj else None
        alive = [u for u in state.agents if u.alive and u.entity_type in self.VEHICLES
                 and u.communication_ok() is not False]
        hq = state.key_object_by_uid(route.hq_uid) if route.hq_uid is not None else None
        if hq is not None and route.hq_secured and route.owned(state, hq) and route._strategy.occupants(state, hq) == 0:
            for uid in (uid for uid in route.order if uid != route.hq_uid):
                assigned = sum(self.reserve.get(u.uid) == uid for u in alive)
                while assigned < 2:
                    leads = [u for u in alive if u.uid in self.front]
                    if len(leads) <= 1:
                        break
                    unit = leads[-1]
                    self.front.remove(unit.uid)
                    self.reserve[unit.uid] = uid
                    assigned += 1
        self._active = [u for u in alive if u.uid in self.front or self.reserve.get(u.uid) == self._current]
        # Replace depleted support before the last lead vehicle is destroyed.
        # Preserve secondary assignments for the remaining rear vehicles.
        armed = [u for u in alive if self._ammo(u) != 0]
        supporting = [u for u in self._active if self._ammo(u) != 0]
        while obj is not None and len(supporting) < min(2, len(armed)):
            active_ids = {u.uid for u in self._active}
            spare = [u for u in armed if u.uid not in active_ids]
            if not spare:
                break
            promoted = min(spare, key=lambda u: (AssaultRoute.distance(u.position, obj.position), u.uid))
            self.front.add(promoted.uid)
            self.reserve.pop(promoted.uid, None)
            self._active.append(promoted)
            supporting.append(promoted)
        if obj is None:
            return
        gate = self._gates.setdefault(obj.uid, {'start': None, 'recon': False, 'shots': set(),
                                               'clear_frames': 0, 'released': False})
        gate['recon'] = bool(recon_ready)
        for unit in self._active:
            previous = self._pending_fire.get(unit.uid)
            ammo = self._ammo(unit)
            if (previous is not None and previous[0] == obj.uid and 0 < state.step - previous[1] <= 2
                    and ammo is not None and previous[2] is not None and ammo < previous[2]):
                gate['shots'].add(unit.uid)
        self._pending_fire = {uid: data for uid, data in self._pending_fire.items() if state.step - data[1] <= 2}
        mobile = [u for u in state.agents if u.alive and u.entity_type in self.FOOT | self.VEHICLES
                  and u.communication_ok() is not False and u.uid not in route._garrison]
        if gate['start'] is None and any(AssaultRoute.distance(u.position, obj.position) <= self.APPROACH for u in mobile):
            gate['start'] = state.step
        near_vehicles = [u for u in self._active if AssaultRoute.distance(u.position, obj.position) <= 16000]
        clear = gate['recon'] and route._strategy.occupants(state, obj) == 0 and bool(near_vehicles)
        gate['clear_frames'] = gate['clear_frames'] + 1 if clear else 0
        if (not self._active or gate['clear_frames'] >= 3 or len(gate['shots']) >= min(2, len(self._active))
                or gate['start'] is not None and state.step - gate['start'] >= self.HANDOFF_WAIT):
            gate['released'] = True

    def record(self, state, agent, action, route):
        obj = route.objective(state)
        if (obj is not None and agent.entity_type in self.VEHICLES and action.command == 'NormalAttacking'
                and agent.uid in {u.uid for u in self._active}):
            target = next((e for e in state.perceived_opponents(agent) if e.uid == action.target_uid), None)
            if target is not None and AssaultRoute.distance(target.position, obj.position) <= 20000:
                self._pending_fire[agent.uid] = (obj.uid, state.step, self._ammo(agent))

    @staticmethod
    def _guard(agent, obj):
        # Hold a reached position, not a directional patrol that may advance.
        # VehicleFireControl owns aiming toward an actual firing target.
        return Action.guard_position(agent.position)

    def choose(self, state, agent, route):
        obj = route.objective(state)
        if (obj is None or not agent.alive or agent.communication_ok() is False
                or agent.entity_type not in self.FOOT | self.VEHICLES or agent.uid in route._garrison):
            return None
        # Never force a unit to assemble while a locally perceived enemy can
        # engage it, or on the frame it takes damage. Fire control still gates shots.
        if agent.uid in self._under_fire or (agent.entity_type in self.FOOT
                and any(e.alive for e in state.perceived_opponents(agent))):
            return None
        distance = AssaultRoute.distance(agent.position, obj.position)
        if route.inside_garrison(obj, agent.position):
            return None
        gate = self._gates[obj.uid]
        if agent.entity_type in self.VEHICLES:
            assigned = self.reserve.get(agent.uid)
            holding = state.key_object_by_uid(assigned) if assigned is not None else None
            if (holding is not None and route.owned(state, holding)
                    and AssaultRoute.distance(agent.position, holding.position) <= 22000):
                return self._guard(agent, holding)
            if (assigned is not None and assigned != obj.uid and distance <= self.RESERVE_LINE
                    and not gate['released']):
                # Release this initial reserve gate with the assault. The
                # additional 28-step follow wait was retired after Oct20.
                return self._guard(agent, obj)
            if (not gate['recon'] and gate['start'] is not None
                    and any(u.alive and u.entity_type == 'BP_Base_UAV_C' and u.communication_ok() is not False
                            for u in state.agents)
                    and state.step - gate['start'] < self.RECON_WAIT and distance <= self.RESERVE_LINE):
                return self._guard(agent, obj)
        elif not gate['released'] and self._active and distance <= self.APPROACH:
            # Hold early foot arrivals where they are; rear units continue along
            # the same route. Never move infantry backwards into a rally pile.
            front_distance = min(AssaultRoute.distance(u.position, obj.position) for u in self._active)
            if distance <= max(18000, front_distance + self.SCREEN_GAP):
                return self._guard(agent, obj)
        return None
