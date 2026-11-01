from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


class AdaptiveAssault:
    """Compare objectives using public observations, with bounded memory.

    Occupant counts describe the capture box, not all nearby defenders. A zero
    count therefore retains a small approach cost, and absent data is unknown.
    No replay, enemy initial positions, or full simulation state is consulted.
    """

    GROUND = {'BP_BaseSoldier_C', 'BP_RoboDog_C',
              'BP_MNWS_Vehicle_Armored_C', 'BP_MNWS_Vehicle_6x6UGV_C'}
    CONTACT_TTL = 16
    CONFIRMATIONS = 3
    COMMIT_STEPS = 30
    RECON_DWELL = 20
    ASSESS_RANGE = 40000.0

    def __init__(self):
        self._frame = None
        self._contacts = {}
        self._candidate = None
        self._samples = 0
        self._changed = None
        self._deferred = {}
        self._recon = None
        self._recon_started = None
        self._visited = {}
        self._arrival = 0
        self._utilities = {}

    @staticmethod
    def occupants(state, obj):
        try:
            value = float(obj.raw['blueteamNum' if state.team == 0 else 'redTeamNum'])
            return value if math.isfinite(value) and value >= 0 else None
        except (KeyError, TypeError, ValueError):
            return None

    def observe(self, state):
        frame = (state.episode, state.step)
        if self._frame == frame:
            return
        if self._frame is not None and (frame[0] != self._frame[0] or frame[1] <= self._frame[1]):
            self.__init__()
        if self._frame is not None and state.step != self._frame[1] + 1:
            self._candidate, self._samples = None, 0
        self._frame = frame
        self._utilities.clear()
        for enemy in state.visible_opponents:
            if not enemy.alive:
                self._contacts.pop(enemy.uid, None)
            elif enemy.entity_type in self.GROUND:
                weight = 2.0 if 'Vehicle' in enemy.entity_type else .5 if 'RoboDog' in enemy.entity_type else 1.0
                self._contacts[enemy.uid] = (state.step, tuple(enemy.position), weight)
        self._contacts = {uid: row for uid, row in self._contacts.items()
                          if state.step - row[0] <= self.CONTACT_TTL}

    def pressure(self, state, route, obj):
        count = self.occupants(state, obj)
        reported = sum(weight for _, point, weight in self._contacts.values()
                       if route.distance(point, obj.position) <= 20000)
        # Unknown is never treated as an empty capture box.
        return max(4.0 if count is None else count, reported)

    def utility(self, state, route, obj):
        key = (obj.uid, tuple(sorted(route._garrison)))
        if key in self._utilities:
            return self._utilities[key]
        mobile = [unit for unit in state.agents if unit.alive and unit.entity_type in self.GROUND
                  and unit.communication_ok() is not False and unit.uid not in route._garrison]
        distances = sorted(route.distance(unit.position, obj.position) for unit in mobile)
        distance = distances[len(distances) // 2] if distances else 0
        origin = min(mobile, key=lambda u: abs(route.distance(u.position, obj.position)-distance)).position if mobile else obj.position
        transit = sum(self.pressure(state, route, other)
                      for other in route.transit_objectives(state, origin, obj))
        value = 1600.0 if route.is_hq(obj) else 800.0
        # A twofold value advantage cannot outweigh a heavily defended box.
        # A distance penalty prevents marginal preferences causing long detours.
        result = value / ((2.0 + self.pressure(state, route, obj) + transit) * (1.0 + distance / 100000.0))
        self._utilities[key] = result
        return result

    def select(self, state, route, *, next_objective=False):
        self.observe(state)
        candidates = [state.key_object_by_uid(uid) for uid in route.order[route.stage:]]
        candidates = [obj for obj in candidates if obj is not None and obj.valid]
        current = route.objective(state)
        if not candidates or current is None:
            return None
        # Both the 5630- and 860-point recordings started with 16 HQ defenders.
        # Observe while advancing; initial placement does not identify whether
        # blue will hold HQ or disperse. Do not replace the proven opening road
        # until the main body approaches or an aircraft actually sees forward
        # contacts. This is a spatial gate, not a replay-specific turn timer.
        mobile = [unit for unit in state.agents if unit.alive and unit.entity_type in self.GROUND
                  and unit.communication_ok() is not False and unit.uid not in route._garrison]
        distances = sorted(route.distance(unit.position, current.position) for unit in mobile)
        forward_contact = any(unit.alive and unit.entity_type == 'BP_Base_UAV_C'
                              and unit.communication_ok() is not False
                              and 'agentPerception' in unit.raw
                              and any(route.distance(enemy.position, obj.position) <= 20000
                                      for enemy in state.perceived_opponents(unit)
                                      for obj in candidates)
                              for unit in state.agents)
        if (self._changed is None and not next_objective
                and not forward_contact
                and (not distances or distances[len(distances) // 2] > self.ASSESS_RANGE)):
            self._candidate, self._samples = None, 0
            return None
        # Secure a foothold before redirecting its assigned defenders.
        if route.owned(state, current):
            self._candidate, self._samples = None, 0
            return None
        eligible = [obj for obj in candidates if obj.uid == current.uid
                    or state.step >= self._deferred.get(obj.uid, -1)]
        best = max(eligible, key=lambda obj: (self.utility(state, route, obj), obj.uid == current.uid))
        convincing = self.utility(state, route, best) > self.utility(state, route, current) * 1.35
        if best.uid == current.uid or not convincing:
            self._candidate, self._samples = None, 0
            return None
        if self._candidate == best.uid:
            self._samples += 1
        else:
            self._candidate, self._samples = best.uid, 1
        if not next_objective and (self._samples < self.CONFIRMATIONS or (
                self._changed is not None and state.step - self._changed < self.COMMIT_STEPS)):
            return None
        self._changed = state.step
        self._deferred[current.uid] = state.step + self.COMMIT_STEPS
        self._candidate, self._samples = None, 0
        logger.info('公开敌情调整主攻目标: step=%d old=%d next=%d old_pressure=%.1f next_pressure=%.1f',
                    state.step, current.uid, best.uid, self.pressure(state, route, current),
                    self.pressure(state, route, best))
        return best.uid

    def recon_pending(self, state, route):
        obj = route.objective(state)
        if obj is None or self.occupants(state, obj) is not None:
            return False
        return not any(route.distance(point, obj.position) <= 20000
                       for _, point, _ in self._contacts.values())

    def recon_objective(self, state, route, agent=None):
        self.observe(state)
        available = [obj for obj in state.key_objects if obj.valid and not route.owned(state, obj)]
        if not available:
            return route.objective(state)
        old = next((obj for obj in available if obj.uid == self._recon), None)
        if old is not None:
            if agent is not None and route.distance(agent.position, old.position) <= 18000:
                self._arrival += 1
            else:
                self._arrival = 0
            # Travel time alone cannot rotate a scout away before it arrives.
            if self._arrival < 3 or state.step - self._recon_started < self.RECON_DWELL:
                return old
            self._visited[old.uid] = state.step
        current = route.objective(state)
        chosen = min(available, key=lambda obj: (
            self._visited.get(obj.uid, -1), self.occupants(state, obj) is not None,
            obj.uid != (current.uid if current else None), obj.uid))
        self._recon, self._recon_started, self._arrival = chosen.uid, state.step, 0
        return chosen
