"""Reviewed guard overlays over the immutable historical route."""
import math

from .Champion6cb5a79.AssaultRoute import AssaultRoute


class TiedHQGuardRoute(AssaultRoute):
    def _assign_garrison(self, state, include_current=False):
        previous = dict(self._garrison)
        slots = dict(self._guard_slots)
        hq = state.key_object_by_uid(self.hq_uid)
        retained = {}
        # Do not invent ownership, new guards, or a new departure permission.
        # An explicit remnant withdrawal continues to take precedence.
        if (self.adaptive and self.stage > 0 and not self._remnant.active
                and hq is not None and hq.valid and hq.team is None
                and not self.owned(state, hq) and self.count(hq, 'percent') >= 1):
            friendly, enemy = (('redTeamNum', 'blueteamNum') if state.team == 0
                               else ('blueteamNum', 'redTeamNum'))
            if self.count(hq, friendly) > 0 and self.count(hq, friendly) == self.count(hq, enemy):
                retained = {u.uid: hq.uid for u in state.agents
                            if previous.get(u.uid) == hq.uid and u.alive
                            and u.entity_type in {'BP_BaseSoldier_C', 'BP_RoboDog_C'}
                            and u.communication_ok() is not False
                            and self.distance(u.position, hq.position) <= 22000}
        super()._assign_garrison(state, include_current)
        # Other secured objectives retain their normal allocation priority.
        retained = {uid: point for uid, point in retained.items() if uid not in self._garrison}
        self._garrison.update(retained)
        self._guard_slots.update({key: point for key, point in slots.items()
                                  if retained.get(key[0]) == key[1]})


class HQGuardEntryRoute(TiedHQGuardRoute):
    """Repair a near-zero return request without relaxing the guarded region.

    Round 10 soldiers stopped just outside the inset HQ box while repeatedly
    requesting its boundary. Those centimetre-scale moves were below the motion
    watchdog's 300 cm threshold. Keep the reached interior slot unchanged, but
    give an outside guard a fixed goal beyond this arrival dead zone. This is a
    navigation request, not proof of mesh clearance or a new capture condition.
    """

    RETURN_DEAD_ZONE = 450.0
    RETURN_INSET = 800.0

    def guard_slot(self, agent, obj):
        point = super().guard_slot(agent, obj)
        extent = self._footprint(obj)
        if (not self.is_hq(obj) or self._garrison.get(agent.uid) != obj.uid
                or not agent.alive or agent.hp <= 0
                or agent.entity_type not in {'BP_BaseSoldier_C', 'BP_RoboDog_C'}
                or agent.communication_ok() is False or extent is None
                or self.inside_garrison(obj, agent.position)
                or self.distance(agent.position, point) > self.RETURN_DEAD_ZONE):
            return point
        x, y, yaw = self._local_point(obj, point)
        if min(extent[0] - abs(x), extent[1] - abs(y)) > self.RETURN_DEAD_ZONE:
            return point
        # Cache the repaired slot; never move a target ahead of the approaching
        # guard. Preserve elevation and handle rotated/small public boxes.
        inner_x, inner_y = (max(size / 2, size - self.RETURN_INSET) for size in extent)
        x, y = max(-inner_x, min(inner_x, x)), max(-inner_y, min(inner_y, y))
        point = (float(obj.position[0]) + x * math.cos(yaw) - y * math.sin(yaw),
                 float(obj.position[1]) + x * math.sin(yaw) + y * math.cos(yaw), float(point[2]))
        self._guard_slots[agent.uid, obj.uid] = point
        return point
