"""One guard-retention experiment over the immutable historical route."""
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
