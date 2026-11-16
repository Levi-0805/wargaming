"""Reviewed guard overlays over the immutable historical route."""
import logging
import math

from .Champion6cb5a79.AdaptiveAssault import AdaptiveAssault

from .Champion6cb5a79.AssaultRoute import AssaultRoute
from .Champion6cb5a79.RemnantCapture import RemnantCapture

logger = logging.getLogger(__name__)


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

    HQ_SOLDIER_MIN = 3

    @staticmethod
    def _hq_soldier(unit):
        return (unit.alive and unit.hp > 0 and unit.entity_type == 'BP_BaseSoldier_C'
                and unit.communication_ok() is not False)

    def garrison_needed(self, state, obj):
        needed = super().garrison_needed(state, obj)
        return max(needed, self.HQ_SOLDIER_MIN) if self.is_hq(obj) else needed

    def _assign_garrison(self, state, include_current=False):
        previous = dict(self._garrison)
        super()._assign_garrison(state, include_current)
        hq = state.key_object_by_uid(self.hq_uid)
        if hq is None or hq.uid not in self.order[:self.stage + int(include_current)]:
            return
        if not hq.valid:
            return
        # A previously held HQ can need reinforcements even while occupancy is
        # tied. Never use a tied bar to start a new departure.
        friendly = 'redTeamNum' if state.team == 0 else 'blueteamNum'
        enemy_count = AdaptiveAssault.occupants(state, hq)
        tied_after_departure = (self.hq_secured and hq.uid in self.order[:self.stage]
                                and not self._remnant.active and hq.team is None
                                and self.count(hq, 'percent') >= 1 and enemy_count is not None
                                and enemy_count > 0 and self.count(hq, friendly) == enemy_count)
        if not (self.owned(state, hq) or tied_after_departure):
            return
        assigned = [unit for unit in state.agents if self._garrison.get(unit.uid) == hq.uid]
        soldiers_needed = self.HQ_SOLDIER_MIN - sum(self._hq_soldier(unit) for unit in assigned)
        if soldiers_needed <= 0:
            return
        available = [unit for unit in state.agents if unit.uid not in self._garrison
                     and self._hq_soldier(unit) and self.distance(unit.position, hq.position) <= 22000]
        available.sort(key=lambda unit: (not self.inside_garrison(hq, unit.position),
                                         previous.get(unit.uid) != hq.uid,
                                         self.distance(unit.position, hq.position), unit.uid))
        replaceable = [unit for unit in reversed(assigned) if not self._hq_soldier(unit)]
        capacity = self.garrison_needed(state, hq)
        for soldier in available[:soldiers_needed]:
            if len(assigned) >= capacity and replaceable:
                released = replaceable.pop(0)
                self._garrison.pop(released.uid)
                self._guard_slots.pop((released.uid, hq.uid), None)
                assigned.remove(released)
                leg = min(self.stage, self.order.index(hq.uid) + 1)
                route = self._route_for(released, leg)
                self._cursor[released.uid] = (leg, min(
                    range(len(route)), key=lambda i: self.distance(released.position, route[i])))
                self._waypoint_watch.pop(released.uid, None)
            self._garrison[soldier.uid] = hq.uid
            assigned.append(soldier)

    # The immutable historical snapshot has no departure hook. Preserve its
    # update order and apply the stricter HQ check before confirming departure.
    def update(self, state):
        if self._episode != state.episode or (self._step is not None and state.step < self._step):
            self._initialize(state)
            self._step = None
        elif not self.order:
            self._initialize(state)
        self._episode = state.episode
        if self._step == state.step:
            return
        consecutive = self._step is not None and state.step == self._step + 1
        self._step = state.step
        if not consecutive:
            self._confirmed = 0
            self._clear.clear()
            self._incidental.clear()
        self._select(state)
        for uid in self.order:
            point = state.key_object_by_uid(uid)
            clear = (point is not None and self.count(point, 'percent') >= 1
                     and AdaptiveAssault.occupants(state, point) == 0
                     and (point.team is None or point.team == state.team)
                     and not any(enemy.alive and enemy.entity_type in {
                         'BP_BaseSoldier_C', 'BP_RoboDog_C', 'BP_MNWS_Vehicle_Armored_C',
                         'BP_MNWS_Vehicle_6x6UGV_C'} and self.distance(enemy.position, point.position) <= 12000
                         for enemy in state.visible_opponents))
            self._clear[uid] = self._clear.get(uid, 0) + 1 if clear else 0
        self._secure_incidental(state)
        # A full progress bar can be reversed immediately by surviving defenders.
        # Revisit a lost earlier objective instead of leaving vehicles en route to B.
        lost = next((i for i, uid in enumerate(self.order[:self.stage])
                     if (previous := state.key_object_by_uid(uid)) is not None
                     and (previous.team != state.team if previous.team is not None
                          else self.count(previous, "percent") < 1)), None)
        self._lost = self._lost + 1 if lost is not None else 0
        if self._lost >= 2:
            if self.hq_first and self.order[lost] == self.hq_uid:
                held = [uid for uid in self.order[:self.stage] if uid != self.hq_uid
                        and (holding := state.key_object_by_uid(uid)) is not None and self.owned(state, holding)]
                pending = [uid for uid in self.order if uid not in held and uid != self.hq_uid]
                self.hq_secured = False
                self._remnant = RemnantCapture()
                self._reorder(state, (*held, self.hq_uid, *pending), len(held))
                self._lost = 0
                logger.warning('指挥所失守，暂停次要目标并重新夺回: step=%d', state.step)
            elif self.adaptive:
                # Reassess lost holdings without recalling the entire column
                # into a newly fortified point. Its surviving guards may join.
                lost_uid = self.order[lost]
                order = list(self.order)
                order.remove(lost_uid)
                order.append(lost_uid)
                self._reorder(state, order, self.stage - 1)
                self._lost = 0
                self._select(state)
                logger.warning('已占目标失守，加入待评估目标: step=%d uid=%d', state.step, lost_uid)
            else:
                self.stage, self._confirmed, self._lost = lost, 0, 0
                route = self._routes[self.stage]
                for unit in state.agents:
                    cursor = self._cursor.get(unit.uid)
                    if cursor is not None and cursor[0] > self.stage:
                        self._cursor[unit.uid] = (self.stage, min(
                            range(len(route)), key=lambda i: self.distance(unit.position, route[i])))
                logger.warning("已占目标失守，停止转场: step=%d uid=%d", state.step, self.order[self.stage])
        obj = self.objective(state)
        if obj is None:
            return
        friendly_field, enemy_field = (("redTeamNum", "blueteamNum") if state.team == 0
                                       else ("blueteamNum", "redTeamNum"))
        ours = self.owned(state, obj)
        # A full bar with surviving defenders is only a contested foothold
        # at ANY point. Oct20 rounds 7/10/20 left B for A with 9-10 defenders
        # still inside, and never received B's capture reward.
        try:
            empty = float(obj.raw[enemy_field]) == 0
        except (KeyError, TypeError, ValueError):
            empty = False
        secure = ours and empty and self.count(obj, "percent") >= 1.0
        # Reserve defenders before departure, not after everybody has started
        # moving to B. A label is not evidence that a unit is inside the point.
        self._assign_garrison(state, include_current=ours)
        guards = [unit for unit in state.agents if self._garrison.get(unit.uid) == obj.uid]
        ready = len(guards) >= self.garrison_needed(state, obj) and all(
            self.inside_garrison(obj, unit.position) for unit in guards)
        if self.is_hq(obj):
            # UE reports aggregate occupancy, so unidentified occupants must
            # count toward the enemy soldier limit.
            enemy_occupants = AdaptiveAssault.occupants(state, obj)
            ready = (ready and enemy_occupants in (0, 1)
                     and self.count(obj, friendly_field) >= self.garrison_needed(state, obj)
                     and sum(self._hq_soldier(unit) for unit in guards) >= self.HQ_SOLDIER_MIN)
        # Keep the historical stability interval and friendly advantage for
        # departure with one enemy occupant still in the HQ.
        held = (self.adaptive and self.is_hq(obj) and ours and ready
                and self.count(obj, 'percent') >= 1.0
                and self.enemy_count(state, obj) == 1
                and self.count(obj, friendly_field) >= 2*self.enemy_count(state, obj)+4
                and sum(u.alive and u.entity_type in {'BP_BaseSoldier_C', 'BP_RoboDog_C'}
                        and u.communication_ok() is not False and u.uid not in self._garrison
                        for u in state.agents) >= 6)
        self._hq_foothold_steps = self._hq_foothold_steps + 1 if held else 0
        if self._hq_foothold_steps >= self.HQ_STABLE_FOOTHOLD_STEPS:
            secure = True
        self._confirmed = self._confirmed + 1 if secure and ready else 0
        if self.is_hq(obj) and ours and not secure and state.step % 20 == 0:
            logger.info("指挥所仍需清场，暂不转场: step=%d enemy_inside=%d guards=%d",
                        state.step, self.enemy_count(state, obj), len(guards))
        if (self._confirmed >= self.CAPTURE_CONFIRMATIONS and ready
                and self.stage + 1 < len(self.order) and not self._remnant.active):
            if self.is_hq(obj):
                self.hq_secured = True
            self.stage += 1
            self._confirmed = 0
            self._select(state, next_objective=True)
            logger.info("据点占领已确认，主力转向下一目标: step=%d uid=%d", state.step, self.order[self.stage])

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
