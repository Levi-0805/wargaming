from __future__ import annotations

import logging
import math

from .AdaptiveAssault import AdaptiveAssault
from .RemnantCapture import RemnantCapture

logger = logging.getLogger(__name__)


class AssaultRoute:
    """One objective sequence, with independent, monotonic route cursors.

    Coordinates are cm. LOWER samples actual ground trajectories in
    Output_20261008_230451 (136/176). HQ_B samples the ground path in
    Output_20260924_150433 episode 2 (671). Only map geometry is reused;
    runtime decisions never read replay enemies. Other maps use UE navigation.
    """

    LOWER = (
        (77387, 36261, 7394), (65185, 36303, 6082),
        (59734, 36261, 5646), (49548, 34208, 5747),
        (38220, 32771, 5998), (31881, 31436, 6411),
        (20366, 30043, 6408), (9200, 28897, 6408),
        (2534, 28097, 6408), (-6470, 30114, 6425),
        (-17959, 27728, 6425), (-29782, 25170, 6426),
    )
    # Approach the right edge, pass along the front, then turn back into the
    # courtyard from the other end. Friendly soldier 136, 20261014_010705
    # round 1, steps 95-263 supplies these ground positions. The old northern
    # corner (-40107, 36086) is not part of either ingress corridor.
    HQ_FRONT = ((-34879, 30436, 6425), (-36235, 30632, 6425),
                (-36090, 28203, 6425), (-36852, 26177, 6425),
                (-40082, 26221, 6425), (-43531, 26968, 6425),
                (-44927, 26068, 6429), (-45008, 27567, 6429),
                (-44228, 28395, 6429), (-43891, 29266, 6429),
                (-43066, 30654, 6429), (-41156, 30883, 6429))
    # Same-side alternative from friendly dog 361 in the same round, steps
    # 95-303. These are navigation requests: vehicle traversal of the new
    # corridor still needs live validation, not a claim about mesh clearance.
    HQ_FRONT_ALTERNATE = ((-34497, 29404, 6387), (-35084, 28248, 6387),
                          (-36004, 26175, 6387), (-39469, 26175, 6387),
                          (-42213, 25438, 6387), (-44132, 25509, 6387),
                          (-45100, 26121, 6391), (-44993, 26737, 6391),
                          (-44747, 27883, 6391), (-43938, 29102, 6391),
                          (-43860, 30336, 6391), (-42834, 30934, 6391),
                          (-41052, 30988, 6391))
    HQ_B = (
        (-37782, 24389, 6408), (-46037, 15885, 6425),
        (-50063, 5399, 6425), (-49816, -6016, 5943),
        (-53046, -18007, 5880), (-47310, -27482, 5617),
    )
    # Friendly soldier 821, 20261014_211207 round 2, steps 190/195/220/
    # 225/230/235. This exit passed the wall where 776/771/861 stalled.
    B_A = ((-51455, -25065, 5613), (-46251, -27855, 5617),
           (-41155, -30706, 5620), (-38673, -35629, 5620),
           (-34023, -34923, 5763), (-30334, -29802, 5684))
    ANCHORS = ((-41810, 32275), (-45450, -30945), (-3000, -27530))
    ARRIVAL = 1600.0
    CAPTURE_CONFIRMATIONS = 4
    GARRISON_CORE = 2500.0
    GARRISON_MIN = 4
    GARRISON_MARGIN = 3
    SECONDARY_GARRISON = 2
    CLEAR_STEPS = 8
    WAYPOINT_STALL_CALLS = 20
    BYPASS_RADIUS = 3500.0

    def __init__(self, *, adaptive=False, hq_first=True):
        self.adaptive = bool(adaptive)
        self.hq_first = bool(hq_first)
        self.hq_secured = False
        self._strategy = AdaptiveAssault() if adaptive else None
        self.hq_uid = None
        self._map_order = ()
        self._adaptive_paths = {}
        self._replanned = False
        self._episode = None
        self._step = None
        self.order = ()
        self.stage = 0
        self._confirmed = 0
        self._cursor = {}
        self._routes = ()
        self.known_map = False
        self._garrison = {}
        self._guard_slots = {}
        self._waypoint_watch = {}
        self._lost = 0
        self._clear = {}
        self._incidental = {}
        self._remnant = RemnantCapture()

    @staticmethod
    def distance(left, right):
        return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))

    @staticmethod
    def count(obj, field):
        try:
            value = float(obj.raw.get(field, 0) or 0)
            return value if math.isfinite(value) else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _initialize(self, state):
        self.order, self.stage, self._confirmed = (), 0, 0
        self.hq_secured = False
        self._cursor.clear()
        self._garrison.clear()
        self._guard_slots.clear()
        self._waypoint_watch.clear()
        self._lost = 0
        self._clear.clear()
        self._incidental.clear()
        self._remnant = RemnantCapture()
        self._adaptive_paths.clear()
        self._replanned = False
        self._strategy = AdaptiveAssault() if self.adaptive else None
        self.hq_uid = None
        self._map_order = ()
        self._routes = ()
        self.known_map = False
        objects = [obj for obj in state.key_objects if obj.valid]
        if not objects:
            return
        matched = [min(objects, key=lambda obj: self.distance(obj.position, point)) for point in self.ANCHORS]
        self.known_map = len({obj.uid for obj in matched}) == 3 and all(
            self.distance(obj.position, point) < 500 for obj, point in zip(matched, self.ANCHORS)
        )
        if self.known_map:
            ordered = matched  # HQ -> B -> A, left to right in the supplied view.
        else:
            hq = next((obj for obj in objects if any(
                word in str(obj.raw.get("rSVD1", "")).lower()
                for word in ("指挥", "headquarter", "hq")
            )), objects[0])
            ordered = [hq]
            remaining = [obj for obj in objects if obj.uid != hq.uid]
            while remaining:
                target = min(remaining, key=lambda obj: self.distance(ordered[-1].position, obj.position))
                ordered.append(target)
                remaining = [obj for obj in remaining if obj.uid != target.uid]
        self.order = tuple(obj.uid for obj in ordered)
        self.hq_uid = ordered[0].uid
        self._map_order = self.order
        self._routes = tuple(
            tuple(prefix) + (tuple(float(v) for v in obj.position),)
            for obj, prefix in zip(ordered, (self.LOWER + self.HQ_FRONT, self.HQ_B, self.B_A)
                                   if self.known_map else [()] * len(ordered))
        )
        logger.info("统一推进目标顺序: %s 下路坐标匹配=%s", self.order, self.known_map)

    def is_hq(self, obj):
        return obj is not None and obj.uid == self.hq_uid

    def is_base_b(self, obj):
        if obj is None:
            return False
        if self.known_map and len(self._map_order) > 1:
            return obj.uid == self._map_order[1]
        name = str(obj.raw.get("rSVD1", "")).upper().replace(" ", "")
        return name in {"B", "BASEB", "军事基地B"}

    def owned(self, state, obj):
        friendly, enemy = (("redTeamNum", "blueteamNum") if state.team == 0
                           else ("blueteamNum", "redTeamNum"))
        return obj.team == state.team if obj.team is not None else (
            self.count(obj, 'percent') >= 1.0 and self.count(obj, friendly) > self.count(obj, enemy))

    def recon_objective(self, state, agent=None):
        return self._strategy.recon_objective(state, self, agent) if self._strategy else self.objective(state)

    def recon_pending(self, state):
        return self._strategy.recon_pending(state, self) if self._strategy else False

    def _reorder(self, state, order, stage=None):
        """Keep confirmed holdings separate from the pending attack order."""
        self.order = tuple(order)
        self._replanned = True
        if stage is not None:
            self.stage = stage
        self._confirmed = 0
        self._cursor.clear()
        self._adaptive_paths.clear()
        self._waypoint_watch.clear()
        self._routes = tuple(
            (self.LOWER + self.HQ_FRONT if self.known_map and uid == self.hq_uid else ())
            + (tuple(float(v) for v in state.key_object_by_uid(uid).position),)
            for uid in self.order)

    def _select(self, state, *, next_objective=False):
        if self._strategy is None:
            return
        if self.hq_first and not self.hq_secured:
            # Reconnaissance can change approach/fire decisions, never the
            # user's mandatory first objective. A full bar alone is not enough.
            self._strategy.observe(state)
            return
        if self._remnant.update(state, self):
            return
        chosen = self._strategy.select(state, self, next_objective=next_objective)
        if chosen is not None:
            suffix = list(self.order[self.stage:])
            suffix.remove(chosen)
            self._reorder(state, (*self.order[:self.stage], chosen, *suffix))

    def _route_for(self, agent, leg):
        if not self.adaptive or not self.known_map or not self._replanned:
            return self._routes[leg]
        key = (agent.uid, leg, self.order[leg])
        if key in self._adaptive_paths:
            return self._adaptive_paths[key]
        # Reuse sampled lower-road and connecting corridors in either direction.
        # The short LOWER -> HQ_B join bypasses the HQ centre; UE navigation
        # still resolves each ground segment. Other maps use UE navigation.
        corridor = self.LOWER + self.HQ_B + self.B_A
        destination = self._map_order.index(self.order[leg])
        end = (len(self.LOWER) - 1, len(self.LOWER) + len(self.HQ_B) - 1,
               len(corridor) - 1)[destination]
        target = self._routes[leg][-1]
        if destination == 0 and self.distance(agent.position, self.HQ_FRONT[-1]) <= 1800:
            path = (target,)
        elif destination == 0 and self.distance(agent.position, target) <= 18000:
            path = self.HQ_FRONT + (target,)
        elif destination != 0 and self.distance(agent.position, target) <= self.GARRISON_CORE:
            path = (target,)
        else:
            start = 1 if agent.position[1] < 26000 and agent.position[0] > 50000 else min(
                range(len(corridor)), key=lambda i: self.distance(agent.position, corridor[i]))
            indices = range(start, end + 1) if start <= end else range(start, end - 1, -1)
            path = (tuple(corridor[i] for i in indices)
                    + (self.HQ_FRONT if destination == 0 else ()) + (target,))
        self._adaptive_paths[key] = path
        return path

    def transit_objectives(self, state, origin, target):
        """Public capture boxes crossed by the actual sampled corridor.

        A is reached through the B corridor. An empty A does not make that
        approach cheap when B is defended. No hidden opponent positions.
        """
        if not self.known_map or target.uid not in self._map_order:
            return ()
        corridor = self.LOWER + self.HQ_B + self.B_A
        end = (len(self.LOWER)-1, len(self.LOWER)+len(self.HQ_B)-1,
               len(corridor)-1)[self._map_order.index(target.uid)]
        start = min(range(len(corridor)), key=lambda i: self.distance(origin, corridor[i]))
        indices = range(start, end+1) if start <= end else range(start, end-1, -1)
        path = [tuple(origin), *(corridor[i] for i in indices), tuple(target.position)]
        crossed = []
        for obj in state.key_objects:
            if not obj.valid or obj.uid == target.uid or self.owned(state, obj):
                continue
            found = False
            for left, right in zip(path, path[1:]):
                samples = max(1, math.ceil(self.distance(left, right) / 2000))
                if any(self.inside_garrison(obj, tuple(left[j] + (right[j]-left[j])*i/samples
                                                        for j in range(3)))
                       for i in range(samples+1)):
                    found = True
                    break
            if found:
                crossed.append(obj)
        return tuple(crossed)

    def _secure_incidental(self, state):
        if self._remnant.active or not self.adaptive or self.hq_first and not self.hq_secured:
            return
        # A column can capture a base it crosses before reaching its intended
        # destination. Promote that real holding, keep the intended objective,
        # and reserve defenders rather than marching everyone out of it.
        for uid in self.order[self.stage+1:]:
            obj = state.key_object_by_uid(uid)
            ready = obj is not None and self.owned(state, obj) and AdaptiveAssault.occupants(state, obj) == 0
            if ready:
                inside = sum(u.alive and u.entity_type in {'BP_BaseSoldier_C', 'BP_RoboDog_C'}
                             and u.communication_ok() is not False and self.inside_garrison(obj, u.position)
                             for u in state.agents)
                ready = inside >= self.garrison_needed(state, obj)
            self._incidental[uid] = self._incidental.get(uid, 0) + 1 if ready else 0
            if self._incidental[uid] >= self.CAPTURE_CONFIRMATIONS:
                pending = [other for other in self.order[self.stage:] if other != uid]
                self._reorder(state, (*self.order[:self.stage], uid, *pending), self.stage+1)
                self._incidental.pop(uid, None)
                logger.info('沿途据点占领确认，留守并继续当前主攻: step=%d uid=%d', state.step, uid)
                break

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
        enemy_field = 'blueteamNum' if state.team == 0 else 'redTeamNum'
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
            ready = ready and self.count(obj, friendly_field) >= self.garrison_needed(state, obj)
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

    def garrison_needed(self, state, obj):
        if self._remnant.active and obj.uid == self._remnant.target_uid:
            return max(1, min(2, len(self._remnant.capturers(state))))
        if self._remnant.active and self.is_hq(obj):
            return 1 if len(self._remnant.capturers(state)) > 4 else 0
        if self._clear.get(obj.uid, 0) >= self.CLEAR_STEPS:
            if self.adaptive:
                capturers = sum(u.alive and u.entity_type in {'BP_BaseSoldier_C', 'BP_RoboDog_C'}
                                and u.communication_ok() is not False for u in state.agents)
                pending = any(o.valid and not self.owned(state, o) for o in state.key_objects)
                if pending and capturers <= self.SECONDARY_GARRISON * len(self.order):
                    return 1  # Known clear only; retain capacity to capture remaining points.
            if not self.is_hq(obj):
                return self.SECONDARY_GARRISON
            if self.adaptive:
                return 3
        return max(self.GARRISON_MIN, self.enemy_count(state, obj) + self.GARRISON_MARGIN)

    def enemy_count(self, state, obj):
        return int(self.count(obj, "blueteamNum" if state.team == 0 else "redTeamNum"))

    def _assign_garrison(self, state, include_current=False):
        previous, self._garrison = self._garrison, {}
        for uid in self.order[:self.stage + int(include_current)]:
            if (self._remnant.active and uid != self._remnant.target_uid
                    and not (uid == self.hq_uid and len(self._remnant.capturers(state)) > 4)):
                continue
            obj = state.key_object_by_uid(uid)
            if obj is None or not obj.valid or self.adaptive and not self.owned(state, obj):
                continue
            nearby = [unit for unit in state.agents if unit.alive
                      and unit.entity_type in {"BP_BaseSoldier_C", "BP_RoboDog_C"}
                      and unit.communication_ok() is not False
                      and unit.uid not in self._garrison
                      and self.distance(unit.position, obj.position) <= 22000]
            needed = self.garrison_needed(state, obj)
            nearby.sort(key=lambda unit: (
                not self.inside_garrison(obj, unit.position),
                previous.get(unit.uid) != uid,
                self.distance(unit.position, obj.position), unit.uid))
            chosen = nearby[:needed]
            # Keep a living soldier with a small unmanned garrison when one is
            # available locally, rather than retaining two dependent dogs alone.
            if needed == self.SECONDARY_GARRISON and chosen:
                soldier = next((unit for unit in nearby if unit.entity_type == 'BP_BaseSoldier_C'
                                and self.inside_garrison(obj, unit.position)), None)
                dog = next((unit for unit in nearby if unit.entity_type == 'BP_RoboDog_C'
                            and self.inside_garrison(obj, unit.position)), None)
                if soldier is not None and dog is not None:
                    chosen = [soldier, dog]
                elif soldier is not None and not any(unit.uid == soldier.uid for unit in chosen):
                    chosen[-1] = soldier
            self._garrison.update({unit.uid: uid for unit in chosen})
        # Released defenders join the next leg from their actual position.
        # Their old approach cursor must not send them back towards the HQ.
        for unit in state.agents:
            old = previous.get(unit.uid)
            if old is not None and unit.uid not in self._garrison and old in self.order:
                leg = min(self.stage, self.order.index(old) + 1)
                route = self._route_for(unit, leg)
                self._cursor[unit.uid] = (leg, min(range(len(route)),
                                                 key=lambda i: self.distance(unit.position, route[i])))
                self._waypoint_watch.pop(unit.uid, None)
        self._guard_slots = {key: point for key, point in self._guard_slots.items()
                             if self._garrison.get(key[0]) == key[1]}

    @staticmethod
    def _footprint(obj):
        # UE supplies box extents in this scenario (HQ: 77m x 65m).
        # Keep a 4m inset. Never assume the building centre is walkable.
        scale = obj.raw.get("scale", {})
        if isinstance(scale, dict) and float(scale.get("x", 0)) > 400 and float(scale.get("y", 0)) > 400:
            return (float(scale["x"]) - 400, float(scale["y"]) - 400)
        return None

    @staticmethod
    def _local_point(obj, point):
        yaw = math.radians(float(obj.raw.get("rotation", {}).get("yaw", 0)))
        dx, dy = float(point[0]) - obj.position[0], float(point[1]) - obj.position[1]
        return dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw), yaw

    def inside_garrison(self, obj, point):
        extent = self._footprint(obj)
        if extent:
            x, y, _ = self._local_point(obj, point)
            # Rotating a clamped boundary point back can exceed its extent by
            # ~2e-12 cm. Do not reject our own guard position on round-off.
            return abs(x) <= extent[0] + 1e-6 and abs(y) <= extent[1] + 1e-6
        return self.distance(obj.position, point) <= self.GARRISON_CORE

    def guard_slot(self, agent, obj):
        key = (agent.uid, obj.uid)
        if self._footprint(obj):
            # An actual reached position is a better hold point than an
            # unverified central waypoint inside a building.
            return self._guard_slots.setdefault(key, self.clamp_guard_point(obj, agent.position))
        # Small fixed interior slots, independent of casualties and target
        # movement. Do not send defenders chasing a contact outside the point.
        slot = agent.team_index % 9
        return (float(obj.position[0]) + (slot % 3 - 1) * 700,
                float(obj.position[1]) + (slot // 3 - 1) * 700, float(obj.position[2]))

    def clamp_guard_point(self, obj, point):
        extent = self._footprint(obj)
        if extent:
            x, y, yaw = self._local_point(obj, point)
            x, y = max(-extent[0], min(extent[0], x)), max(-extent[1], min(extent[1], y))
            return (float(obj.position[0]) + x * math.cos(yaw) - y * math.sin(yaw),
                    float(obj.position[1]) + x * math.sin(yaw) + y * math.cos(yaw), float(point[2]))
        distance = self.distance(obj.position, point)
        ratio = min(1.0, (self.GARRISON_CORE - 400) / max(distance, 1.0))
        return (float(obj.position[0]) + (float(point[0]) - obj.position[0]) * ratio,
                float(obj.position[1]) + (float(point[1]) - obj.position[1]) * ratio, float(obj.position[2]))

    def garrison_for(self, state, agent):
        uid = self._garrison.get(agent.uid)
        return state.key_object_by_uid(uid) if uid is not None else None

    def departing(self, state, agent):
        return self.stage > 0 and agent.uid not in self._garrison and any(
            obj is not None and self.distance(agent.position, obj.position) <= 25000
            for obj in (state.key_object_by_uid(uid) for uid in self.order[:self.stage])
        )

    def objective(self, state):
        return state.key_object_by_uid(self.order[self.stage]) if self.order else None

    def join_hq_front(self, agent, *, completed=False):
        """Hand off the entrance cursor without returning to the lower road."""
        leg = self.order.index(self.hq_uid)
        route = self._route_for(agent, leg)
        front = [i for i, point in enumerate(route) if point in self.HQ_FRONT]
        index = (len(route) - 1 if completed or not front else
                 min(front, key=lambda i: self.distance(agent.position, route[i])))
        self._cursor[agent.uid] = (leg, index)
        self._waypoint_watch.pop(agent.uid, None)

    def point(self, state, agent, *, ignore_garrison=False):
        """Rear units finish the approach even after the front captures HQ."""
        if not self.order:
            return None
        remnant = self._remnant.point(state, agent, self)
        if remnant is not None:
            return remnant
        garrison = self.garrison_for(state, agent)
        if garrison is not None and not ignore_garrison:
            return tuple(float(v) for v in garrison.position)
        position = agent.position
        if agent.uid not in self._cursor:
            # A unit already at a captured point can join its departure route;
            # rear units still finish the original lower-road approach.
            leg = max((i + 1 for i, uid in enumerate(self.order[:self.stage])
                       if (obj := state.key_object_by_uid(uid)) is not None
                       and self.distance(position, obj.position) <= 15000), default=0)
            if self.adaptive and self._replanned:
                leg = self.stage
            route = self._route_for(agent, leg)
            if self.adaptive and self._replanned:
                index = 0
            elif self.known_map:
                # Join at the observed road entrance, not west through the
                # buildings adjacent to the x~60000,y~12000 spawn.
                index = 1 if leg == 0 and position[1] < 26000 and position[0] > 50000 else min(
                    range(len(route)), key=lambda i: self.distance(position, route[i])
                )
            else:
                index = 0
            self._cursor[agent.uid] = (leg, index)
        leg, index = self._cursor[agent.uid]
        if self.adaptive and self._replanned and leg < self.stage:
            leg, index = self.stage, 0
            self._waypoint_watch.pop(agent.uid, None)
        route = self._route_for(agent, leg)
        # Combat may carry a unit past a waypoint. Rejoin at an actually
        # reached later point instead of ordering it back down the road.
        ingress = self.known_map and self.order[leg] == self.hq_uid
        front = {i for i, point in enumerate(route) if point in self.HQ_FRONT}
        later = [i for i in range(index + 1, len(route))
                 if (not ingress or i not in front and i != len(route) - 1)
                 if self.distance(position, route[i]) <= self.ARRIVAL]
        if later:
            index = max(later)
        if leg < self.stage and self.distance(position, route[-1]) <= 15000:
            leg += 1
            route = self._route_for(agent, leg)
            index = min(range(len(route)), key=lambda i: self.distance(position, route[i]))
            front = {i for i, point in enumerate(route) if point in self.HQ_FRONT}
        arrival = (1200 if 'Vehicle' in agent.entity_type else 650) if index in front else self.ARRIVAL
        while self.distance(position, route[index]) <= arrival:
            if index + 1 < len(route):
                index += 1
            elif leg < self.stage:
                leg, index = leg + 1, 0
                route = self._route_for(agent, leg)
                front = {i for i, point in enumerate(route) if point in self.HQ_FRONT}
            else:
                break
            arrival = (1200 if 'Vehicle' in agent.entity_type else 650) if index in front else self.ARRIVAL
        # Intermediate road samples are not objectives. A unit stuck 18-20m
        # from a 16m arrival gate must not retry that sample all match. Only
        # relax a nearby transit waypoint after repeated failed approaches;
        # never skip the final capture target or change the objective stage.
        if self.known_map and index < len(route) - 1 and agent.communication_ok() is not False:
            distance = self.distance(position, route[index])
            key = (leg, index)
            watch = self._waypoint_watch.get(agent.uid)
            if (watch is None or watch[0] != key or state.step - watch[3] > 2
                    or distance < watch[1] - 200):
                watch = (key, distance, 0, state.step)
            elif watch[3] != state.step:
                watch = (key, watch[1], watch[2] + 1, state.step)
            self._waypoint_watch[agent.uid] = watch
            if watch[2] >= self.WAYPOINT_STALL_CALLS:
                # Once on the entrance, advance only one corner on timeout.
                # Never jump across the entire return leg toward the centre.
                nearby = [i for i in (range(index, index + 1) if index in front
                                      else range(index, len(route) - 1))
                          if self.distance(position, route[i]) <= self.BYPASS_RADIUS]
                if nearby:
                    old_index, index = index, max(nearby) + 1
                    self._waypoint_watch.pop(agent.uid, None)
                    logger.warning("近路点反复无进展，接续下段: uid=%d step=%d leg=%d old=%d next=%d",
                                   agent.uid, state.step, leg, old_index, index)
        self._cursor[agent.uid] = (leg, index)
        return route[index]
