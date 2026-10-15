from __future__ import annotations

import logging

from cssim.protocol import Action
from .AssaultRoute import AssaultRoute
from .UrbanContact import UrbanContact

logger = logging.getLogger(__name__)


class HQEntry:
    """Escalate facade oscillation to an observed, ordered courtyard entrance.

    Static geometry only: 20261009_003121 round 1, friendly dog 346 steps
    108-120 and friendly soldier 101 steps 151-155. Both actually reached the
    courtyard. Enemy locations/UIDs from that recording are never reused.
    """

    PATH = ((-36335, 30318, 6425), (-37026, 31271, 6425),
            (-37200, 32593, 6425), (-37334, 33617, 6425),
            (-37497, 34856, 6425), (-37649, 36093, 6425),
            (-38820, 36237, 6429), (-40107, 36086, 6425),
            (-40591, 34959, 6425), (-41108, 33820, 6425),
            (-41625, 32682, 6425), (-40105, 32287, 6429),
            (-39043, 31926, 6429))
    STALL_STEPS = 18
    ARRIVAL = 650
    VEHICLES = {"BP_MNWS_Vehicle_Armored_C", "BP_MNWS_Vehicle_6x6UGV_C"}
    # In 20261015_135458, cars stopped ~9-10m from requested points while
    # the old 6.5m gate waited indefinitely. This is a policy tolerance based
    # on observed stops, not an assertion about UE's internal acceptance radius.
    VEHICLE_ARRIVAL = 1200
    WAYPOINT_TRIALS = 16
    # Actual red vehicle positions from 20261014_010705: round 1, wheeled
    # vehicle 491, steps 170-181. Keep the lane and its corner order.
    VEHICLE_PATH = ((-34863, 29503, 6335), (-36237, 30315, 6335),
                    (-37043, 31469, 6335), (-36812, 33264, 6335),
                    (-37221, 35581, 6335), (-37713, 36489, 6335),
                    (-39885, 36470, 6335), (-40928, 34988, 6335),
                    (-41209, 33084, 6335))
    # Back out along the reached approach, then join the southern approach
    # used by round 2 red lynx 516, steps 109-160. No enemy coordinates.
    VEHICLE_ALTERNATE = ((-34863, 29503, 6335), (-34453, 28503, 6335),
                         (-33414, 26265, 6318), (-33189, 25821, 6318),
                         (-34275, 25953, 6320), (-35272, 26075, 6335),
                         (-36329, 26203, 6335), (-37380, 26335, 6335),
                         (-37575, 28171, 6335), (-37057, 30668, 6335),
                         (-37557, 33719, 6335), (-37035, 35448, 6335),
                         (-39378, 36583, 6335), (-40973, 35196, 6335),
                         (-41250, 33723, 6335))

    def __init__(self):
        self._frame = None
        self._watch = {}
        self._routes = {}
        self._trials = {}
        self._completed = set()
        self._leg_watch = {}
        self._alternate = set()
        self._exhausted = set()

    def _path(self, agent):
        if agent.entity_type not in self.VEHICLES:
            return self.PATH
        return self.VEHICLE_ALTERNATE if agent.uid in self._alternate else self.VEHICLE_PATH

    def _complete(self, agent, guard, route, urban):
        self._completed.add(agent.uid)
        self._routes.pop(agent.uid, None)
        self._leg_watch.pop(agent.uid, None)
        urban.cancel(agent)
        if guard is not None:
            route._guard_slots[(agent.uid, guard.uid)] = route.clamp_guard_point(guard, agent.position)

    def _vehicle_stalled(self, state, agent, cursor, point):
        distance = AssaultRoute.distance(agent.position, point)
        old = self._leg_watch.get(agent.uid)
        if old is None or old[0] != cursor or distance < old[1] - 200:
            watch = (cursor, distance, 0, state.step)
        else:
            # Count navigation decisions, not elapsed time spent successfully
            # firing. Small circles cannot repeatedly reset the best distance.
            watch = (cursor, old[1], old[2] + int(old[3] != state.step), state.step)
        self._leg_watch[agent.uid] = watch
        return watch[2] >= self.WAYPOINT_TRIALS

    def choose(self, state, agent, route, urban, contacts, *, allow_contact_override=False):
        frame = (state.episode, state.step)
        if self._frame is None or frame[0] != self._frame[0] or frame[1] < self._frame[1]:
            self._watch.clear()
            self._routes.clear()
            self._trials.clear()
            self._completed.clear()
            self._leg_watch.clear()
            self._alternate.clear()
            self._exhausted.clear()
        self._frame = frame
        hq = state.key_object_by_uid(route.order[0]) if route.order else None
        guard = route.garrison_for(state, agent)
        eligible = (hq is not None and route.known_map and agent.entity_type in UrbanContact.GROUND
                    and agent.alive and agent.communication_ok() is not False
                    and (route.stage == 0 or guard is not None and guard.uid == hq.uid)
                    and AssaultRoute.distance(agent.position, hq.position) < 18000
                    and (route.enemy_count(state, hq) > 0 or route.count(hq, "percent") < 1))
        if not eligible:
            self._watch.pop(agent.uid, None)
            self._routes.pop(agent.uid, None)
            self._leg_watch.pop(agent.uid, None)
            return None
        # Navigation recovery must not suppress an armed unit's local contact
        # on nine out of twelve frames. Keep the cursor for when contact ends;
        # ordinary ineffective-fire recovery still handles a blocked shot.
        local = {enemy.uid for enemy in state.perceived_opponents(agent) if enemy.alive}
        ammo = urban.ammo(agent)
        if (not allow_contact_override and (ammo is None or ammo > 0)
                and any(enemy.alive and enemy.uid in local and enemy.entity_type in UrbanContact.GROUND
                        for enemy in contacts)):
            return None
        if agent.uid in self._exhausted:
            return None  # Both recorded approaches failed; allow public-contact navigation.
        path = self._path(agent)
        arrival = self.VEHICLE_ARRIVAL if agent.entity_type in self.VEHICLES else self.ARRIVAL
        ground = [enemy for enemy in contacts if enemy.entity_type in UrbanContact.GROUND]
        effective = any(urban.effective(state, agent, enemy) for enemy in ground)
        distance = AssaultRoute.distance(agent.position, hq.position)
        watch = self._watch.get(agent.uid)
        if watch is None or effective or distance < watch[0] - 1500:
            watch = (distance, state.step)
            self._watch[agent.uid] = watch
        if effective:
            return None  # Effective fire interrupts travel without rewinding it.
        cursor = self._routes.get(agent.uid)
        if cursor is None:
            if agent.uid in self._completed or state.step - watch[1] < self.STALL_STEPS:
                return None
            reached = [i for i, point in enumerate(path)
                       if AssaultRoute.distance(agent.position, point) <= arrival]
            cursor = max(reached) if reached else 0
            self._routes[agent.uid] = cursor
            urban.cancel(agent)
            logger.info("外墙接敌无进展，改走已通行入口: uid=%d step=%d waypoint=%d",
                        agent.uid, state.step, cursor)
        # A fresh local contact gets a bounded firing trial, rather than every
        # shared report cancelling the entrance route again.
        if ground and (ammo is None or ammo > 0) and any(e.uid in local for e in ground):
            trial = self._trials.get(agent.uid, -100)
            if state.step - trial >= 12:
                trial = state.step
                self._trials[agent.uid] = trial
            if state.step - trial < 3:
                return None
        while AssaultRoute.distance(agent.position, path[cursor]) <= arrival:
            if cursor == len(path) - 1:
                self._complete(agent, guard, route, urban)
                return None
            cursor += 1
        if (agent.entity_type in self.VEHICLES
                and self._vehicle_stalled(state, agent, cursor, path[cursor])):
            self._leg_watch.pop(agent.uid, None)
            urban.cancel(agent)
            if agent.uid not in self._alternate:
                blocked = path[cursor]
                self._alternate.add(agent.uid)
                path = self.VEHICLE_ALTERNATE
                cursor = 0
                while cursor < len(path) - 1 and (
                        AssaultRoute.distance(agent.position, path[cursor]) <= arrival
                        or AssaultRoute.distance(blocked, path[cursor]) < 600):
                    cursor += 1
                logger.warning("车辆入口航点无进展，改走南侧已通行路线: uid=%d step=%d waypoint=%d",
                               agent.uid, state.step, cursor)
            elif cursor < len(path) - 1:
                # One blocked waypoint must not monopolize the whole episode.
                # Try the next recorded corner, never skip directly to the HQ.
                cursor += 1
                logger.warning("车辆备用入口航点超时，尝试下一转角: uid=%d step=%d waypoint=%d",
                               agent.uid, state.step, cursor)
            else:
                self._exhausted.add(agent.uid)
                self._routes.pop(agent.uid, None)
                logger.warning("车辆两条入口路线未完成，恢复公开接敌导航: uid=%d step=%d", agent.uid, state.step)
                return None
            # A fallback may itself lead to a corner already reached by a
            # previous escape. Do not reissue that completed waypoint.
            while AssaultRoute.distance(agent.position, path[cursor]) <= arrival:
                if cursor == len(path) - 1:
                    self._complete(agent, guard, route, urban)
                    return None
                cursor += 1
        self._routes[agent.uid] = cursor
        # A full, fixed waypoint lets UE navigation and MotionRecovery finish
        # the corner. Target changes and short local circles cannot reset it.
        return Action.move_at(path[cursor])
