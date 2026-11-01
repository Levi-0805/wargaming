from __future__ import annotations

import logging

from cssim.protocol import Action
from .AssaultRoute import AssaultRoute
from .UrbanContact import UrbanContact

logger = logging.getLogger(__name__)


class HQEntry:
    """Use the front-side return entrance before ordinary HQ navigation.

    Only friendly historical ground positions define the corridor. Active
    local fire pauses its cursor; navigation failures try the same-side
    alternative for every ground type, never the retired northern bend.
    """

    PATH = AssaultRoute.HQ_FRONT
    ALTERNATE = AssaultRoute.HQ_FRONT_ALTERNATE
    ARRIVAL = 650
    VEHICLES = {"BP_MNWS_Vehicle_Armored_C", "BP_MNWS_Vehicle_6x6UGV_C"}
    # In 20261015_135458, cars stopped ~9-10m from requested points while
    # the old 6.5m gate waited indefinitely. This is a policy tolerance based
    # on observed stops, not an assertion about UE's internal acceptance radius.
    VEHICLE_ARRIVAL = 1200
    WAYPOINT_TRIALS = 16
    # Preserve the existing vehicle height convention while changing the
    # horizontal corridor. Foot passage does not establish vehicle clearance.
    VEHICLE_PATH = tuple((x, y, 6335) for x, y, _ in PATH)
    VEHICLE_ALTERNATE = tuple((x, y, 6335) for x, y, _ in ALTERNATE)

    def __init__(self):
        self._frame = None
        self._watch = {}
        self._routes = {}
        self._trials = {}
        self._completed = set()
        self._leg_watch = {}
        self._alternate = set()
        self._exhausted = set()
        self._hq_secured = False

    def _path(self, agent):
        if agent.entity_type not in self.VEHICLES:
            return self.ALTERNATE if agent.uid in self._alternate else self.PATH
        return self.VEHICLE_ALTERNATE if agent.uid in self._alternate else self.VEHICLE_PATH

    def begin(self, state, route):
        frame = (state.episode, state.step)
        if (self._frame is None or frame[0] != self._frame[0] or frame[1] < self._frame[1]
                or self._hq_secured and not route.hq_secured):
            self._watch.clear()
            self._routes.clear()
            self._trials.clear()
            self._completed.clear()
            self._leg_watch.clear()
            self._alternate.clear()
            self._exhausted.clear()
        self._frame = frame
        self._hq_secured = route.hq_secured

    def maneuvering(self, state, agent, route):
        return self.pending(state, agent, route) and agent.uid not in self._exhausted

    def pending(self, state, agent, route):
        """Keep both dedicated entry and its ordinary-route fallback outside
        provisional garrison constraints until the return entrance is reached.
        """
        self.begin(state, route)
        hq = state.key_object_by_uid(route.hq_uid) if route.hq_uid is not None else None
        objective = route.objective(state)
        return (hq is not None and route.known_map and objective is not None and objective.uid == hq.uid
                and (route.enemy_count(state, hq) > 0 or route.count(hq, "percent") < 1)
                and AssaultRoute.distance(agent.position, hq.position) < 18000
                and (agent.uid in self._routes or agent.uid in self._exhausted)
                and agent.uid not in self._completed and agent.alive
                and agent.communication_ok() is not False)

    def _complete(self, agent, guard, route, urban):
        self._completed.add(agent.uid)
        self._routes.pop(agent.uid, None)
        self._leg_watch.pop(agent.uid, None)
        route.join_hq_front(agent, completed=True)
        urban.cancel(agent)
        if guard is not None:
            route._guard_slots[(agent.uid, guard.uid)] = route.clamp_guard_point(guard, agent.position)

    def _waypoint_stalled(self, state, agent, cursor, point):
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
        self.begin(state, route)
        hq = state.key_object_by_uid(route.hq_uid) if route.hq_uid is not None else None
        objective = route.objective(state)
        guard = route.garrison_for(state, agent)
        eligible = (hq is not None and route.known_map and agent.entity_type in UrbanContact.GROUND
                    and agent.alive and agent.communication_ok() is not False
                    and objective is not None and objective.uid == hq.uid
                    and AssaultRoute.distance(agent.position, hq.position) < 18000
                    and (route.enemy_count(state, hq) > 0 or route.count(hq, "percent") < 1))
        if not eligible:
            self._watch.pop(agent.uid, None)
            self._routes.pop(agent.uid, None)
            self._leg_watch.pop(agent.uid, None)
            return None
        path = self._path(agent)
        arrival = self.VEHICLE_ARRIVAL if agent.entity_type in self.VEHICLES else self.ARRIVAL
        cursor = self._routes.get(agent.uid)
        if cursor is None:
            if agent.uid in self._completed:
                return None
            # Already inside the return entrance: do not recall an existing
            # courtyard unit to the outside of the building.
            if AssaultRoute.distance(agent.position, path[-1]) <= 1800:
                self._complete(agent, guard, route, urban)
                return None
            if agent.uid in self._exhausted:
                return None  # Same-side ordinary navigation takes over below.
            reached = [i for i, point in enumerate(path)
                       if AssaultRoute.distance(agent.position, point) <= arrival]
            cursor = max(reached) if reached else 0
            self._routes[agent.uid] = cursor
            urban.cancel(agent)
            logger.info("从右侧接近正面，再向左后方回折入内: uid=%d step=%d waypoint=%d",
                        agent.uid, state.step, cursor)
        # Establish ingress before a first-frame firefight, otherwise a new
        # garrison assignment can pull the unit directly through the facade.
        local = {enemy.uid for enemy in state.perceived_opponents(agent) if enemy.alive}
        ammo = urban.ammo(agent)
        ground = [enemy for enemy in contacts if enemy.entity_type in UrbanContact.GROUND]
        if (any(urban.effective(state, agent, enemy) for enemy in ground)
                or not allow_contact_override and (ammo is None or ammo > 0)
                and any(enemy.alive and enemy.uid in local for enemy in ground)):
            return None
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
        if self._waypoint_stalled(state, agent, cursor, path[cursor]):
            self._leg_watch.pop(agent.uid, None)
            urban.cancel(agent)
            if agent.uid not in self._alternate:
                blocked = path[cursor]
                self._alternate.add(agent.uid)
                path = self._path(agent)
                # Join the corresponding part of the alternative; a blockage
                # near the return entrance must not recall the unit to the start.
                cursor = min(range(len(path)), key=lambda i: AssaultRoute.distance(blocked, path[i]))
                while cursor < len(path) - 1 and (
                        AssaultRoute.distance(agent.position, path[cursor]) <= arrival
                        or AssaultRoute.distance(blocked, path[cursor]) < 600):
                    cursor += 1
                logger.warning("入口航点无进展，改走同侧正面备用路线: uid=%d step=%d waypoint=%d",
                               agent.uid, state.step, cursor)
            elif cursor < len(path) - 1:
                # One blocked waypoint must not monopolize the whole episode.
                # Try the next recorded corner, never skip directly to the HQ.
                cursor += 1
                logger.warning("正面备用入口航点超时，尝试下一转角: uid=%d step=%d waypoint=%d",
                               agent.uid, state.step, cursor)
            else:
                self._exhausted.add(agent.uid)
                self._routes.pop(agent.uid, None)
                route.join_hq_front(agent)
                logger.warning("正面入口尝试未完成，交回同侧接近导航: uid=%d step=%d", agent.uid, state.step)
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
