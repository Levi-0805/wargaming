from __future__ import annotations

import math

from cssim.protocol import Action


class VehicleFireControl:
    """Patrol normally; aim and request bounded fire at a local, in-range unit.

    UE exposes normal fire by target UID, not by direction. Directional Guard
    is the supported aiming request preceding that fire. ``fireRange=1000`` is
    the scenario's known generic placeholder (including unarmed commanders).
    Its former use as a 10 m gate suppressed almost all ground fire; see the
    20261008 lower-road audit. Vehicle-attributed kills in the Oct17 replay
    occurred with preceding separations up to 134 m. For the placeholder use
    a 150 m trial envelope with bounded bursts, not a claim about the engine's
    physical maximum range or a guaranteed line of sight.
    Other valid reported ranges are respected; missing/invalid ranges deny
    explicit fire. Perception range never substitutes for a weapon range.
    """

    VEHICLES = {"BP_MNWS_Vehicle_Armored_C", "BP_MNWS_Vehicle_6x6UGV_C"}
    AIR = {"BP_Base_UAV_C", "BP_Helicopter_C"}
    MAX_UNCONFIRMED_SHOTS = 2
    RETRY_STEPS = 8
    LEGACY_GENERIC_RANGE_CM = 1000.0
    ENGAGEMENT_POLICY_CM = 15000.0
    ADVANCE_SHOTS = 4
    ADVANCE_STEPS = 16
    ADVANCE_DISTANCE_CM = 2000.0

    def __init__(self):
        self._frame = None
        self._results = {}
        self._aimed = {}
        self._shots = {}
        self._progress = {}

    @classmethod
    def fire_range(cls, agent):
        try:
            value = float(agent.properties.get("fireRange", 0))
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(value) or value <= 0:
            return 0.0
        return cls.ENGAGEMENT_POLICY_CM if value == cls.LEGACY_GENERIC_RANGE_CM else value

    @staticmethod
    def ammo(agent):
        values = agent.raw.get("interaction", ()) or ()
        try:
            counts = [float(value) for value in values]
        except (TypeError, ValueError):
            return None
        return sum(counts) if counts and all(math.isfinite(n) and n >= 0 for n in counts) else None

    @staticmethod
    def distance(agent, target):
        return math.sqrt(sum((float(agent.position[i]) - float(target.position[i])) ** 2
                             for i in range(3)))

    @staticmethod
    def guard_toward(agent, point):
        dx, dy = float(point[0] - agent.position[0]), float(point[1] - agent.position[1])
        if not math.isfinite(dx) or not math.isfinite(dy) or math.hypot(dx, dy) < 1:
            return Action.guard_position(agent.position)
        directions = ("+X", "+X+Y", "+Y", "-X+Y", "-X", "-X-Y", "-Y", "+X-Y")
        return Action.guard(directions[round(math.atan2(dy, dx) / (math.pi / 4)) % 8])

    @staticmethod
    def patrol(action):
        """Convert waypoint travel to Guard patrol, preserving recovery moves.

        Apply after MotionRecovery, which tracks ordinary movement requests.
        A direction-only Moving is a short physical escape, not an aiming
        request. Rewriting it as directional Guard cancels that movement.
        Waypoint Guard travel still requires confirmation in the live engine.
        """
        if action.command != "Moving":
            return action
        if action.points is not None:
            return Action.guard_at(action.points)
        return action

    def eligible(self, state, agent):
        """Only this vehicle's current perception can authorize explicit fire."""
        limit = self.fire_range(agent)
        if (agent.entity_type not in self.VEHICLES or not agent.alive or agent.hp <= 0
                or agent.communication_ok() is False or limit <= 0 or self.ammo(agent) == 0):
            return ()
        local = agent.perceived_uids()
        return tuple(target for target in state.perceived_opponents(agent)
                     if target.uid in local and target.team != agent.team
                     and target.alive and target.hp > 0 and math.isfinite(float(target.hp))
                     and target.entity_type not in self.AIR
                     and self.distance(agent, target) <= limit)

    def _begin(self, state):
        frame = (state.episode, state.step)
        if frame == self._frame:
            return
        if self._frame is None or frame[0] != self._frame[0] or frame[1] < self._frame[1]:
            self._aimed.clear()
            self._shots.clear()
            self._progress.clear()
        self._frame = frame
        self._results.clear()
        self._shots = {key: value for key, value in self._shots.items()
                       if state.step - value["step"] <= 64}

    def advancing(self, state, agent):
        watch = self._progress.get(agent.uid)
        return watch is not None and watch.get("until", -1) > state.step

    def _advance_after_failed_fire(self, state, agent):
        """Bound failures across targets, not just each individual target.

        Enabled by the caller only inside contested B. Navigation owns the
        actual waypoint; stop suppressing fire after 20 m or sixteen steps.
        This does not assert line of sight or authorize fire at shared targets.
        """
        position = tuple(float(v) for v in agent.position[:2])
        watch = self._progress.setdefault(agent.uid, {"failed": 0, "until": -1, "origin": position})
        if watch["until"] < 0 and math.dist(position, watch["origin"]) >= self.ADVANCE_DISTANCE_CM:
            watch.update(failed=0, origin=position)
        previous = watch.pop("shot", None)
        if previous is not None:
            step, uid, old_ammo, old_hp = previous
            target, ammo = state.entity_by_uid(uid), self.ammo(agent)
            effective = (state.step == step + 1 and target is not None
                         and uid in agent.perceived_uids() and old_ammo is not None
                         and ammo is not None and ammo < old_ammo and float(target.hp) < old_hp)
            watch["failed"] = 0 if effective else watch["failed"] + 1
        if watch["until"] >= 0:
            moved = math.dist(position, watch["origin"])
            if state.step < watch["until"] and moved < self.ADVANCE_DISTANCE_CM:
                return True
            watch.update(failed=0, until=-1, origin=position)
        if watch["failed"] >= self.ADVANCE_SHOTS:
            watch.update(until=state.step + self.ADVANCE_STEPS,
                         origin=position)
            self._aimed.pop(agent.uid, None)
            return True
        return False

    def choose(self, state, agent, *, advance_on_stall=False):
        """Return aim/fire/stop-fire action, or None to continue the route.

        A failed burst is followed by one Guard request, then route movement can
        resume. Cooling targets do not pin the vehicle in place indefinitely.
        """
        self._begin(state)
        if agent.uid in self._results:
            return self._results[agent.uid]
        advance_on_stall = (advance_on_stall and agent.alive and agent.hp > 0
                            and agent.communication_ok() is not False and self.ammo(agent) != 0)
        if not advance_on_stall:
            self._progress.pop(agent.uid, None)
        moving = advance_on_stall and self._advance_after_failed_fire(state, agent)
        action = None if moving else self._choose(state, agent)
        if advance_on_stall and action is not None and action.command == "NormalAttacking":
            target = state.entity_by_uid(action.target_uid)
            self._progress[agent.uid]["shot"] = (
                state.step, target.uid, self.ammo(agent), float(target.hp))
        self._results[agent.uid] = action
        return action

    def _choose(self, state, agent):
        candidates = self.eligible(state, agent)
        if not candidates:
            self._aimed.pop(agent.uid, None)
            return None
        available, stopped = [], []
        ammo = self.ammo(agent)
        for target in candidates:
            watch = self._shots.get((agent.uid, target.uid))
            if watch is not None:
                # Both spent ammunition and observed damage are required;
                # expenditure alone cannot refresh an ineffective burst.
                if (watch["step"] == state.step - 1 and watch["ammo"] is not None
                        and ammo is not None and ammo < watch["ammo"]
                        and float(target.hp) < watch["hp"]):
                    watch["count"] = 0
                if watch["retry"] >= 0 and state.step >= watch["retry"]:
                    watch["count"], watch["retry"] = 0, -1
                if watch["count"] >= self.MAX_UNCONFIRMED_SHOTS and watch["retry"] < 0:
                    watch["retry"] = state.step + self.RETRY_STEPS
                    stopped.append(target)
                if watch["retry"] > state.step:
                    continue
            available.append(target)
        if not available:
            self._aimed.pop(agent.uid, None)
            return self.guard_toward(agent, stopped[0].position) if stopped else None
        target = min(available, key=lambda item: (
            item.entity_type != 'BP_MNWS_Vehicle_6x6UGV_C',
            item.entity_type not in self.VEHICLES, float(item.hp), self.distance(agent, item), item.uid))
        aim = self.guard_toward(agent, target.position)
        signature = (target.uid, aim.direction)
        previous = self._aimed.get(agent.uid)
        self._aimed[agent.uid] = (signature, state.step)
        if previous is None or previous[0] != signature or state.step - previous[1] > 1:
            return aim
        key = (agent.uid, target.uid)
        watch = self._shots.setdefault(key, {"count": 0, "retry": -1})
        watch.update(count=watch["count"] + 1, hp=float(target.hp), ammo=ammo, step=state.step)
        return Action.attack(target)
