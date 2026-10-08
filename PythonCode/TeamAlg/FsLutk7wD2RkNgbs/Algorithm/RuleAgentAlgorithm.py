from __future__ import annotations

from cssim.algorithms import RuleAlgorithm
from cssim.protocol import Action

from .AssaultRoute import AssaultRoute
from .CommandLinks import CommandLinks
from .MotionRecovery import MotionRecovery
from .UAVControl import UAVControl


class ReinforceAgentAlgorithm(RuleAlgorithm):
    """All units use the lower approach, fight contacts, then capture HQ, B, A.

    No median-distance waiting, casualty-based regrouping or three-lane
    offsets. Local perception authorizes the documented attack API; the
    generic fireRange=1000 metadata is not an extra 9.5m firing gate.
    """

    _SOLDIER = "BP_BaseSoldier_C"
    _DOG = "BP_RoboDog_C"
    _UAV = "BP_Base_UAV_C"
    _HELI = "BP_Helicopter_C"
    _PRIORITY = {_SOLDIER: 0, _DOG: 1, "BP_MNWS_Vehicle_Armored_C": 2, _UAV: 3}

    def __init__(self, context, enable_parent_assignment: bool = True):
        super().__init__(context)
        self.enable_parent_assignment = bool(enable_parent_assignment)
        self._command_links = CommandLinks()
        self._motion_recovery = MotionRecovery()
        self._route = AssaultRoute()
        self._uav = UAVControl()
        self._frame = None
        self._support = {}
        self._fire_watch = {}

    @staticmethod
    def _xyz(point):
        return tuple(float(v) for v in point)

    _horizontal = staticmethod(AssaultRoute.distance)

    def _prepare(self, state):
        frame = (state.episode, state.step)
        if self._frame == frame:
            return
        if self._frame is None or self._frame[0] != state.episode or state.step < self._frame[1]:
            self._fire_watch.clear()
        self._frame = frame
        self._route.update(state)
        self._support = {}
        for unit in state.agents:
            if unit.alive:
                for enemy in state.perceived_opponents(unit):
                    if enemy.alive:
                        self._support[enemy.uid] = self._support.get(enemy.uid, 0) + 1

    def _attack_target(self, agent, enemies):
        return min(enemies, key=lambda enemy: (
            self._PRIORITY.get(enemy.entity_type, 4),
            -self._support.get(enemy.uid, 0),
            max(float(enemy.hp), 0),
            self._horizontal(agent.position, enemy.position), enemy.uid,
        ))

    def _fire_or_close(self, state, agent, target):
        # A visible contact is actionable, as in the official rule example.
        # If repeated attacks do no damage and consume no ammo, move closer
        # briefly instead of standing forever behind cover/out of range.
        ammo = tuple(agent.raw.get("interaction", ()) or ())
        signature = (target.uid, float(target.hp), ammo)
        old = self._fire_watch.get(agent.uid)
        start = old[1] if old and old[0] == signature else state.step
        self._fire_watch[agent.uid] = (signature, start)
        if 8 <= state.step - start < 11:
            return Action.move_at(target.position)
        if state.step - start >= 11:
            self._fire_watch[agent.uid] = (signature, state.step)
        return Action.attack(target)

    def _combat_action(self, state, agent):
        if not agent.alive:
            return Action.idle()
        if agent.entity_type == self._UAV:
            return self._uav.choose(state, agent, self._route.objective(state))
        contacts = [enemy for enemy in state.perceived_opponents(agent) if enemy.alive]
        if self._route.departing(state, agent):
            # Do not let distant survivors at the captured HQ pin the entire
            # column. The garrison handles them; the main force fires only at
            # close threats or contacts around its next objective.
            objective = self._route.objective(state)
            contacts = [enemy for enemy in contacts if
                        self._horizontal(agent.position, enemy.position) <= 2500
                        or (objective is not None and self._horizontal(enemy.position, objective.position) <= 18000)]
        if contacts:
            target = self._attack_target(agent, contacts)
            return self._fire_or_close(state, agent, target)
        self._fire_watch.pop(agent.uid, None)
        point = self._route.point(state, agent)
        objective = self._route.objective(state)
        visible = [enemy for enemy in state.visible_opponents if enemy.alive]
        # Shared contacts guide movement without fabricating local visibility.
        if visible and (objective is None or (
            self._horizontal(agent.position, objective.position) <= 18000
            and self._horizontal(agent.position, point) <= 2000
        )):
            nearby = [enemy for enemy in visible if objective is None or
                      self._horizontal(enemy.position, objective.position) <= 18000]
            if nearby:
                point = self._xyz(self._attack_target(agent, nearby).position)
        if point is None:
            return Action.guard_position(agent.position)
        if agent.entity_type in {self._UAV, self._HELI}:
            point = (point[0], point[1], point[2] + 1800)
        if self._horizontal(agent.position, point) <= 450:
            return Action.guard_position(point)
        return Action.move_at(point)

    def choose_action(self, state, agent):
        self._prepare(state)
        if self.enable_parent_assignment:
            repair = self._command_links.reassign(state, agent)
            if repair is not None:
                self._motion_recovery.pause(state, agent)
                return repair
        action = self._combat_action(state, agent)
        if self.enable_parent_assignment:
            action = self._command_links.constrain(state, agent, action)
        if agent.entity_type == self._UAV:
            return action  # UAVControl owns flight recovery and strike persistence.
        return self._motion_recovery.apply(state, agent, action)

    def decide(self, state):
        self._prepare(state)
        return [self.choose_action(state, agent) for agent in state.agents]


RuleAgentAlgorithm = ReinforceAgentAlgorithm
