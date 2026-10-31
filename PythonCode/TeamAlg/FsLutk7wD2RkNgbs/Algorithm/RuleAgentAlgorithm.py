from __future__ import annotations

import logging
import math

from cssim.algorithms import RuleAlgorithm
from cssim.protocol import Action

from .AssaultRoute import AssaultRoute
from .CommandLinks import CommandLinks
from .MotionRecovery import MotionRecovery
from .UAVControl import UAVControl
from .UrbanContact import UrbanContact
from .HQEntry import HQEntry
from .CombatReadiness import CombatReadiness
from .ApproachCohesion import ApproachCohesion
from .AssaultFormation import AssaultFormation
from .VehicleFireControl import VehicleFireControl
from .ColumnAdvance import ColumnAdvance
from .EntryWaves import EntryWaves
from .RearCover import RearCover
from .HQExit import HQExit
from .DefenseDoctrine import DefenseDoctrine, KINDS as DEFENSE_KINDS

logger = logging.getLogger(__name__)


class ReinforceAgentAlgorithm(RuleAlgorithm):
    """Use the lower approach and choose objectives from public reconnaissance.

    A bounded opening rendezvous reduces staggered arrivals. Ground engagement
    uses local contacts; shared reports alone cannot interrupt objective travel.
    Retired interventions can be enabled separately for controlled comparisons.
    """

    _SOLDIER = "BP_BaseSoldier_C"
    _DOG = "BP_RoboDog_C"
    _UAV = "BP_Base_UAV_C"
    _HELI = "BP_Helicopter_C"
    _PRIORITY = {_SOLDIER: 0, _DOG: 1, "BP_MNWS_Vehicle_Armored_C": 2,
                 "BP_MNWS_Vehicle_6x6UGV_C": 2, _UAV: 3}
    SHARED_ENGAGEMENT_CM = 15000.0
    FIRE_REPOSITION_STEPS = 3
    NO_DAMAGE_STEPS = 8
    VEHICLE_NO_DAMAGE_STEPS = 2
    VEHICLES = {"BP_MNWS_Vehicle_Armored_C", "BP_MNWS_Vehicle_6x6UGV_C"}

    def __init__(self, context, enable_parent_assignment: bool = True, *,
                 enable_preservation_pause: bool = False,
                 enable_shared_contact_maneuver: bool = False,
                 enable_forced_entry_under_contact: bool = False,
                 enable_continuous_column: bool = False,
                 enable_entry_waves: bool = False):
        super().__init__(context)
        self.enable_parent_assignment = bool(enable_parent_assignment)
        self.enable_preservation_pause = enable_preservation_pause
        self.enable_shared_contact_maneuver = enable_shared_contact_maneuver
        self.enable_forced_entry_under_contact = enable_forced_entry_under_contact
        self.enable_continuous_column = enable_continuous_column
        self.enable_entry_waves = enable_entry_waves
        logger.info("Ground rule comparison switches: preservation_pause=%s shared_contact_maneuver=%s forced_entry_under_contact=%s",
                    enable_preservation_pause, enable_shared_contact_maneuver, enable_forced_entry_under_contact)
        logger.info("Formation comparison switches: continuous_column=%s entry_waves=%s",
                    enable_continuous_column, enable_entry_waves)
        self._command_links = CommandLinks()
        self._motion_recovery = MotionRecovery()
        self._route = AssaultRoute(adaptive=True)
        self._uav = UAVControl()
        self._urban = UrbanContact()
        self._hq_entry = HQEntry()
        self._readiness = CombatReadiness()
        self._cohesion = ApproachCohesion()
        self._formation = AssaultFormation()
        self._vehicle_fire = VehicleFireControl()
        self._column = ColumnAdvance()
        self._entry_waves = EntryWaves()
        self._rear_cover = RearCover()
        self._hq_exit = HQExit()
        self._doctrine = DefenseDoctrine()
        self._opening_column_start = None
        self._frame = None
        self._support = {}
        self._fire_watch = {}
        self._hit_watch = {}

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
            self._hit_watch.clear()
            self._opening_column_start = None
        self._frame = frame
        self._urban.begin(state)
        self._route.update(state)
        self._doctrine.observe(state, self._route)
        self._doctrine.apply(self._route, self._formation)
        self._hq_exit.begin(state, self._route)
        self._rear_cover.begin(state, self._route)
        self._hq_entry.begin(state, self._route)
        self._uav.prepare_recon(state, self._route)
        objective = self._route.objective(state)
        surveyed = self._uav.recon_status(state, objective)['ready'] if objective is not None else False
        self._formation.begin(state, self._route, recon_ready=surveyed)
        self._column.begin(state, self._route)
        if self.enable_entry_waves:
            self._entry_waves.begin(state, self._route, self._column)
        self._cohesion.begin(state, self._route)
        if self.enable_preservation_pause:
            self._readiness.begin(state, self._route)
        self._support = {}
        for unit in state.agents:
            if unit.alive:
                for enemy in state.perceived_opponents(unit):
                    if enemy.alive:
                        self._support[enemy.uid] = self._support.get(enemy.uid, 0) + 1

    def _attack_target(self, agent, enemies, local=None):
        return min(enemies, key=lambda enemy: (
            local is not None and enemy.uid not in local,
            agent.entity_type in self.VEHICLES and enemy.entity_type not in self.VEHICLES,
            self._PRIORITY.get(enemy.entity_type, 4),
            -self._support.get(enemy.uid, 0),
            max(float(enemy.hp), 0),
            self._horizontal(agent.position, enemy.position), enemy.uid,
        ))

    @staticmethod
    def _has_ammo(agent):
        values = agent.raw.get("interaction", ()) or ()
        try:
            return not values or any(float(value) > 0 for value in values)
        except (TypeError, ValueError):
            return True  # Missing/unknown metadata is not proof of an empty weapon.

    def _local_vehicle_threat(self, state, agent, target):
        # A local vehicle can threaten a withdrawing unit well beyond the old
        # 25 m departure filter. A shared report alone cannot enable this rule.
        return (target.alive and target.hp > 0 and target.team != agent.team
                and target.entity_type in self.VEHICLES
                and target.uid in agent.perceived_uids()
                and self._horizontal(agent.position, target.position) <= VehicleFireControl.ENGAGEMENT_POLICY_CM)

    def _engage(self, state, agent, contacts, local, *, priority_targets=()):
        local_contacts = [enemy for enemy in contacts if enemy.uid in local]
        candidates = local_contacts or contacts
        available = [enemy for enemy in candidates if
                     self._hit_watch.get((agent.uid, enemy.uid), {}).get("retry", -1) <= state.step]
        objective = self._route.objective(state)
        if (not available and candidates and objective is not None
                and not self._route.is_hq(objective) and agent.uid not in self._route._garrison):
            # All failed targets cooling down used to fall back to the same
            # candidates, defeating the cooldown and pinning units to a wall.
            # Preserve HQ/guard self-defence; mobile secondary assaults resume
            # navigation/clearance without assuming elevation proves occlusion.
            if self._route.inside_garrison(objective, agent.position):
                return self._urban.search_point(state, agent, objective, self._route)
            point = self._route.point(state, agent)
            if point is not None:
                return Action.move_at(point)
        pool = available or candidates
        priority = {enemy.uid for enemy in priority_targets}
        preferred = [enemy for enemy in pool if enemy.uid in priority]
        target = self._attack_target(agent, preferred or pool, local)
        if target.uid not in local and agent.entity_type in self._urban.GROUND:
            target = self._urban.planned_target(state, agent, contacts) or target
            if not self._urban.shared_fire_allowed(state, agent, target):
                return self._urban.reposition(state, agent, target)
        return self._fire_or_close(state, agent, target)

    def _fire_or_close(self, state, agent, target):
        # Local detection is not proof of an effective weapon line of sight.
        # Bound repeated shots without ammunition use or target damage.
        ammo = tuple(agent.raw.get("interaction", ()) or ())
        key = (agent.uid, target.uid)
        hit = self._hit_watch.get(key)
        if hit is None or (hit.get("retry", -1) <= state.step and (
                state.step - hit["seen"] > 2 or hit.get("retry", -1) >= 0)):
            hit = {"hp": float(target.hp), "start": state.step, "seen": state.step, "retry": -1}
        if float(target.hp) < hit["hp"] - .01:
            hit.update(hp=float(target.hp), start=state.step, retry=-1)
        hit["seen"] = state.step
        self._hit_watch[key] = hit
        budget = self.VEHICLE_NO_DAMAGE_STEPS if agent.entity_type in self.VEHICLES else self.NO_DAMAGE_STEPS
        if state.step - hit["start"] >= budget and hit["retry"] < 0:
            hit["retry"] = state.step + 8
            logger.info("持续攻击无伤害，换位或换目标: uid=%d target=%d step=%d", agent.uid, target.uid, state.step)
        ineffective = hit["retry"] > state.step
        if self._urban.effective(state, agent, target):
            hit.update(start=state.step, retry=-1)
            ineffective = False
        signature = (target.uid, ammo)
        old = self._fire_watch.get(agent.uid)
        start = old[1] if old and old[0] == signature else state.step
        self._fire_watch[agent.uid] = (signature, start)
        if ineffective or self.FIRE_REPOSITION_STEPS <= state.step - start < self.FIRE_REPOSITION_STEPS + 2:
            # Seek a firing angle using a fixed, bounded movement attempt.
            return self._urban.reposition(state, agent, target, duration=6 if ineffective else 2)
        if state.step - start >= self.FIRE_REPOSITION_STEPS + 2:
            self._fire_watch[agent.uid] = (signature, state.step)
        return Action.attack(target)

    def _defend(self, state, agent, obj, contacts, local):
        point = self._route.guard_slot(agent, obj)
        armor = [enemy for enemy in contacts if self._local_vehicle_threat(state, agent, enemy)]
        if armor and self._has_ammo(agent):
            # Judge self-defence from the defender, not the HQ centre. Even a
            # guard just outside the box must be able to respond before moving.
            action = self._engage(state, agent, contacts, local, priority_targets=armor)
            return self._urban.constrain(agent, action, obj, self._route)
        if not self._route.inside_garrison(obj, agent.position):
            return Action.move_at(point)
        threats = [enemy for enemy in contacts if self._horizontal(enemy.position, obj.position) <= 12000]
        if threats and self._has_ammo(agent):
            action = self._engage(state, agent, threats, local)
            return self._urban.constrain(agent, action, obj, self._route)
        reaction = self._rear_cover.guard_reaction(state, agent, obj, self._route)
        if reaction is not None:
            return reaction
        if self._route.enemy_count(state, obj) > 0:
            action = self._urban.search_point(state, agent, obj, self._route)
            return self._urban.constrain(agent, action, obj, self._route)
        # The blind-hit response may have moved this unit's reached hold slot.
        point = self._route.guard_slot(agent, obj)
        return Action.move_at(point) if self._horizontal(agent.position, point) > 450 else self._rear_cover.scan(state, agent, obj)

    def _combat_action(self, state, agent):
        if not agent.alive:
            return Action.idle()
        if agent.entity_type == self._UAV:
            return self._uav.choose(state, agent, self._route.objective(state), self._route)
        remnant = self._route._remnant.point(state, agent, self._route)
        if remnant is not None:
            # A withdrawal must override the old B firing/search/assembly
            # priorities. Otherwise a visible defender keeps the survivor at
            # B even though the objective selector has already chosen A.
            return (Action.guard_position(remnant) if self._horizontal(agent.position, remnant) <= 450
                    else Action.move_at(remnant))
        if agent.entity_type in self.VEHICLES:
            objective = self._route.objective(state)
            advancing_b = (self._route.hq_secured and self._route.is_base_b(objective)
                           and self._route.enemy_count(state, objective) > 0
                           and self._route.inside_garrison(objective, agent.position))
            fire = self._vehicle_fire.choose(state, agent, advance_on_stall=advancing_b)
            if fire is not None:
                return fire
            if advancing_b and self._vehicle_fire.advancing(state, agent):
                # A dense group must not keep a car cycling failed bursts.
                # Let the fixed search goal reach navigation and its watchdog.
                return self._urban.search_point(state, agent, objective, self._route)
            cover = self._rear_cover.choose_vehicle(state, agent, self._route)
            if cover is not None:
                return cover
        formation = self._formation.choose(state, agent, self._route)
        if formation is not None:
            return formation
        # Vehicles form the first wave; the opening rendezvous applies to the
        # capture wave only. Rear units continue on their existing road.
        assembly = self._cohesion.choose(state, agent, self._route) if agent.entity_type not in self.VEHICLES else None
        if assembly is not None:
            return assembly
        local = {enemy.uid for enemy in state.perceived_opponents(agent)}
        contacts = [enemy for enemy in state.visible_opponents if enemy.alive and (
            enemy.uid in local or (self.enable_shared_contact_maneuver
                                  and self._horizontal(agent.position, enemy.position) <= self.SHARED_ENGAGEMENT_CM))]
        if agent.entity_type in self.VEHICLES:
            # VehicleFireControl exclusively owns explicit vehicle fire. A
            # visible out-of-range unit must not fall through to legacy attack.
            contacts = []
        if self.enable_preservation_pause:
            preserve = self._readiness.choose(state, agent, self._route, self._urban, contacts)
            if preserve is not None:
                return preserve
        entry = self._hq_entry.choose(state, agent, self._route, self._urban, contacts,
                                     allow_contact_override=self.enable_forced_entry_under_contact)
        if entry is not None:
            return entry
        if self._hq_entry.pending(state, agent, self._route):
            # A provisional garrison label must not cut across the facade,
            # including after the bounded entrance retry budget is exhausted.
            if contacts and self._has_ammo(agent):
                return self._engage(state, agent, contacts, local)
            if agent.uid in self._hq_entry._exhausted:
                return Action.move_at(self._route.point(state, agent, ignore_garrison=True))
        garrison = self._route.garrison_for(state, agent)
        if garrison is not None:
            return self._defend(state, agent, garrison, contacts, local)
        priority = ()
        if self._route.departing(state, agent):
            # Keep ignoring unrelated distant stragglers, but never discard a
            # locally detected vehicle solely because it is behind the column.
            objective = self._route.objective(state)
            priority = [enemy for enemy in contacts if self._local_vehicle_threat(state, agent, enemy)]
            priority_ids = {enemy.uid for enemy in priority}
            contacts = [enemy for enemy in contacts if
                        self._horizontal(agent.position, enemy.position) <= 2500
                        or enemy.uid in priority_ids
                        or (objective is not None and self._horizontal(enemy.position, objective.position) <= 18000)]
        if contacts and self._has_ammo(agent):
            return self._engage(state, agent, contacts, local, priority_targets=priority)
        self._fire_watch.pop(agent.uid, None)
        point = self._route.point(state, agent)
        exit_action = self._hq_exit.choose(state, agent, self._route, point)
        if exit_action is not None:
            self._urban.cancel(agent)
            return exit_action
        objective = self._route.objective(state)
        visible = [enemy for enemy in state.visible_opponents if enemy.alive]
        if (objective is not None
                and agent.entity_type in self._urban.GROUND
                and self._route.inside_garrison(objective, agent.position)
                and self._route.enemy_count(state, objective) > 0):
            # Every contested objective needs active clearance, including
            # non-garrison units at B/A. Public counts never authorize fire
            # at an unseen enemy.
            return self._urban.search_point(state, agent, objective, self._route)
        # Retired shortcut: a reported enemy behind a wall used to replace the
        # navigable objective route even when this unit could not see it.
        if agent.entity_type not in self.VEHICLES and self.enable_shared_contact_maneuver and self._has_ammo(agent) and point is not None and visible and (objective is None or (
            self._horizontal(agent.position, objective.position) <= 18000
            and self._horizontal(agent.position, point) <= 2000
        )):
            nearby = [enemy for enemy in visible if
                      (agent.entity_type not in self.VEHICLES or enemy.entity_type not in {self._UAV, self._HELI})
                      and (objective is None or self._horizontal(enemy.position, objective.position) <= 18000)]
            if nearby:
                point = self._xyz(self._attack_target(agent, nearby).position)
        if point is None:
            return Action.guard_position(agent.position)
        # Local contacts already interrupt navigation above. A periodic blind
        # Guard scan here reset MotionRecovery every four movement frames,
        # preventing its eight-frame obstruction check from ever firing.
        # Spread the final approach by at most eight metres, keeping the same
        # lower-road route and capture objective. Slots do not change on deaths.
        if (self._route.known_map and agent.entity_type in {self._SOLDIER, self._DOG}
                and objective is not None and self._horizontal(point, objective.position) < 15000
                and self._horizontal(agent.position, objective.position) < 25000
                and self._route.garrison_for(state, agent) is None):
            dx, dy = objective.position[:2] - agent.position[:2]
            distance = max(math.hypot(dx, dy), 1)
            offset = (agent.team_index % 5 - 2) * 400
            point = (point[0] - dy / distance * offset, point[1] + dx / distance * offset, point[2])
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
        result = self._motion_recovery.apply(state, agent, action)
        if self.enable_entry_waves:
            result = self._entry_waves.apply(state, agent, result, self._route, self._column)
        # Keep the opening car/foot rendezvous, then let the assembled body
        # advance. Repeated whole-route holds and a second entrance assembly
        # are retired by default after the two failed twenty-round batches.
        # Bound even an early spacing stop outside the rendezvous radius:
        # ApproachCohesion's deadline has not started at that point yet.
        opening_column = (not self._cohesion.released and (
            self._opening_column_start is None
            or state.step - self._opening_column_start < self._cohesion.MAX_WAIT))
        if (self.enable_continuous_column or opening_column) and not self._route._remnant.active:
            result = self._column.apply(state, agent, result, self._route, self._formation,
                                        self._entry_waves.released if self._entry_waves.waves else ())
            if self._column.held and self._opening_column_start is None:
                self._opening_column_start = state.step
        if agent.uid in self._column.held or agent.uid in self._entry_waves.held:
            # A deliberate formation wait is not a failed movement attempt.
            # Otherwise recovery would manufacture escape moves and skip road
            # corners while the unit was explicitly ordered to stay here.
            self._motion_recovery.pause(state, agent)
            self._route._waypoint_watch.pop(agent.uid, None)
            self._hq_entry._leg_watch.pop(agent.uid, None)
            self._hq_entry._watch.pop(agent.uid, None)
        garrison = self._route.garrison_for(state, agent)
        if (garrison is not None and result.command == "Moving"
                and self._route.inside_garrison(garrison, agent.position)
                and not self._hq_entry.pending(state, agent, self._route)):
            result = self._urban.constrain(agent, result, garrison, self._route)
        self._urban.record(state, agent, result)
        self._formation.record(state, agent, result, self._route)
        # Apply existing navigation/recovery before serializing the supported
        # Guard patrol command. A disconnected unit still gets link recovery.
        if agent.entity_type in self.VEHICLES and agent.communication_ok() is not False and result.command == 'Moving':
            result = VehicleFireControl.patrol(result)
        return result

    def decide(self, state):
        self._prepare(state)
        return [self.choose_action(state, agent) for agent in state.agents]

    def policy_metadata(self):
        return {**super().policy_metadata(), **self._doctrine.metadata()}

    def training_metrics(self):
        return {**super().training_metrics(),
                'defense_kind_code': DEFENSE_KINDS.index(self._doctrine.kind)}


RuleAgentAlgorithm = ReinforceAgentAlgorithm
