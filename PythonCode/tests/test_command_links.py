import asyncio
from dataclasses import replace

import numpy as np
import pytest

from cssim.algorithms import AlgorithmContext
from cssim.environment.models import Entity, TeamState
from cssim.protocol import Action, ActionSetFactory, ChangeParentAction
from cssim.runtime.special_commands import SpecialCommandExecutor
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.CommandLinks import CommandLinks
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RuleAgentAlgorithm import ReinforceAgentAlgorithm
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RLAgentAlgorithm import (
    AlgorithmConfig, ReinforceAgentAlgorithm as EntryPoint,
)


def unit(uid, x=0, *, kind="BP_BaseSoldier_C", parent=None, comm=1, team=0, alive=True, z=0):
    result = Entity.from_property({
        "uId": uid, "agentTeam": team, "type": kind, "agentHp": 100,
        "initLocation": {"x": x, "y": 0, "z": z},
        "indexInTeam": parent, "rSVD1": f"unit;{comm}",
    }, uid)
    result.alive = alive
    return result


def view(*agents, step=0, episode=1, commanders=()):
    return TeamState(0, np.ones((len(agents), 1), dtype=bool), tuple(agents), (), {},
                     episode, step, False, commanders=tuple(commanders))


def test_dead_parent_is_replaced_and_repair_occupies_only_the_child_row():
    dead = unit(1, alive=False)
    survivor = unit(2, 500)
    dog = unit(3, 1000, kind="BP_RoboDog_C", parent=1, comm=0)
    state = view(dead, survivor, dog)
    context = AlgorithmContext(0, state.agents, (), ActionSetFactory.standard(), "cpu", {})
    rule = ReinforceAgentAlgorithm(context)
    actions = rule.act(state)
    commands = rule.take_selected_special_commands()
    assert actions[2].command == "Idle"
    assert len(commands) == 1
    assert commands[0].parameters() == {"parent_uid": 2, "child_uid": 3}
    assert rule.take_selected_special_commands() == ()
    dog.raw["indexInTeam"], dog.raw["rSVD1"] = 2, "unit;1"
    assert not isinstance(rule.choose_action(replace(state, step=1), dog), ChangeParentAction)


@pytest.mark.parametrize("kind", ["BP_RoboDog_C", "BP_Base_UAV_C", "BP_MNWS_Vehicle_Armored_C",
                                 "BP_MNWS_Vehicle_6x6UGV_C", "BP_Helicopter_C"])
def test_obstruction_outage_inside_one_km_triggers_handoff(kind):
    old = unit(1)
    nearby = unit(2, 1800)
    child = unit(3, 2000, kind=kind, parent=1, comm=0)
    command = CommandLinks().reassign(view(old, nearby, child), child)
    assert command.parent_uid == 2


def test_handoff_happens_before_one_km_and_uses_3d_distance():
    commander = unit(1, kind="BaseCommander_C", parent=1)
    nearby = unit(2, 65_000)
    dog = unit(3, 70_000, kind="BP_RoboDog_C", parent=1)
    command = CommandLinks().reassign(view(nearby, dog, commanders=[commander]), dog)
    assert command.parent_uid == 2
    high = unit(4, 70_000, z=110_000)
    assert not CommandLinks().reassign(view(high, dog, commanders=[commander]), dog)


def test_healthy_link_does_not_switch_for_a_slightly_closer_soldier():
    current = unit(1)
    nearby = unit(2, 900)
    dog = unit(3, 1000, kind="BP_RoboDog_C", parent=1)
    assert CommandLinks().reassign(view(current, nearby, dog), dog) is None


def test_candidates_exclude_dead_enemy_disconnected_and_cyclic_parents():
    dead = unit(1, alive=False)
    enemy = unit(2, team=1)
    offline = unit(3, comm=0)
    cycle = unit(4, parent=6)
    good = unit(5, 1000)
    dog = unit(6, kind="BP_RoboDog_C", parent=1, comm=0)
    command = CommandLinks().reassign(view(dead, enemy, offline, cycle, good, dog), dog)
    assert command.parent_uid == 5


def test_commander_fallback_and_no_out_of_range_repair():
    commander = unit(1, kind="BaseCommander_C", parent=1)
    dog = unit(2, 5000, kind="BP_RoboDog_C", parent=99, comm=0)
    state = view(dog, commanders=[commander])
    assert CommandLinks().reassign(state, dog).parent_uid == 1
    dog.position[0] = 101_000
    assert CommandLinks().reassign(state, dog) is None


def test_failed_rpc_does_not_idle_the_unit_every_frame_and_tries_another_node():
    one, two = unit(1, 100), unit(2, 500)
    dog = unit(3, kind="BP_RoboDog_C", parent=99, comm=0)
    links = CommandLinks()
    assert links.reassign(view(one, two, dog), dog).parent_uid == 1
    assert links.reassign(view(one, two, dog, step=1), dog) is None
    assert links.reassign(view(one, two, dog, step=4), dog).parent_uid == 2


def test_pending_parent_dies_before_rpc_confirmation():
    one, two = unit(1, 100), unit(2, 500)
    dog = unit(3, kind="BP_RoboDog_C", parent=99, comm=0)
    links = CommandLinks()
    links.reassign(view(one, two, dog), dog)
    one.alive = False
    assert links.reassign(view(one, two, dog, step=1), dog).parent_uid == 2


def test_new_episode_clears_pending_attempts():
    soldier = unit(1)
    dog = unit(2, 100, kind="BP_RoboDog_C", parent=99, comm=0)
    links = CommandLinks()
    links.reassign(view(soldier, dog), dog)
    assert links.reassign(view(soldier, dog, episode=2), dog).parent_uid == 1


def test_radio_travel_limit_only_applies_to_unmanned_units():
    parent = unit(1)
    uav = unit(2, 1000, kind="BP_Base_UAV_C", parent=1)
    state = view(parent, uav)
    move = Action.move_at((120_000, 0, 1200))
    constrained = CommandLinks().constrain(state, uav, move)
    assert np.linalg.norm(np.array(constrained.points) - parent.position) <= 40_001
    assert CommandLinks().constrain(state, parent, move) is move


def test_link_repair_precedes_bombing():
    soldier = unit(1)
    uav = unit(2, 1000, kind="BP_Base_UAV_C", parent=99, comm=0)
    state = view(soldier, uav)
    context = AlgorithmContext(0, state.agents, (), ActionSetFactory.standard(), "cpu", {})
    assert isinstance(ReinforceAgentAlgorithm(context).choose_action(state, uav), ChangeParentAction)


def test_factory_enables_link_management_by_default():
    state = view(unit(1))
    context = AlgorithmContext(0, state.agents, (), ActionSetFactory.standard(), "cpu", {})
    assert AlgorithmConfig().enable_parent_assignment
    assert EntryPoint(context).enable_parent_assignment


def test_disconnected_dogs_cannot_set_the_columns_pace():
    soldiers = [unit(index, 10_000) for index in range(1, 7)]
    dogs = [unit(index, -50_000, kind="BP_RoboDog_C", comm=0) for index in range(10, 23)]
    state = view(*soldiers, *dogs)
    context = AlgorithmContext(0, state.agents, (), ActionSetFactory.standard(), "cpu", {})
    assert ReinforceAgentAlgorithm(context)._pace_troops(state) == soldiers


def test_repair_is_sent_through_the_real_special_command_executor():
    from cssim.environment.models import SimulationState

    parent = unit(1)
    child = unit(2, kind="BP_RoboDog_C", parent=99, comm=0)
    state = view(parent, child)
    command = CommandLinks().reassign(state, child)
    calls = []

    class Client:
        async def change_actor_parent(self, parent_uid, child_uid):
            calls.append((parent_uid, child_uid))
            return {"success": True}

    snapshot = SimulationState(state.agents, {}, 1, 0, False)
    results = asyncio.run(SpecialCommandExecutor(Client()).execute(0, snapshot, [command]))
    assert calls == [(1, 2)]
    assert results[0].success
