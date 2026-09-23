from __future__ import annotations

import numpy as np

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RuleAgentAlgorithm import (
    ReinforceAgentAlgorithm as RedRule,
)
from cssim.algorithms import AlgorithmContext
from cssim.environment.models import Entity, KeyObject, TeamState
from cssim.protocol import ActionSetFactory


def entity(uid, team, index, x, y=0, *, kind="BP_BaseSoldier_C", hp=100, fire=1000):
    return Entity.from_property({
        "uId": uid,
        "agentTeam": team,
        "type": kind,
        "agentHp": hp,
        "fireRange": fire,
        "perceptionRange": 10000,
        "initLocation": {"x": x, "y": y, "z": 0},
    }, index)


def objective(uid, x, y=0, *, blue=0, red=0, percent=0):
    return KeyObject.from_raw({
        "uId": uid,
        "location": {"x": x, "y": y, "z": 0},
        "blueteamNum": blue,
        "redTeamNum": red,
        "percent": percent,
    })


def state(agents, enemies=(), objectives=()):
    return TeamState(
        team=0,
        action_mask=np.ones((len(agents), 1), dtype=bool),
        agents=tuple(agents),
        visible_opponents=tuple(enemies),
        raw={},
        episode=1,
        step=0,
        training=False,
        key_objects=tuple(objectives),
    )


def algorithm(agents):
    context = AlgorithmContext(
        0, tuple(agents), (), ActionSetFactory.standard(), "cpu", {}
    )
    return RedRule(context)


def test_prefers_soldier_inside_weapon_range():
    soldier = entity(10, 0, 0, 0)
    dog = entity(21, 1, 0, 200, kind="BP_RoboDog_C")
    enemy_soldier = entity(20, 1, 1, 500)
    action = algorithm([soldier]).act(state(
        [soldier], [dog, enemy_soldier]
    ))[0]
    assert action.command == "NormalAttacking"
    assert action.target_uid == 20


def test_closes_distance_before_firing_outside_weapon_range():
    soldier = entity(10, 0, 0, 0)
    enemy = entity(20, 1, 0, 4000)
    action = algorithm([soldier]).act(state([soldier], [enemy]))[0]
    assert action.command == "Moving"
    assert action.points[0] > 1000


def test_teams_advance_on_the_blue_strongpoint_in_separate_lanes():
    soldiers = [entity(index, 0, index, 0, 0) for index in range(6)]
    dogs = [
        entity(100 + index, 0, 20 + index, 40, 0, kind="BP_RoboDog_C")
        for index in range(6)
    ]
    empty = objective(1, 5000, -4000, blue=0)
    held_by_blue = objective(2, 20000, 0, blue=30)
    agents = soldiers + dogs
    actions = algorithm(agents).act(state(agents, objectives=[empty, held_by_blue]))
    by_uid = {agent.uid: action for agent, action in zip(agents, actions)}
    assert by_uid[0].command == "Moving"
    assert by_uid[0].points[0] > 10000
    assert by_uid[100].points[0] > 10000
    assert abs(by_uid[0].points[1] - by_uid[5].points[1]) > 4000


def test_forward_unit_keeps_advancing():
    soldier = entity(10, 0, 0, 0, 0)
    road = objective(1, 20000, 0)
    action = algorithm([soldier]).act(state([soldier], objectives=[road]))[0]
    assert action.command == "Moving"
    assert action.points[0] > 10000


def test_soldiers_march_to_the_area_a_uav_has_seen():
    soldier = entity(10, 0, 0, 0, 0)
    soldier.raw["agentPerception"] = []
    enemy = entity(20, 1, 0, 8000, 1000)
    held = objective(2, 30000, 0, blue=30)
    rule = algorithm([soldier])
    first = rule.act(state([soldier], [enemy], [held]))[0]
    assert first.command == "Moving"
    assert 6000 < first.points[0] < 12000
    second = rule.act(state([soldier], [], [held]))[0]
    assert second.command == "Moving"
    assert 6000 < second.points[0] < 12000


def test_uav_steps_toward_the_scout_point_and_holds_on_arrival():
    far = entity(1, 0, 0, 0, 0, kind="BP_Base_UAV_C")
    held = objective(2, 20000, 0, blue=10)
    far_action = algorithm([far]).act(state([far], objectives=[held]))[0]
    assert far_action.command == "Moving"
    assert 5000 < far_action.points[0] < 12000
    near = entity(1, 0, 0, 20000, -4000, kind="BP_Base_UAV_C")
    near.position[:] = (20000, -4000, 1200)
    near_action = algorithm([near]).act(state([near], objectives=[held]))[0]
    assert near_action.command == "Guard"


def test_lynx_uses_a_direction_because_point_moves_do_not_run():
    lynx = entity(7, 0, 0, 0, 0, kind="BP_MNWS_Vehicle_6x6UGV_C")
    held = objective(2, -30000, 0, blue=10)
    action = algorithm([lynx]).act(state([lynx], objectives=[held]))[0]
    assert action.command == "Moving"
    assert action.direction == "-X-Y"
    assert action.points is None


def test_armored_vehicle_keeps_its_own_altitude():
    car = entity(50, 0, 0, 0, 0, kind="BP_MNWS_Vehicle_Armored_C")
    car.position[2] = 80
    held = objective(2, 20000, 0, blue=10)
    held.position[2] = 5000
    action = algorithm([car]).act(state([car], objectives=[held]))[0]
    assert action.command == "Moving"
    assert action.points[2] == 80


def test_dead_agent_stays_idle():
    dead = entity(10, 0, 0, 0)
    dead.alive = False
    dead.hp = 0
    action = algorithm([dead]).act(state([dead]))[0]
    assert action.command == "Idle"
