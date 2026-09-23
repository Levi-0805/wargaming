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


def test_vanguard_waits_until_the_column_closes_up():
    lead = entity(1, 0, 0, 15000, 0)
    rear = [entity(10 + index, 0, index + 1, 0, 0) for index in range(8)]
    held = objective(2, 20000, 0, blue=30)
    agents = [lead, *rear]
    actions = algorithm(agents).act(state(agents, objectives=[held]))
    assert actions[0].command == "Guard"
    assert actions[1].command == "Moving"


def test_dead_agent_stays_idle():
    dead = entity(10, 0, 0, 0)
    dead.alive = False
    dead.hp = 0
    action = algorithm([dead]).act(state([dead]))[0]
    assert action.command == "Idle"
