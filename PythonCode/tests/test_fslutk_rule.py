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


def test_five_pairs_share_a_road_and_the_sixth_pair_takes_another():
    soldiers = [entity(index, 0, index, 0, 0) for index in range(6)]
    dogs = [
        entity(100 + index, 0, 20 + index, 40, 0, kind="BP_RoboDog_C")
        for index in range(6)
    ]
    right = objective(1, 6000, -4000)
    left = objective(2, 6000, 8000)
    agents = soldiers + dogs
    actions = algorithm(agents).act(state(agents, objectives=[left, right]))
    by_uid = {agent.uid: action for agent, action in zip(agents, actions)}
    assert by_uid[0].command == "Moving"
    assert by_uid[100].command == "Moving"
    assert by_uid[0].points[1] < 0
    assert by_uid[100].points[1] < 0
    assert by_uid[5].points[1] > 0
    assert by_uid[105].points[1] > 0


def test_forward_unit_keeps_advancing():
    soldier = entity(10, 0, 0, 0, 0)
    road = objective(1, 20000, 0)
    action = algorithm([soldier]).act(state([soldier], objectives=[road]))[0]
    assert action.command == "Moving"
    assert action.points[0] > 10000


def test_dead_agent_stays_idle():
    dead = entity(10, 0, 0, 0)
    dead.alive = False
    dead.hp = 0
    action = algorithm([dead]).act(state([dead]))[0]
    assert action.command == "Idle"
