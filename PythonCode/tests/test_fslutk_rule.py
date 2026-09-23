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


def test_ground_force_uses_the_right_hand_objective():
    soldiers = [entity(10, 0, 0, 0, 0), entity(11, 0, 1, 100, 0)]
    right = objective(1, 5000, -4000)
    left = objective(2, 5000, 8000, blue=0)
    actions = algorithm(soldiers).act(state(soldiers, objectives=[left, right]))
    assert all(action.command == "Moving" for action in actions)
    assert all(action.points[1] < 0 for action in actions)


def test_holds_a_captured_point_while_the_counterattack_is_near():
    soldier = entity(10, 0, 0, 0, -4000)
    held = objective(1, 0, -4000, blue=0, red=6, percent=1)
    nxt = objective(2, 8000, -4000)
    attacker = entity(20, 1, 0, 5000, -4000)
    action = algorithm([soldier]).act(state(
        [soldier], [attacker], [held, nxt]
    ))[0]
    assert action.points[0] < 2000


def test_advances_after_the_captured_point_is_clear():
    soldier = entity(10, 0, 0, 0, -4000)
    held = objective(1, 0, -4000, blue=0, red=6, percent=1)
    nxt = objective(2, 8000, -4000)
    action = algorithm([soldier]).act(state([soldier], objectives=[held, nxt]))[0]
    assert action.command == "Moving"
    assert action.points[0] > 4000


def test_straggler_catches_the_column_before_the_objective():
    rear = entity(10, 0, 0, 0, 0)
    middle = entity(11, 0, 1, 15000, 0)
    lead = entity(12, 0, 2, 40000, 0)
    target = objective(1, 50000, 0)
    actions = algorithm([rear, middle, lead]).act(state(
        [rear, middle, lead], objectives=[target]
    ))
    assert actions[0].command == "Moving"
    assert 5000 < actions[0].points[0] < 20000
    assert actions[2].command == "Guard"
    assert actions[2].points[0] > 30000


def test_uavs_scout_separate_objectives():
    soldier = entity(10, 0, 0, 0, 0)
    first = entity(31, 0, 1, 0, 6000, kind="BP_Base_UAV_C")
    second = entity(32, 0, 2, 100, 6500, kind="BP_Base_UAV_C")
    right = objective(1, 4000, -3000)
    left = objective(2, 4000, 9000)
    actions = algorithm([soldier, first, second]).act(state(
        [soldier, first, second], objectives=[right, left]
    ))
    assert actions[1].command == "Moving"
    assert actions[2].command == "Moving"
    assert actions[1].points[1] != actions[2].points[1]


def test_dead_agent_stays_idle():
    dead = entity(10, 0, 0, 0)
    dead.alive = False
    dead.hp = 0
    action = algorithm([dead]).act(state([dead]))[0]
    assert action.command == "Idle"
