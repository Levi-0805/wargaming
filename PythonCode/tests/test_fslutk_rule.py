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


def objective(uid, x, *, blue=0, red=0, percent=0):
    return KeyObject.from_raw({
        "uId": uid,
        "location": {"x": x, "y": 0, "z": 0},
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


def test_whole_force_moves_to_same_weaker_objective():
    agents = [entity(10, 0, 0, 0), entity(11, 0, 1, 100)]
    crowded = objective(1, 0, blue=20)
    empty = objective(2, 5000, blue=0)
    actions = algorithm(agents).act(state(agents, objectives=[crowded, empty]))
    assert all(action.command == "Moving" for action in actions)
    assert all(action.points[0] > 2000 for action in actions)


def test_leaves_a_captured_point_for_the_next_objective():
    agents = [entity(10, 0, 0, 0)]
    held = objective(1, 0, blue=0, red=6, percent=1)
    nxt = objective(2, 8000, blue=0, red=0, percent=0)
    action = algorithm(agents).act(state(agents, objectives=[held, nxt]))[0]
    assert action.command == "Moving"
    assert action.points[0] > 4000


def test_dead_agent_stays_idle():
    dead = entity(10, 0, 0, 0)
    dead.alive = False
    dead.hp = 0
    action = algorithm([dead]).act(state([dead]))[0]
    assert action.command == "Idle"
