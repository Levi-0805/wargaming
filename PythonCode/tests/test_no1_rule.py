from __future__ import annotations

import numpy as np

from TeamAlg.No1RuleRed.Algorithm.RLAgentAlgorithm import ReinforceAgentAlgorithm
from TeamAlg.No1RuleRed.Algorithm.RuleAgentAlgorithm import (
    ReinforceAgentAlgorithm as No1Rule,
)
from cssim.algorithms import AlgorithmContext
from cssim.environment.models import Entity, KeyObject, TeamState
from cssim.protocol import ActionSetFactory


def entity(uid, team, index, x, *, kind="BP_BaseSoldier_C", hp=100):
    return Entity.from_property({
        "uId": uid,
        "agentTeam": team,
        "type": kind,
        "agentHp": hp,
        "perceptionRange": 5000,
        "initLocation": {"x": x, "y": 0, "z": 0},
    }, index)


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
    return ReinforceAgentAlgorithm(context)


def test_no1_attacks_visible_high_priority_target():
    soldier = entity(10, 0, 0, 0)
    far_soldier = entity(20, 1, 0, 100)
    nearer_uav = entity(21, 1, 1, 20, kind="BP_Base_UAV_C", hp=80)
    actions = algorithm([soldier]).act(state([soldier], [far_soldier, nearer_uav]))
    assert actions[0].command == "NormalAttacking"
    assert actions[0].target_uid == 21


def test_no1_assigns_objectives_and_guards_inside_radius():
    first = entity(10, 0, 0, 0)
    second = entity(11, 0, 1, 950)
    objective = KeyObject.from_raw({"uId": 300, "team": 1, "location": {"x": 0, "y": 0, "z": 0}})
    actions = algorithm([first, second]).act(state([first, second], objectives=[objective]))
    assert actions[0].command == "Guard"
    assert actions[1].command == "Moving"
    assert actions[1].points == (0.0, 0.0, 0.0)


def test_no1_keeps_dead_agent_idle():
    dead = entity(10, 0, 0, 0)
    dead.alive = False
    dead.hp = 0
    action = algorithm([dead]).act(state([dead]))[0]
    assert action.command == "Idle"


def test_no1_entry_rejects_non_rule_mode():
    agent = entity(10, 0, 0, 0)
    context = AlgorithmContext(
        0, (agent,), (), ActionSetFactory.standard(), "cpu", {"mode": "reinforcement_learning"}
    )
    try:
        ReinforceAgentAlgorithm(context)
    except ValueError as exc:
        assert "mode=rule" in str(exc)
    else:
        raise AssertionError("No.1 应拒绝非规则模式")
