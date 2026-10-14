from dataclasses import replace

import numpy as np
import pytest

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RuleAgentAlgorithm import ReinforceAgentAlgorithm as RedRule
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from cssim.algorithms import AlgorithmContext
from cssim.environment.models import Entity, KeyObject, TeamState
from cssim.protocol import ActionSetFactory


def entity(uid, team, index, x, y=0, *, kind="BP_BaseSoldier_C", hp=100):
    return Entity.from_property({
        "uId": uid, "agentTeam": team, "type": kind, "agentHp": hp,
        "fireRange": 1000, "perceptionRange": 10000,
        "initLocation": {"x": x, "y": y, "z": 0},
    }, index)


def objective(uid, x, y=0, *, blue=0, red=0, percent=0, name=""):
    return KeyObject.from_raw({
        "uId": uid, "location": {"x": x, "y": y, "z": 0},
        "blueteamNum": blue, "redTeamNum": red, "percent": percent, "rSVD1": name,
    })


def state(agents, enemies=(), objectives=(), step=0, episode=1):
    return TeamState(0, np.ones((len(agents), 1), dtype=bool), tuple(agents),
                     tuple(enemies), {}, episode, step, False, key_objects=tuple(objectives))


def algorithm(agents):
    return RedRule(AlgorithmContext(0, tuple(agents), (), ActionSetFactory.standard(), "cpu", {}),
                   enable_parent_assignment=False)


def map_objectives():
    return [objective(uid, *point, name=name) for uid, point, name in
            zip((901, 902, 903), AssaultRoute.ANCHORS, ("HQ", "B", "A"))]


def guards_at(obj, first=100, count=4):
    return [entity(first + i, 0, i, obj.position[0], obj.position[1]) for i in range(count)]


def test_local_contact_at_95_m_is_attacked_despite_generic_10_m_metadata():
    soldier = entity(10, 0, 0, 0)
    enemy = entity(20, 1, 0, 9500)
    soldier.raw["agentPerception"] = [enemy.uid]
    action = algorithm([soldier]).act(state([soldier], [enemy]))[0]
    assert action.command == "NormalAttacking" and action.target_uid == 20


def test_nearby_shared_contact_gets_an_attack_request_without_fabricating_local_sight():
    soldier = entity(10, 0, 0, 0)
    soldier.raw["agentPerception"] = []
    enemy = entity(20, 1, 0, 4000)
    action = algorithm([soldier]).act(state([soldier], [enemy]))[0]
    assert action.command == "NormalAttacking" and action.target_uid == 20
    assert soldier.raw["agentPerception"] == []


def test_common_contact_receives_concentrated_fire():
    agents = [entity(1, 0, 0, 0), entity(2, 0, 1, 100)]
    enemies = [entity(21, 1, 0, 800), entity(22, 1, 1, 4000)]
    agents[0].raw["agentPerception"] = [21, 22]
    agents[1].raw["agentPerception"] = [22]
    actions = algorithm(agents).act(state(agents, enemies))
    assert [a.target_uid for a in actions] == [22, 22]


def test_all_ground_types_join_lower_road_without_waiting_or_three_lanes():
    agents = [entity(i, 0, i, 60000, 12000, kind=kind) for i, kind in enumerate([
        "BP_BaseSoldier_C", "BP_RoboDog_C", "BP_MNWS_Vehicle_Armored_C", "BP_MNWS_Vehicle_6x6UGV_C"])]
    actions = algorithm(agents).act(state(agents, objectives=map_objectives()))
    assert all(a.command == "Moving" and a.points == AssaultRoute.LOWER[1] for a in actions)
    # Full terrain-aware waypoint is retained, not a 20m segment inside a building.
    assert actions[0].points[2] == 6082


def test_forward_soldier_does_not_wait_for_stuck_or_disconnected_rear():
    lead = entity(1, 0, 0, 15000)
    rear = [entity(10 + i, 0, i + 1, 0, kind="BP_RoboDog_C") for i in range(8)]
    for dog in rear:
        dog.raw["rSVD1"] = "dog;0"
    agents = [lead, *rear]
    rule = algorithm(agents)
    actions = [rule.act(state(agents, objectives=[objective(100, 20000)], step=s)) for s in range(6)]
    assert sum(row[0].command == "Moving" for row in actions) == 4
    assert all(a.command == "Moving" for row in actions for a in row[1:])


def test_casualties_do_not_change_survivors_route():
    one = entity(1, 0, 0, 60000, 12000)
    two = entity(2, 0, 1, 60000, 12000)
    rule = algorithm([one, two])
    before = rule.act(state([one, two], objectives=map_objectives()))[1]
    one.alive = False
    after = rule.act(state([one, two], objectives=map_objectives(), step=1))[1]
    assert after == before


def test_capture_sequence_does_not_skip_empty_bases_or_switch_to_larger_enemy_count():
    agent = entity(1, 0, 0, 60000, 35000)
    objects = map_objectives()
    objects[2].raw["blueteamNum"] = 50
    agents = [agent, *guards_at(objects[0]), *guards_at(objects[1], 200)]
    route = AssaultRoute()
    route.update(state([agent], objectives=objects))
    assert route.objective(state([agent], objectives=objects)).uid == 901
    objects[0].raw.update(percent=1, redTeamNum=20, blueteamNum=1)
    for step in range(1, 5):
        route.update(state(agents, objectives=objects, step=step))
    assert route.stage == 0  # A full bar with surviving occupants is still contested.
    objects[0].raw["blueteamNum"] = 0
    for step in range(5, 9):
        route.update(state(agents, objectives=objects, step=step))
    assert route.stage == 1  # Empty B still needs its 800-point capture.
    objects[1].raw.update(percent=1, redTeamNum=3)
    for step in range(9, 13):
        route.update(state(agents, objectives=objects, step=step))
    assert route.stage == 2
    route.update(state([agent], objectives=map_objectives(), episode=2))
    assert route.stage == 0


def test_rear_keeps_its_waypoint_when_front_captures_hq():
    rear = entity(1, 0, 0, 60000, 12000)
    front = entity(2, 0, 1, -41810, 36275)
    objects = map_objectives()
    route = AssaultRoute()
    initial = state([rear, front, *guards_at(objects[0])], objectives=objects)
    route.update(initial)
    rear_point = route.point(initial, rear)
    route.point(initial, front)
    objects[0].raw.update(percent=1, redTeamNum=5)
    for step in [1, 2, 3, 4]:
        current = replace(initial, step=step)
        route.update(current)
    assert route.point(current, rear) == rear_point
    assert route.point(current, front) == AssaultRoute.HQ_B[0]


def test_uav_only_bombs_confirmed_local_contact_and_flies_to_bounded_points():
    uav = entity(1, 0, 0, 60000, 12000, kind="BP_Base_UAV_C")
    uav.raw["agentPerception"] = []
    rule = algorithm([uav])
    objects = map_objectives()
    objects[0].raw["blueteamNum"] = 20
    move = rule.act(state([uav], objectives=objects))[0]
    assert move.command == "Moving" and move.points[2] == 8000
    assert AssaultRoute.distance(move.points, uav.position) == pytest.approx(25000)
    assert AssaultRoute.distance(move.points, objects[0].position) < AssaultRoute.distance(uav.position, objects[0].position)
    first_leg = move.points
    uav.position[2] = 8000
    move = rule.act(state([uav], objectives=objects, step=1))[0]
    assert move.points == first_leg
    enemy = entity(20, 1, 0, 61000, 12000)
    uav.raw["agentPerception"] = [20]
    bomb = rule.act(state([uav], [enemy], objects, step=2))[0]
    assert bomb.command == "SelfDestruct" and bomb.points == (61000, 12000, 0)


def test_unsuccessful_attack_moves_closer_but_resumes_firing():
    soldier = entity(1, 0, 0, 0)
    enemy = entity(20, 1, 0, 9000)
    soldier.raw["agentPerception"] = [20]
    rule = algorithm([soldier])
    commands = [rule.act(state([soldier], [enemy], step=s))[0].command for s in range(12)]
    assert commands[:3] == ["NormalAttacking"] * 3
    assert commands[3:5] == ["Moving"] * 2
    assert commands[5] == "NormalAttacking"


def test_dead_agent_stays_idle():
    dead = entity(1, 0, 0, 0)
    dead.alive = False
    assert algorithm([dead]).act(state([dead]))[0].command == "Idle"
