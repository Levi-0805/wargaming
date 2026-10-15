"""Regressions from the 20261013_192519 replay, plus bounded recovery checks."""
import math

from cssim.protocol import Action
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.MotionRecovery import MotionRecovery
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.UAVControl import UAVControl
from test_fslutk_rule import entity, state, algorithm, map_objectives, guards_at
from test_air_and_departure import aircraft


def test_replay_brief_full_capture_during_collapse_does_not_send_cars_to_b():
    route = AssaultRoute()
    objects = map_objectives()
    for step, (red, blue, percent) in enumerate([
            (20, 16, .4), (20, 16, 1), (19, 16, 1), (17, 16, 1),
            (15, 15, 1), (11, 15, .75), (12, 15, .35), (8, 15, 0)]):
        objects[0].raw.update(redTeamNum=red, blueteamNum=blue, percent=percent)
        route.update(state([], objectives=objects, step=step))
        assert route.stage == 0


def test_lost_hq_returns_to_hq_after_confirmed_capture():
    route = AssaultRoute()
    objects = map_objectives()
    objects[0].raw.update(redTeamNum=5, percent=1)
    guards = guards_at(objects[0])
    for step in range(4):
        route.update(state(guards, objectives=objects, step=step))
    assert route.stage == 1
    objects[0].raw.update(redTeamNum=0, blueteamNum=5, percent=0)
    for step in (4, 5):
        route.update(state([], objectives=objects, step=step))
    assert route.stage == 0


def test_retake_b_does_not_send_forward_units_back_to_hq_approach():
    route = AssaultRoute()
    objects = map_objectives()
    car = entity(1, 0, 0, -3000, -27530, kind="BP_MNWS_Vehicle_Armored_C")
    agents = [car, *guards_at(objects[0]), *guards_at(objects[1], 200)]
    for obj in objects[:2]:
        obj.raw.update(redTeamNum=5, percent=1)
    for step in range(8):
        route.update(state(agents, objectives=objects, step=step))
    route._cursor[car.uid] = (2, 0)
    objects[1].raw.update(redTeamNum=0, blueteamNum=5, percent=0)
    for step in (8, 9):
        current = state(agents, objectives=objects, step=step)
        route.update(current)
    assert route.stage == 1
    point = route.point(current, car)
    assert point[1] < 0  # Rejoin B's approach, not HQ's positive-Y approach.


def test_slow_backward_car_turns_even_though_it_is_making_net_progress():
    car = entity(481, 0, 1, 60818, 35411, kind="BP_MNWS_Vehicle_Armored_C")
    car.yaw = -17.5
    move = Action.move_at((49548, 34208, 5747))
    control = MotionRecovery()
    for step in range(3):
        car.position[0] -= 280  # Same slow displacement seen in the replay.
        action = control.apply(state([car], step=step), car, move)
        if step < 2:
            assert action.command == 'Guard' and action.direction == '-X'
        else:
            assert action is move
    car.yaw = -175
    assert control.apply(state([car], step=3), car, move) is move


def test_vehicle_turn_attempt_is_bounded_and_restores_normal_waypoint():
    car = entity(1, 0, 0, 0, kind="BP_MNWS_Vehicle_Armored_C")
    move = Action.move_at((-20000, 0, 0))
    recovery = MotionRecovery()
    for step in range(2):
        assert recovery.apply(state([car], step=step), car, move).command == 'Guard'
    for step in range(2, 8):
        assert recovery.apply(state([car], step=step), car, move) is move


def test_uav_advances_horizontally_while_low_and_can_strike_without_height_gate():
    uav = aircraft(z=1000)
    enemy = entity(20, 1, 0, 10000)
    control = UAVControl()
    action = control.choose(state([uav], [enemy]), uav, None)
    assert action.points == (10000, 0, 8000)
    action = control.choose(state([uav], [enemy], step=1), uav, None)
    assert action.points == (10000, 0, 8000)
    uav.position[0] = 9000
    uav.raw["agentPerception"] = [20]
    for step in (2, 3):
        assert control.choose(state([uav], [enemy], step=step), uav, None).command != "SelfDestruct"
    assert control.choose(state([uav], [enemy], step=4), uav, None).command == "SelfDestruct"


def test_uav_does_not_spend_strike_on_enemy_aircraft():
    uav = aircraft()
    enemy = aircraft(20, x=1000)
    enemy.team = 1
    uav.raw["agentPerception"] = [20]
    assert UAVControl().choose(state([uav], [enemy]), uav, None).command != "SelfDestruct"


def test_distant_shared_contact_does_not_pin_infantry_in_place():
    soldier = entity(1, 0, 0, 60000, 12000)
    soldier.raw["agentPerception"] = []
    enemy = entity(20, 1, 0, -40000, 30000)
    action = algorithm([soldier]).act(state([soldier], [enemy], map_objectives()))[0]
    assert action.command == "Moving"


def test_no_ammo_consumption_repositions_sideways_without_charging_enemy():
    soldier = entity(1, 0, 0, 0)
    enemy = entity(20, 1, 0, 4000)
    soldier.raw.update(agentPerception=[20], interaction=["250"])
    rule = algorithm([soldier])
    for step in range(4):
        action = rule.act(state([soldier], [enemy], step=step))[0]
    assert action.command == "Moving"
    assert abs(action.points[1]) == 800 and action.points[0] == 0


def test_actual_shooting_does_not_get_cancelled_for_periodic_repositioning():
    soldier = entity(1, 0, 0, 0)
    enemy = entity(20, 1, 0, 4000)
    soldier.raw["agentPerception"] = [20]
    rule = algorithm([soldier])
    for step in range(12):
        soldier.raw["interaction"] = [str(250 - step)]
        enemy.hp = 100 - step * 2
        assert rule.act(state([soldier], [enemy], step=step))[0].command == "NormalAttacking"


def test_final_approach_spreads_slots_without_waiting_for_rear_units():
    agents = [entity(i, 0, i, -17959, 27728) for i in range(5)]
    actions = algorithm(agents).act(state(agents, objectives=map_objectives()))
    assert all(a.command == "Moving" for a in actions)
    assert len({a.points for a in actions}) == 5
