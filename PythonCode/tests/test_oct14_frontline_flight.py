"""Regressions identified in the actual 1200/1700/2840 run, not score forecasts."""
import math

from cssim.protocol import Action
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.CombatReadiness import CombatReadiness
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.FlightNavigation import FlightNavigation
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.MotionRecovery import MotionRecovery
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.UrbanContact import UrbanContact
from test_fslutk_rule import entity, state, map_objectives, algorithm
from test_air_and_departure import aircraft


def battle(stage=0, count=37, enemy=20):
    objects = map_objectives()
    obj = objects[stage]
    obj.raw['blueteamNum'] = enemy
    troops = [entity(i+1, 0, i, obj.position[0]+15000, obj.position[1]) for i in range(count)]
    cars = [entity(101+i, 0, count+i, obj.position[0]+60000, obj.position[1],
                   kind='BP_MNWS_Vehicle_Armored_C') for i in range(5)]
    for u in troops+cars:
        u.raw['interaction'] = ['200']
    route = AssaultRoute()
    route.update(state(troops+cars, objectives=objects))
    route.stage = stage
    return troops, cars, objects, route, CombatReadiness()


def test_remote_loaded_cars_cannot_mask_first_objective_collapse():
    troops, cars, objects, route, ready = battle()
    ready.begin(state(troops+cars, objectives=objects), route)
    for u in troops[14:]:
        u.alive = False
    ready.begin(state(troops+cars, objectives=objects, step=5), route)
    assert ready.front_power == 14 and ready.rear_power == 10
    assert ready.weak


def test_reinforcement_counts_only_after_arrival_and_with_ammo_and_link():
    troops, cars, objects, route, ready = battle(stage=1, count=4, enemy=12)
    ready.begin(state(troops+cars, objectives=objects), route)
    assert ready.weak and ready.front_power == 4
    for u in cars:
        u.position[0] = objects[1].position[0] + 15000
        u.raw['rSVD1'] = 'car;0'
    ready.begin(state(troops+cars, objectives=objects, step=1), route)
    assert ready.weak
    for u in cars:
        u.raw.update(rSVD1='car;1', interaction=['0'])
    ready.begin(state(troops+cars, objectives=objects, step=2), route)
    assert ready.weak
    for u in cars:
        u.raw['interaction'] = ['200']
    for step in range(3, 6):
        ready.begin(state(troops+cars, objectives=objects, step=step), route)
    assert not ready.weak and ready.front_power == 14


def test_reported_guns_outside_capture_box_are_not_ignored():
    troops, cars, objects, route, ready = battle(stage=1, count=4, enemy=1)
    threats = [entity(200+i, 1, i, objects[1].position[0]+19000, objects[1].position[1]) for i in range(12)]
    ready.begin(state(troops+cars, threats, objects), route)
    assert ready.enemy_pressure == 12 and ready.weak
    for u in threats:
        u.alive = False
    for step in range(1, 4):
        ready.begin(state(troops+cars, threats, objects, step), route)
    assert ready.enemy_pressure == 1 and not ready.weak


def test_preservation_does_not_hold_armed_reinforcements_out_of_support_range():
    troops, cars, objects, route, ready = battle(stage=1, count=4, enemy=12)
    for u in troops+cars:
        u.position[0] = objects[1].position[0] + 24500
    s = state(troops+cars, objectives=objects)
    ready.begin(s, route)
    assert ready.weak
    assert ready.choose(s, troops[0], route, UrbanContact(), []).command == 'Guard'
    assert ready.choose(s, cars[0], route, UrbanContact(), []) is None


def test_full_army_still_departs_even_when_no_unit_is_yet_near_objective():
    troops, cars, objects, route, ready = battle(enemy=40)
    for u in troops:
        u.position[0] += 70000
    for step in range(25):
        ready.begin(state(troops+cars, objectives=objects, step=step), route)
        assert not ready.weak and ready.front_power == 0


def test_one_good_strength_sample_cannot_alternate_fallback_and_advance():
    troops, cars, objects, route, ready = battle(stage=1, count=4, enemy=12)
    ready.begin(state(troops+cars, objectives=objects), route)
    for step in range(1, 9):
        objects[1].raw['blueteamNum'] = 1 if step % 2 else 12
        ready.begin(state(troops+cars, objectives=objects, step=step), route)
        assert ready.weak
    objects[1].raw['blueteamNum'] = 0
    ready.begin(state(troops+cars, objectives=objects, step=9), route)
    assert not ready.weak  # Empty points can be captured immediately.


def test_almost_exhausted_capture_units_cannot_be_replaced_by_loaded_cars():
    troops, cars, objects, route, ready = battle(enemy=10)
    ready.begin(state(troops+cars, objectives=objects), route)
    for u in troops[3:]:
        u.alive = False
    for u in cars:
        u.position[0] = objects[0].position[0]+10000
    ready.begin(state(troops+cars, objectives=objects, step=30), route)
    assert ready.front_power == 13 and ready.weak


def test_direct_flight_goal_survives_changing_reports_without_intermediate_hover():
    uav = aircraft(z=1000)
    nav = FlightNavigation()
    first = nav.apply(state([uav]), uav, (90000, 0, 14000), 1)
    assert first.points == (90000, 0, 14000)
    for step in range(1, 5):
        uav.position[0] += 5000
        assert nav.apply(state([uav], step=step), uav, (90000, step*1000, 14000), 1) == first
    uav.position[0] = 25000
    assert nav.apply(state([uav], step=5), uav, (90000, 5000, 14000), 1) == first
    assert not nav.resets
    assert uav.position[2] == 1000  # Horizontal leg advancement never waits for climbing.


def test_stationary_flight_resets_once_then_reissues_instead_of_direction_loop():
    uav = aircraft()
    nav = FlightNavigation()
    for step in range(9):
        action = nav.apply(state([uav], step=step), uav, (-90000, 0, 14000), 1)
    assert action.command == 'Guard' and action.direction == '-X'
    assert nav.apply(state([uav], step=8), uav, (-90000, 0, 14000), 1) == action
    assert nav.resets[uav.uid] == 1
    assert nav.apply(state([uav], step=9), uav, (-90000, 0, 14000), 1) == action
    assert nav.apply(state([uav], step=10), uav, (-90000, 0, 14000), 1).points == (-90000, 0, 14000)


def test_flight_advances_with_model_that_latches_moving_goal_until_action_transition():
    # This models the observed failure hypothesis, not the actual UE physics.
    uav = aircraft()
    nav = FlightNavigation()
    latched = None
    for step in range(30):
        action = nav.apply(state([uav], step=step), uav, (100000, 0, 8000), 1)
        if action.command != 'Moving':
            latched = None
        else:
            latched = latched or action.points
            delta = latched[0]-uav.position[0]
            if abs(delta) > 2500:
                uav.position[0] += math.copysign(min(5000, abs(delta)-2500), delta)
    assert uav.position[0] >= 97000
    assert nav.resets.get(uav.uid, 0) == 0


def test_flight_objective_change_resets_but_new_episode_drops_old_pending_goal():
    uav = aircraft()
    nav = FlightNavigation()
    nav.apply(state([uav]), uav, (100000, 0, 8000), 1)
    assert nav.apply(state([uav], step=1), uav, (-100000, 0, 8000), 2).command == 'Guard'
    assert nav.apply(state([uav], step=0, episode=2), uav, (0, 100000, 8000), 3).points == (0, 100000, 8000)


def test_local_strike_preempts_pending_flight_reset():
    uav = aircraft(x=60000)
    rule = algorithm([uav])
    objects = map_objectives()
    for step in range(9):
        action = rule.act(state([uav], objectives=objects, step=step))[0]
    assert action.command == 'Guard'
    enemy = entity(20, 1, 0, 59000)
    uav.raw['agentPerception'] = [20]
    action = rule.act(state([uav], [enemy], objects, 9))[0]
    assert action.command == 'SelfDestruct' and not rule._uav._navigation._pending


def test_reverse_heading_request_never_replaces_road_with_repeated_side_arcs():
    car = entity(981, 0, 0, 60000, 35000, kind='BP_MNWS_Vehicle_Armored_C')
    goal = Action.move_at((20000, 35000, 6400))
    recovery = MotionRecovery()
    for step in range(20):
        car.position[0] -= 250
        action = recovery.apply(state([car], step=step), car, goal)
        if step < 2:
            assert action.command == 'Guard' and action.direction == '-X'
        else:
            assert action == goal
