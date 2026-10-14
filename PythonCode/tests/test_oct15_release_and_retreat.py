"""Regressions from 20261014_211207: endless retreat, B exit, rear UAVs."""
from dataclasses import replace

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.FlightNavigation import FlightNavigation
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.UAVControl import UAVControl
from test_fslutk_rule import entity, guards_at, map_objectives, state
from test_facade_and_support import readiness_case
from test_air_and_departure import aircraft


def test_weak_armed_survivor_cannot_be_held_for_200_steps():
    units, objects, route, ready, urban = readiness_case()
    for unit in units:
        unit.position[0] += 3000
    streak = maximum = 0
    for step in range(250):
        s = state(units, objectives=objects, step=step)
        ready.begin(s, route)
        action = ready.choose(s, units[0], route, urban, [])
        assert ready.weak
        if action is not None:
            assert action.command == 'Guard' and action.points == tuple(units[0].position)
        streak = streak + 1 if action else 0
        maximum = max(maximum, streak)
        if 6 <= step < 30:
            assert action is None
    assert maximum == 6


def test_collapsed_unit_inside_contested_hq_is_never_ordered_out():
    units, objects, route, ready, urban = readiness_case(stage=0)
    units += [entity(20+i, 0, 4+i, *objects[0].position[:2]) for i in range(12)]
    ready.begin(state(units, objectives=objects), route)
    for unit in units[4:]:
        unit.alive = False
    units[0].position[:] = objects[0].position
    objects[0].raw.update(percent=1, redTeamNum=1, blueteamNum=8)
    s = state(units, objectives=objects, step=1)
    ready.begin(s, route)
    assert ready.weak and ready.choose(s, units[0], route, urban, []) is None


def test_empty_car_is_not_sent_back_to_a_rear_fallback():
    units, objects, route, ready, urban = readiness_case()
    car = entity(40, 0, 4, objects[2].position[0]-24500, objects[2].position[1],
                 kind='BP_MNWS_Vehicle_Armored_C')
    car.raw['interaction'] = ['0']
    s = state([*units, car], objectives=objects)
    ready.begin(s, route)
    assert ready.weak and ready.choose(s, car, route, urban, []) is None


def test_brief_strength_fluctuation_cannot_reset_pause_deadline():
    units, objects, route, ready, urban = readiness_case()
    units[0].position[0] += 3000
    for step in range(20):
        s = state(units, objectives=objects, step=step)
        ready.begin(s, route)
        ready.weak = step != 3
        action = ready.choose(s, units[0], route, urban, [])
        if step >= 6:
            assert action is None


def test_friendly_motion_during_pause_cannot_trigger_return_to_scan_position():
    units, objects, route, ready, urban = readiness_case()
    units[0].position[0] += 3000
    s = state(units, objectives=objects)
    ready.begin(s, route)
    assert ready.choose(s, units[0], route, urban, []).command == 'Guard'
    units[0].position[0] += 900
    s = replace(s, step=1)
    ready.begin(s, route)
    assert ready.choose(s, units[0], route, urban, []) is None


def secured_b():
    objects = map_objectives()
    for obj in objects[:2]:
        obj.raw.update(percent=1, redTeamNum=8, blueteamNum=0)
    guards = guards_at(objects[0], count=6) + guards_at(objects[1], first=200, count=6)
    dog = entity(250, 0, 12, *objects[1].position[:2], kind='BP_RoboDog_C')
    guards.append(dog)
    route = AssaultRoute()
    route.update(state(guards, objectives=objects))
    route.stage = 1
    return guards, objects, route


def test_safe_secondary_base_releases_surplus_but_keeps_hq_reserve():
    units, objects, route = secured_b()
    for step in range(1, 8):
        route.update(state(units, objectives=objects, step=step))
        if step < 7:
            assert sum(uid == objects[1].uid for uid in route._garrison.values()) == 4
    assert route.stage == 2
    hq = [u for u in units if route._garrison.get(u.uid) == objects[0].uid]
    base = [u for u in units if route._garrison.get(u.uid) == objects[1].uid]
    assert len(hq) == 4 and len(base) == 2
    assert {u.entity_type for u in base} == {'BP_BaseSoldier_C', 'BP_RoboDog_C'}


def test_new_reported_ground_threat_cancels_small_garrison_immediately():
    units, objects, route = secured_b()
    for step in range(1, 8):
        route.update(state(units, objectives=objects, step=step))
    enemy = entity(9000, 1, 0, objects[1].position[0]+11000, objects[1].position[1])
    route.update(state(units, [enemy], objects, step=8))
    assert sum(uid == objects[1].uid for uid in route._garrison.values()) == 4
    assert route.garrison_needed(state(units, [enemy], objects, step=8), objects[1]) == 4


def test_missing_occupant_data_cannot_authorize_reducing_guards():
    units, objects, route = secured_b()
    objects[1].raw.pop('blueteamNum')
    for step in range(1, 20):
        route.update(state(units, objectives=objects, step=step))
    assert sum(uid == objects[1].uid for uid in route._garrison.values()) == 4


def test_discontinuous_observations_restart_secondary_clear_confirmation():
    units, objects, route = secured_b()
    for step in (1, 2, 3, 4, 20, 21, 22):
        route.update(state(units, objectives=objects, step=step))
    assert sum(uid == objects[1].uid for uid in route._garrison.values()) == 4


def test_released_b_guard_with_stale_hq_cursor_joins_b_to_a():
    units, objects, route = secured_b()
    for step in range(1, 7):
        route.update(state(units, objectives=objects, step=step))
    old = dict(route._garrison)
    for uid in old:
        route._cursor[uid] = (0, 0)
    s = state(units, objectives=objects, step=7)
    route.update(s)
    released = [u for u in units if old.get(u.uid) == objects[1].uid and u.uid not in route._garrison]
    assert released
    for unit in released:
        assert route._cursor[unit.uid][0] == 2
        assert route.point(s, unit)[1] < -20000  # Never returns to the HQ road.


def test_soldier_at_recorded_b_exit_wall_gets_reached_bypass_point():
    units, objects, route = secured_b()
    route.stage = 2
    unit = entity(776, 0, 20, -51029.65, -22090.09)
    s = state([*units, unit], objectives=objects, step=1)
    point = route.point(s, unit)
    assert point == route.B_A[0]
    assert route._cursor[unit.uid][0] == 2
    assert point != tuple(objects[2].position)


def test_recorded_rotated_base_boundary_does_not_eject_its_own_clamped_guard():
    obj = map_objectives()[1]
    obj.raw.update(scale={'x': 10500, 'y': 9500}, rotation={'yaw': -12.8125})
    route = AssaultRoute()
    for point in ((-51000, -19000, 5610), (-60000, -50000, 5610), (-53740, -21849, 5718)):
        clamped = route.clamp_guard_point(obj, point)
        assert route.inside_garrison(obj, clamped)


def test_direct_first_flight_goal_even_when_idle_would_not_cancel_latched_goal():
    uav = aircraft()
    nav = FlightNavigation()
    latched = None
    for step in range(24):
        action = nav.apply(state([uav], step=step), uav, (100000, 0, 8000), 1)
        if action.command == 'Moving':
            latched = latched or action.points  # Controller never adopts a replacement.
        if latched:
            uav.position[0] += min(5000, max(0, latched[0] - uav.position[0]))
    assert uav.position[0] == 100000
    assert not nav.resets


def test_initial_uav_goal_is_not_clamped_to_a_moving_rear_soldier():
    objects = map_objectives()
    uav = aircraft(x=60000, y=35000)
    soldier = entity(20, 0, 1, 60000, 35000)
    s = state([uav, soldier], objectives=objects)
    move = UAVControl().choose(s, uav, objects[0])
    assert move.points[:2] == tuple(objects[0].position[:2])


def test_disconnected_uav_cannot_continue_direct_flight_until_link_is_repaired():
    uav = aircraft()
    nav = FlightNavigation()
    nav.apply(state([uav]), uav, (100000, 0, 8000), 1)
    uav.raw['rSVD1'] = 'uav;0'
    assert nav.apply(state([uav], step=1), uav, (100000, 0, 8000), 1).command == 'Idle'
    assert not nav._legs
