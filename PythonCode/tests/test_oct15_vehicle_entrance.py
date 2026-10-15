"""Entrance progress regressions from the first round of 20261015_135458."""
import math

import pytest

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.HQEntry import HQEntry
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.UrbanContact import UrbanContact
from test_fslutk_rule import entity, state, map_objectives


def case(kind='BP_MNWS_Vehicle_Armored_C', point=(-35362, 30478, 6335)):
    unit = entity(1, 0, 0, point[0], point[1], kind=kind)
    unit.position[:] = point
    unit.raw.update(rSVD1='vehicle;1', interaction=['200'], agentPerception=[])
    objects = map_objectives()
    objects[0].raw.update(blueteamNum=18, scale={'x': 7700, 'y': 6500})
    route = AssaultRoute()
    route.update(state([unit], objectives=objects))
    return unit, objects, route, UrbanContact(), HQEntry()


@pytest.mark.parametrize('kind,point', [
    ('BP_MNWS_Vehicle_6x6UGV_C', (-36291, 29355, 6335)),
    ('BP_MNWS_Vehicle_6x6UGV_C', (-35495, 29785, 6335)),
    ('BP_MNWS_Vehicle_Armored_C', (-36540, 30428, 6335)),
    ('BP_MNWS_Vehicle_Armored_C', (-35362, 30478, 6335)),
    ('BP_MNWS_Vehicle_Armored_C', (-35595, 30953, 6335)),
    ('BP_MNWS_Vehicle_Armored_C', (-36653, 31255, 6335)),
])
def test_recorded_facade_stops_advance_past_first_waypoint(kind, point):
    unit, objects, route, urban, entry = case(kind, point)
    for step in range(19):
        action = entry.choose(state([unit], objectives=objects, step=step), unit, route, urban, [])
    assert action.command == 'Moving'
    assert entry._routes[unit.uid] >= 2
    assert action.points != HQEntry.PATH[0]


def test_ten_metre_vehicle_stop_model_completes_corners_without_detour():
    unit, objects, route, urban, entry = case(point=(-32000, 28000, 6335))
    for step in range(19):
        action = entry.choose(state([unit], objectives=objects, step=step), unit, route, urban, [])
    visited = []
    while action is not None:
        visited.append(action.points)
        delta = unit.position[:2] - action.points[:2]
        distance = math.hypot(*delta)
        unit.position[:2] = action.points[:2] + delta * (1000 / distance)
        unit.position[2] = action.points[2]
        step += 1
        action = entry.choose(state([unit], objectives=objects, step=step), unit, route, urban, [])
        assert step < 40
    assert visited == list(HQEntry.VEHICLE_PATH)
    assert unit.uid in entry._completed and not entry._alternate


def test_infantry_keeps_its_original_arrival_radius_and_path():
    unit, objects, route, urban, entry = case('BP_BaseSoldier_C')
    for step in range(19):
        action = entry.choose(state([unit], objectives=objects, step=step), unit, route, urban, [])
    assert entry._routes[unit.uid] == 0 and action.points == HQEntry.PATH[0]


def test_same_unreachable_vehicle_waypoint_switches_to_recorded_other_approach():
    unit, objects, route, urban, entry = case(point=(-32000, 28000, 6335))
    outputs = []
    for step in range(36):
        action = entry.choose(state([unit], objectives=objects, step=step), unit, route, urban, [])
        if action is not None:
            outputs.append(action.points)
    assert unit.uid in entry._alternate
    assert outputs[-1] in HQEntry.VEHICLE_ALTERNATE
    assert any(p != outputs[0] for p in outputs)


def test_local_circles_and_duplicate_calls_do_not_reset_or_accelerate_timeout():
    unit, objects, route, urban, entry = case(point=(-32000, 28000, 6335))
    for step in range(19):
        entry.choose(state([unit], objectives=objects, step=step), unit, route, urban, [])
    for _ in range(30):
        entry.choose(state([unit], objectives=objects, step=18), unit, route, urban, [])
    assert not entry._alternate
    for step in range(19, 36):
        unit.position[1] = 28000 + (step % 2) * 100
        entry.choose(state([unit], objectives=objects, step=step), unit, route, urban, [])
    assert unit.uid in entry._alternate


def test_effective_fire_does_not_consume_navigation_trials():
    unit, objects, route, urban, entry = case(point=(-32000, 28000, 6335))
    enemy = entity(20, 1, 0, -35000, 30000)
    for step in range(19):
        entry.choose(state([unit], [enemy], objects, step), unit, route, urban, [enemy])
    goal = entry._routes[unit.uid]
    for step in range(19, 50):
        urban._effective[unit.uid] = (enemy.uid, step)
        action = entry.choose(state([unit], [enemy], objects, step), unit, route, urban, [enemy])
        assert action is None
    assert not entry._alternate and entry._routes[unit.uid] == goal
    assert entry.choose(state([unit], [enemy], objects, 51), unit, route, urban, [enemy]).command == 'Moving'
    assert not entry._alternate


def test_failed_alternate_cannot_pin_a_vehicle_to_one_goal_for_the_rest_of_round():
    unit, objects, route, urban, entry = case(point=(-32000, 28000, 6335))
    for step in range(400):
        entry.choose(state([unit], objectives=objects, step=step), unit, route, urban, [])
    assert unit.uid in entry._exhausted and unit.uid not in entry._completed
    assert unit.uid not in entry._routes
    assert entry.choose(state([unit], objectives=objects, step=400), unit, route, urban, []) is None


def test_episode_change_resets_failed_and_alternate_entrances():
    unit, objects, route, urban, entry = case(point=(-32000, 28000, 6335))
    for step in range(400):
        entry.choose(state([unit], objectives=objects, step=step), unit, route, urban, [])
    s = state([unit], objectives=objects, episode=2)
    route.update(s)
    assert entry.choose(s, unit, route, urban, []) is None
    assert not entry._alternate and not entry._exhausted and not entry._leg_watch


def test_fallback_never_reissues_a_corner_already_within_vehicle_stop_radius():
    unit, objects, route, urban, entry = case(point=(-36540, 30428, 6335))
    for step in range(400):
        action = entry.choose(state([unit], objectives=objects, step=step), unit, route, urban, [])
        if action is not None:
            assert AssaultRoute.distance(unit.position, action.points) > HQEntry.VEHICLE_ARRIVAL


def test_disconnected_vehicle_does_not_replan_entrance():
    unit, objects, route, urban, entry = case()
    unit.raw['rSVD1'] = 'vehicle;0'
    for step in range(40):
        assert entry.choose(state([unit], objectives=objects, step=step), unit, route, urban, []) is None
    assert not entry._routes and not entry._alternate
