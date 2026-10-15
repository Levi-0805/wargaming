import pytest

from cssim.protocol import Action
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.HQEntry import HQEntry
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.UrbanContact import UrbanContact
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.UAVControl import UAVControl
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.CombatReadiness import CombatReadiness
from test_fslutk_rule import entity, state, algorithm, map_objectives
from test_air_and_departure import aircraft


def courtyard_case():
    soldier = entity(1, 0, 0, -35000, 28500)
    soldier.position[2] = 6425
    soldier.raw.update(agentPerception=[], interaction=['250'])
    objectives = map_objectives()
    objectives[0].raw.update(blueteamNum=5, scale={'x': 7700, 'y': 6500})
    route = AssaultRoute()
    route.update(state([soldier], objectives=objectives))
    return soldier, objectives, route, UrbanContact(), HQEntry()


def test_facade_oscillation_escalates_despite_shared_target_changes():
    unit, objects, route, urban, entry = courtyard_case()
    goals = []
    for step in range(30):
        unit.position[0] = -35000 + (step % 2) * 400
        target = entity(20 + step % 2, 1, 0, -39000, 33000)
        action = entry.choose(state([unit], [target], objects, step), unit, route, urban, [target])
        if action is not None:
            goals.append(action.points)
    assert goals and set(goals) == {HQEntry.PATH[0]}


def test_entrance_cursor_passes_corners_without_returning_to_facade():
    unit, objects, route, urban, entry = courtyard_case()
    for step in range(19):
        action = entry.choose(state([unit], objectives=objects, step=step), unit, route, urban, [])
    visited = []
    while action is not None:
        assert action.points not in visited
        visited.append(action.points)
        unit.position[:] = action.points
        step += 1
        action = entry.choose(state([unit], objectives=objects, step=step), unit, route, urban, [])
        assert step < 50
    assert visited == list(HQEntry.PATH)
    assert entry.choose(state([unit], objectives=objects, step=step+30), unit, route, urban, []) is None


def test_entry_waypoints_keep_existing_hq_guards_in_hold_area():
    unit, objects, route, urban, entry = courtyard_case()
    objects[0].raw.update(percent=1, redTeamNum=10)
    route.update(state([unit], objectives=objects, step=1))
    assert route.garrison_for(state([unit], objectives=objects), unit) is not None
    assert all(route.inside_garrison(objects[0], p) for p in HQEntry.PATH)
    for step in range(1, 21):
        action = entry.choose(state([unit], objectives=objects, step=step), unit, route, urban, [])
    while action is not None:
        unit.position[:] = action.points
        step += 1
        action = entry.choose(state([unit], objectives=objects, step=step), unit, route, urban, [])
    assert route.guard_slot(unit, objects[0]) == tuple(unit.position)


def test_effective_ground_fire_pauses_entry_without_restarting_cursor():
    unit, objects, route, urban, entry = courtyard_case()
    enemy = entity(20, 1, 0, -38000, 33000)
    for step in range(19):
        action = entry.choose(state([unit], [enemy], objects, step), unit, route, urban, [enemy])
    unit.position[:] = action.points
    s = state([unit], [enemy], objects, 19)
    entry.choose(s, unit, route, urban, [enemy])
    urban.begin(s)
    urban.record(s, unit, Action.attack(enemy))
    unit.raw['interaction'] = ['249']
    enemy.hp = 90
    s = state([unit], [enemy], objects, 20)
    urban.begin(s)
    assert entry.choose(s, unit, route, urban, [enemy]) is None
    later = entry.choose(state([unit], [enemy], objects, 24), unit, route, urban, [enemy])
    assert later.points == HQEntry.PATH[1]


def test_unknown_map_and_disconnected_units_do_not_get_hq_fixed_coordinates():
    unit, objects, route, urban, entry = courtyard_case()
    route.known_map = False
    for step in range(30):
        assert entry.choose(state([unit], objectives=objects, step=step), unit, route, urban, []) is None
    route.known_map = True
    unit.raw['rSVD1'] = 'soldier;0'
    for step in range(30, 60):
        assert entry.choose(state([unit], objectives=objects, step=step), unit, route, urban, []) is None


@pytest.mark.parametrize('kind', ['BP_MNWS_Vehicle_Armored_C', 'BP_MNWS_Vehicle_6x6UGV_C'])
def test_vehicle_does_not_spend_its_ground_ammunition_on_air_contacts(kind):
    car = entity(1, 0, 0, -17959, 27728, kind=kind)
    air = entity(20, 1, 0, -20000, 30000, kind='BP_Base_UAV_C')
    air.position[2] = 18000
    car.raw.update(agentPerception=[20], interaction=['200'])
    rule = algorithm([car])
    for step in range(12):
        action = rule.act(state([car], [air], map_objectives(), step))[0]
        assert action.command != 'NormalAttacking'
    ground = entity(21, 1, 1, -18000, 28000)
    car.raw['agentPerception'].append(21)
    assert rule.act(state([car], [air, ground], map_objectives(), 12))[0].target_uid == 21


def test_five_aircraft_have_stable_slots_without_waiting_for_lagging_wing():
    uavs = [aircraft(i + 1, x=60000 + i*100, y=35000) for i in range(5)]
    obj = map_objectives()[0]
    control = UAVControl()
    actions = [control.choose(state(uavs, objectives=[obj]), uav, obj) for uav in uavs]
    assert all(a.command == 'Moving' for a in actions)
    assert len({a.points for a in actions}) == 5
    assert all(AssaultRoute.distance(a.points, obj.position) <= 6001 for a in actions)
    uavs[1].alive = False
    for i in (0, 2, 3, 4):
        assert control.choose(state(uavs, objectives=[obj], step=1), uavs[i], obj) == actions[i]


def test_aircraft_coordinate_stall_transitions_without_altitude_escalation_or_sideways_circle():
    uav = aircraft(x=0, y=0)
    enemy = entity(20, 1, 0, -40000, 3000)
    control = UAVControl()
    for step in range(9):
        action = control.choose(state([uav], [enemy], step=step), uav, None)
    assert action.command == 'Guard'
    assert control._altitude[uav.uid] == 8000
    uav.position[0] = -2000
    resumed = control.choose(state([uav], [enemy], step=10), uav, None)
    assert resumed.command == 'Moving' and resumed.points[0] < -24000
    assert resumed.points[2] == 8000
    assert control.choose(state([uav], [enemy], step=14), uav, None) == resumed


def test_aircraft_cannot_be_pulled_to_reports_at_an_unassigned_objective():
    obj = map_objectives()[0]
    uav = aircraft(x=60000, y=35000)
    other_base_enemy = entity(20, 1, 0, -3000, -27530)
    action = UAVControl().choose(state([uav], [other_base_enemy], [obj]), uav, obj)
    assert action.points[1] > 30000  # HQ approach, not the southern base report.
    assert AssaultRoute.distance(action.points, obj.position) < AssaultRoute.distance(uav.position, obj.position)


def test_aircraft_finishes_locked_strike_even_if_team_moves_to_next_objective():
    objects = map_objectives()
    uav = aircraft(x=-40000, y=30000)
    enemy = entity(20, 1, 0, -41000, 30000)
    uav.raw['agentPerception'] = [20]
    control = UAVControl()
    for step in range(3):
        first = control.choose(state([uav], [enemy], objects, step), uav, objects[0])
    assert first.command == 'SelfDestruct'
    assert control.choose(state([uav], [enemy], objects, 3), uav, objects[1]) == first


def readiness_case(stage=2):
    objects = map_objectives()
    obj = objects[stage]
    obj.raw['blueteamNum'] = 16
    units = [entity(i+1, 0, i, obj.position[0]-27000, obj.position[1]) for i in range(4)]
    for u in units:
        u.raw['interaction'] = ['100']
    route = AssaultRoute()
    route.update(state(units, objectives=objects))
    route.stage = stage
    return units, objects, route, CombatReadiness(), UrbanContact()


def test_depleted_third_assault_stays_at_reached_approach_and_resumes_after_enemy_reduction():
    units, objects, route, ready, urban = readiness_case()
    ready.begin(state(units, objectives=objects), route)
    assert ready.weak
    assert ready.choose(state(units, objectives=objects), units[0], route, urban, []) is None
    for unit in units:
        unit.position[0] += 3000
    s = state(units, objectives=objects, step=1)
    ready.begin(s, route)
    action = ready.choose(s, units[0], route, urban, [])
    assert action.command == 'Guard' and action.points == tuple(units[0].position)
    objects[2].raw['blueteamNum'] = 3
    for step in range(2, 5):
        s = state(units, objectives=objects, step=step)
        ready.begin(s, route)
    assert not ready.weak and ready.choose(s, units[0], route, urban, []) is None


def test_empty_vehicles_do_not_count_as_reinforcing_firepower():
    units, objects, route, ready, urban = readiness_case()
    cars = [entity(30+i, 0, 4+i, -30000, -27530, kind='BP_MNWS_Vehicle_Armored_C') for i in range(6)]
    for car in cars:
        car.raw['interaction'] = ['0']
    ready.begin(state([*units, *cars], objectives=objects), route)
    assert ready.weak
    for car in cars:
        car.raw['interaction'] = ['100']
        car.position[0] = objects[2].position[0] - 15000  # Reinforcement must actually arrive.
    for step in range(1, 4):
        ready.begin(state([*units, *cars], objectives=objects, step=step), route)
    assert not ready.weak


def test_readiness_never_creates_an_opening_whole_army_assembly_gate():
    units, objects, route, ready, urban = readiness_case(stage=0)
    for step in range(20):
        ready.begin(state(units, objectives=objects, step=step), route)
        assert not ready.weak


def test_preserved_unit_can_test_new_local_contact_and_continue_effective_fire():
    units, objects, route, ready, urban = readiness_case()
    ready.begin(state(units, objectives=objects), route)
    enemy = entity(90, 1, 0, units[0].position[0] + 4000, units[0].position[1])
    units[0].raw['agentPerception'] = [90]
    units[0].position[0] += 3000
    for step in (1, 2):
        s = state(units, [enemy], objects, step)
        ready.begin(s, route)
        assert ready.weak and ready.choose(s, units[0], route, urban, [enemy]) is None
    s = state(units, [enemy], objects, 3)
    ready.begin(s, route)
    assert ready.choose(s, units[0], route, urban, [enemy]).command == 'Guard'
    urban.begin(s)
    urban.record(s, units[0], Action.attack(enemy))
    units[0].raw['interaction'] = ['99']
    enemy.hp = 90
    s = state(units, [enemy], objects, 4)
    urban.begin(s)
    ready.begin(s, route)
    assert ready.choose(s, units[0], route, urban, [enemy]) is None


def test_first_assault_collapse_scans_locally_without_retreating_or_pulling_guards_out():
    units, objects, route, ready, urban = readiness_case(stage=0)
    units += [entity(10+i, 0, i+4, objects[0].position[0]-27000, objects[0].position[1]) for i in range(8)]
    ready.begin(state(units, objectives=objects), route)
    fallback = tuple(units[0].position)
    for unit in units:
        unit.position[0] += 8000
    for unit in units[4:]:
        unit.alive = False
    s = state(units, objectives=objects, step=1)
    ready.begin(s, route)
    action = ready.choose(s, units[0], route, urban, [])
    assert ready.weak and action.command == 'Guard'
    assert action.points == tuple(units[0].position) and action.points != fallback
    route._garrison[units[0].uid] = objects[0].uid
    assert ready.choose(s, units[0], route, urban, []) is None
