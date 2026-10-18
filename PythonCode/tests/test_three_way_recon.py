"""Three simultaneous public surveys, independent from the ground column."""
import math

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.UAVControl import UAVControl
from test_air_and_departure import aircraft
from test_fslutk_rule import entity, map_objectives, state


def recon_team(count=6):
    objects = map_objectives()
    units = [aircraft(10 + i, x=70000, y=12000, z=0) for i in range(count)]
    soldiers = [entity(100 + i, 0, i, 70000, 12000) for i in range(8)]
    route, control = AssaultRoute(adaptive=True), UAVControl()
    current = state([*units, *soldiers], objectives=objects)
    route.update(current)
    return objects, units, soldiers, route, control, current


def at_stations(objects, units, control):
    for unit in units:
        obj = next(obj for obj in objects if obj.uid == control._split_objectives[unit.uid])
        unit.position[:] = (obj.position[0] + 1000, obj.position[1], 6000)


def test_opening_flights_split_equally_before_any_ground_troop_leaves_spawn():
    objects, units, soldiers, route, control, current = recon_team()
    actions = [control.choose(current, unit, objects[0], route) for unit in units]
    assert all(action.command == 'Moving' for action in actions)
    assert [control._split_objectives[unit.uid] for unit in units] == [901, 903, 902, 901, 903, 902]
    assert set(control._launched) == {unit.uid for unit in units}
    for unit, action in zip(units, actions):
        assigned = current.key_object_by_uid(control._split_objectives[unit.uid])
        assert math.dist(action.points, assigned.position) < unit.properties['perceptionRange']
        assert not control.recon_status(current, assigned)['ready']


def test_ground_target_switch_and_roster_reordering_do_not_merge_scout_groups():
    objects, units, soldiers, route, control, current = recon_team()
    before = [control.choose(current, unit, objects[0], route) for unit in units]
    assignment = dict(control._split_objectives)
    route._reorder(current, (objects[2].uid, objects[1].uid, objects[0].uid))
    next_state = state([*reversed(units), *soldiers], objectives=objects, step=1)
    after = [control.choose(next_state, unit, objects[2], route) for unit in units]
    assert before == after and control._split_objectives == assignment


def test_flight_does_not_stop_at_legacy_arrival_radius_outside_scouting_range():
    objects, units, soldiers, route, control, current = recon_team(3)
    unit = units[0]
    first = control.choose(current, unit, objects[0], route)
    dx = first.points[0] - objects[0].position[0]
    dy = first.points[1] - objects[0].position[1]
    length = math.hypot(dx, dy)
    unit.position[:] = (first.points[0] + dx / length * 2700,
                        first.points[1] + dy / length * 2700, first.points[2])
    assert math.dist(unit.position, objects[0].position) > unit.properties['perceptionRange']
    current = state(units, objectives=objects, step=1)
    assert control.choose(current, unit, objects[0], route).command == 'Moving'


def test_survey_counts_frames_once_requires_arrival_and_keeps_objectives_separate():
    objects, units, soldiers, route, control, current = recon_team()
    control.prepare_recon(current, route)
    for step in range(1, 6):
        control.prepare_recon(state([*units, *soldiers], objectives=objects, step=step), route)
    assert not any(control.recon_status(current, obj)['ready'] for obj in objects)
    at_stations(objects, units, control)
    objects[0].raw['blueteamNum'] = 12
    objects[1].raw['blueteamNum'] = 2
    objects[2].raw['blueteamNum'] = 3
    for step in range(6, 9):
        current = state([*units, *soldiers], objectives=objects, step=step)
        control.prepare_recon(current, route)
        for unit in units:
            control.choose(current, unit, objects[0], route)
        for obj in objects:
            report = control.recon_status(current, obj)
            assert report['observations'] == step - 5
            assert report['ready'] is (step == 8)
    assert [control.recon_status(current, obj)['enemy_count'] for obj in objects] == [12, 2, 3]


def test_missing_local_list_unknown_radius_or_unknown_count_does_not_prove_scouting():
    objects, units, soldiers, route, control, current = recon_team(3)
    control.prepare_recon(current, route)
    at_stations(objects, units, control)
    units[0].raw.pop('agentPerception')
    units[1].properties.pop('perceptionRange')
    units[1].raw.pop('perceptionRange')
    objects[1].raw.pop('blueteamNum')
    for step in range(1, 8):
        current = state(units, objectives=objects, step=step)
        control.prepare_recon(current, route)
    assert not any(control.recon_status(current, obj)['ready'] for obj in objects)
    assert not control._surveys


def test_survey_gap_restarts_confirmation_and_old_evidence_expires():
    objects, units, soldiers, route, control, current = recon_team(3)
    control.prepare_recon(current, route)
    at_stations(objects, units, control)
    for step in (1, 2, 6, 7):
        current = state(units, objectives=objects, step=step)
        control.prepare_recon(current, route)
        assert not control.recon_status(current, objects[0])['ready']
    current = state(units, objectives=objects, step=8)
    control.prepare_recon(current, route)
    assert control.recon_status(current, objects[0])['ready']
    for unit in units:
        unit.raw['rSVD1'] = 'uav;0'
    current = state(units, objectives=objects, step=8 + control.SURVEY_TTL + 1)
    control.prepare_recon(current, route)
    assert not control.recon_status(current, objects[0])['ready']


def test_lost_scout_is_replaced_without_reshuffling_the_other_groups():
    objects, units, soldiers, route, control, current = recon_team()
    control.prepare_recon(current, route)
    before = dict(control._split_objectives)
    units[0].alive = units[3].alive = False
    control.prepare_recon(state(units, objectives=objects, step=1), route)
    changed = [unit for unit in units if unit.alive and before[unit.uid] != control._split_objectives[unit.uid]]
    assert len(changed) == 1
    assert control._split_objectives[changed[0].uid] == objects[0].uid
    assert {control._split_objectives[unit.uid] for unit in units if unit.alive} == {obj.uid for obj in objects}


def test_initial_groups_scout_instead_of_diving_on_a_transit_contact():
    objects, units, soldiers, route, control, current = recon_team()
    enemy = entity(999, 1, 0, 70000, 12000)
    for unit in units:
        unit.raw['agentPerception'] = [enemy.uid]
    for step in range(5):
        current = state([*units, *soldiers], [enemy], objects, step=step)
        actions = [control.choose(current, unit, objects[0], route) for unit in units]
        assert all(action.command not in {'NormalAttacking', 'SelfDestruct'} for action in actions)


def test_survey_keeps_one_observer_per_objective_and_records_only_local_contacts():
    objects, units, soldiers, route, control, current = recon_team()
    control.prepare_recon(current, route)
    at_stations(objects, units, control)
    enemy = entity(999, 1, 0, *objects[0].position[:2])
    for unit in (units[0], units[3]):
        unit.raw['agentPerception'] = [enemy.uid]
    for step in range(1, 4):
        current = state(units, [enemy], objects, step=step)
        control.prepare_recon(current, route)
    for unit in units[:3]:
        action = control.choose(current, unit, objects[0], route)
        assert action.command not in {'NormalAttacking', 'SelfDestruct'}
    assert control.recon_status(current, objects[0])['contacts'] == (enemy.uid,)
    assert control.recon_status(current, objects[1])['contacts'] == ()
    assert control.recon_status(current, objects[2])['contacts'] == ()


def test_new_episode_discards_previous_survey_and_reassigns_current_roster():
    objects, units, soldiers, route, control, current = recon_team()
    control.prepare_recon(current, route)
    at_stations(objects, units, control)
    for step in range(1, 4):
        control.prepare_recon(state(units, objectives=objects, step=step), route)
    next_state = state(units, objectives=objects, step=0, episode=2)
    assert not control.recon_status(next_state, objects[0])['ready']
    route.update(next_state)
    control.prepare_recon(next_state, route)
    assert control.recon_status(next_state, objects[0])['observations'] == 1
    assert not control.recon_status(next_state, objects[0])['ready']
