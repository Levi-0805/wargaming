"""Occupied B must retain usable support without adding an unbounded wait."""
import pytest

from cssim.protocol import Action
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultFormation import AssaultFormation
from test_fslutk_rule import entity, state
from test_rear_cover import captured_case, frame


def contested():
    foot, cars, objects, route, cover = captured_case()
    objects[1].raw['blueteamNum'] = 7
    return foot, cars, objects, route, cover


def test_no_blind_rear_scan_splits_b_assault_even_with_spare_cars():
    foot, cars, objects, route, cover = contested()
    for step in range(4, 40):
        s = frame(foot, cars, objects, step)
        assert all(cover.choose_vehicle(s, u, route) is None for u in cars)


def test_rear_damage_can_use_a_spare_wheel_but_keeps_two_armed_supporters():
    foot, cars, objects, route, cover = contested()
    cover.begin(frame(foot, cars, objects), route)
    guard = next(u for u in foot if u.uid in route._garrison)
    guard.hp -= 10
    s = frame(foot, cars, objects, 5)
    actions = [cover.choose_vehicle(s, u, route) for u in cars]
    selected = [u for u, a in zip(cars, actions) if a is not None]
    assert len(selected) == 1 and selected[0].entity_type == 'BP_MNWS_Vehicle_Armored_C'
    # A new casualty must release the rear car immediately, not at its deadline.
    cars[0].alive = False
    s = frame(foot, cars, objects, 6)
    assert all(cover.choose_vehicle(s, u, route) is None for u in cars)


@pytest.mark.parametrize('invalid', ['empty', 'critical', 'offline', 'dead'])
def test_spare_count_does_not_treat_unusable_car_as_assault_support(invalid):
    foot, cars, objects, route, cover = contested()
    cover.begin(frame(foot, cars, objects), route)
    if invalid == 'empty':
        cars[0].raw['interaction'] = ['0']
    elif invalid == 'critical':
        cars[0].hp = 1.611
    elif invalid == 'offline':
        cars[0].raw['rSVD1'] = 'vehicle;0'
    else:
        cars[0].alive = False
    next(u for u in foot if u.uid in route._garrison).hp -= 20
    s = frame(foot, cars, objects, 5)
    # A critically damaged car may itself cover, but both usable cars advance.
    assert all(cover.choose_vehicle(s, u, route) is None for u in cars[1:])


def test_local_rear_contact_is_observed_but_shared_report_alone_does_not_hold():
    foot, cars, objects, route, cover = contested()
    guard = next(u for u in foot if u.uid in route._garrison)
    enemy = entity(999, 1, 0, guard.position[0] + 7000, guard.position[1],
                   kind='BP_MNWS_Vehicle_6x6UGV_C')
    s = state(foot + cars, [enemy], objects, step=4)
    assert all(cover.choose_vehicle(s, u, route) is None for u in cars)
    guard.raw['agentPerception'] = [enemy.uid]
    s = state(foot + cars, [enemy], objects, step=5)
    assert sum(cover.choose_vehicle(s, u, route) is not None for u in cars) == 1
    deadline = cover._deadline
    guard.raw['agentPerception'] = []
    s = state(foot + cars, [enemy], objects, step=10)
    assert all(cover.choose_vehicle(s, u, route) is None for u in cars)
    assert cover._deadline == deadline


def formation_case():
    foot, cars, objects, route, _ = contested()
    base = objects[1]
    base.raw.update(scale={'x': 10500, 'y': 9500}, rotation={'yaw': -12.8125})
    mobile = entity(500, 0, 10, base.position[0] + 40000, base.position[1])
    for i, car in enumerate(cars):
        car.position[:2] = base.position[:2] + [1000 + i*600, 0]
    form = AssaultFormation()
    return foot + [mobile], cars, objects, route, form, mobile


def test_b_wait_begins_with_capture_wave_and_expires_once_without_recalling_inside_units():
    foot, cars, objects, route, form, mobile = formation_case()
    for step in (4, 20, 40):
        form.begin(state(foot + cars, objectives=objects, step=step), route)
    gate = form._gates[objects[1].uid]
    assert gate['start'] is None and not gate['released']
    mobile.position[0] = objects[1].position[0] + 17500
    s = state(foot + cars, objectives=objects, step=41)
    form.begin(s, route)
    assert gate['start'] == 41 and form.choose(s, mobile, route).command == 'Guard'
    # A unit already in B and one taking damage are never recalled to wait.
    mobile.position[:] = objects[1].position
    assert form.choose(s, mobile, route) is None
    mobile.position[0] += 17500
    mobile.hp -= 10
    s = state(foot + cars, objectives=objects, step=42)
    form.begin(s, route)
    assert form.choose(s, mobile, route) is None
    s = state(foot + cars, objectives=objects, step=69)
    form.begin(s, route)
    assert gate['released'] and gate['start'] == 41
    assert form.choose(s, mobile, route) is None


@pytest.mark.parametrize('invalid', ['dead', 'spent_only', 'moved_away', 'expired'])
def test_b_handoff_does_not_reuse_dead_or_ineffective_or_distant_volley(invalid):
    foot, cars, objects, route, form, mobile = formation_case()
    base = objects[1]
    mobile.position[0] = base.position[0] + 17500
    s = state(foot + cars, objectives=objects, step=4)
    form.begin(s, route)
    active = form._active[:2]
    enemies = [entity(900+i, 1, i, base.position[0]+1000, base.position[1]) for i in range(2)]
    for unit, enemy in zip(active, enemies):
        unit.raw['agentPerception'] = [enemy.uid]
        form.record(state(foot + cars, enemies, objects, step=4), unit, Action.attack(enemy), route)
        unit.raw['interaction'] = ['99']
        if invalid != 'spent_only':
            enemy.hp -= 10
    if invalid == 'dead':
        active[0].alive = False
    elif invalid == 'moved_away':
        active[0].position[0] += 40000
    step = 12 if invalid == 'expired' else 5
    form.begin(state(foot + cars, enemies, objects, step=step), route)
    assert not form._gates[base.uid]['released']


def test_two_present_cars_with_recent_observed_damage_release_before_timeout():
    foot, cars, objects, route, form, mobile = formation_case()
    base = objects[1]
    mobile.position[0] = base.position[0] + 17500
    form.begin(state(foot + cars, objectives=objects, step=4), route)
    enemy = entity(999, 1, 0, *base.position[:2])
    for unit in form._active[:2]:
        unit.raw['agentPerception'] = [999]
        form.record(state(foot + cars, [enemy], objects, step=4), unit, Action.attack(enemy), route)
        unit.raw['interaction'] = ['99']
    enemy.hp -= 20
    s = state(foot + cars, [enemy], objects, step=5)
    form.begin(s, route)
    assert form._gates[base.uid]['released'] and form.choose(s, mobile, route) is None


def test_no_armed_support_does_not_create_an_endless_b_formation_wait():
    foot, cars, objects, route, form, mobile = formation_case()
    for car in cars:
        car.raw['interaction'] = ['0']
    mobile.position[0] = objects[1].position[0] + 17500
    s = state(foot + cars, objectives=objects, step=4)
    form.begin(s, route)
    assert form.choose(s, mobile, route) is None
