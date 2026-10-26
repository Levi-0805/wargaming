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


def join_case():
    foot, cars, objects, route, form, mobile = formation_case()
    base = objects[1]
    mobile.position[:2] = base.position[:2] + [45000, 0]
    group = [mobile] + [entity(501+i, 0, 11+i, base.position[0]+45000, base.position[1]+i*100)
                        for i in range(3)]
    foot += group[1:]
    for i, car in enumerate(cars):
        car.position[:2] = base.position[:2] + [25000+i*600, 0]
    return foot, cars, objects, route, form, group


def test_recorded_b_edge_failure_no_longer_parks_infantry_at_180_metres():
    from test_fslutk_rule import algorithm
    foot, cars, objects, route, form, mobile = formation_case()
    mobile.position[0] = objects[1].position[0] + 17500
    rule = algorithm(foot + cars)
    rule._route, rule._formation = route, form
    rule._cohesion.released = True
    for step in range(4, 32):
        s = state(foot + cars, objectives=objects, step=step)
        form.begin(s, route)
        assert form.choose(s, mobile, route) is None
        assert rule._combat_action(s, mobile).command == 'Moving'


def test_join_holds_lead_only_and_releases_for_group_not_single_straggler():
    foot, cars, objects, route, form, group = join_case()
    s = state(foot + cars, objectives=objects, step=4)
    form.begin(s, route)
    lead = form._active[0]
    assert form.choose(s, lead, route).points == tuple(lead.position)
    assert all(form.choose(s, f, route) is None for f in group)
    group[0].position[:2] = lead.position[:2] + [6000, 0]
    s = state(foot + cars, objectives=objects, step=5)
    form.begin(s, route)
    assert form.choose(s, lead, route).command == 'Guard'
    for f in group[1:]:
        f.position[:2] = lead.position[:2] + [6000, 0]
    s = state(foot + cars, objectives=objects, step=6)
    form.begin(s, route)
    assert all(form.choose(s, car, route) is None for car in cars)
    for f in group:
        f.position[0] += 20000
    s = state(foot + cars, objectives=objects, step=7)
    form.begin(s, route)
    assert form.choose(s, lead, route) is None


def test_join_deadline_does_not_restart_for_casualty_or_late_car():
    foot, cars, objects, route, form, group = join_case()
    form.begin(state(foot + cars, objectives=objects, step=4), route)
    gate = form._gates[objects[1].uid]
    assert gate['start'] == 4
    form._active[0].alive = False
    for step in (10, 21, 22, 45):
        s = state(foot + cars, objectives=objects, step=step)
        form.begin(s, route)
        assert gate['start'] == 4
        assert all(form.choose(s, f, route) is None for f in group)
        if step >= 22:
            assert all(form.choose(s, car, route) is None for car in cars)


def test_initial_cohesion_does_not_disable_later_join_or_let_rear_car_release_lead():
    foot, cars, objects, route, form, group = join_case()
    for f in group:
        f.position[:2] = cars[0].position[:2] + [7000, 0]
    form.begin(state(foot + cars, objectives=objects, step=4), route)
    gate = form._gates[objects[1].uid]
    assert gate['start'] is None and not gate['released']
    lead = min(form._active, key=lambda u: route.distance(u.position, objects[1].position))
    lead.position[0] -= 4000  # Faster car opens a gap; rear car stays with foot.
    s = state(foot + cars, objectives=objects, step=5)
    form.begin(s, route)
    assert gate['start'] == 5 and not gate['released']
    assert form.choose(s, lead, route).command == 'Guard'
    s = state(foot + cars, objectives=objects, step=6)
    form.begin(s, route)
    assert not gate['released']


@pytest.mark.parametrize('release', ['damage', 'ground_contact', 'air_contact', 'foot_inside', 'car_inside'])
def test_contact_or_entry_releases_join_and_all_reinforcements(release):
    foot, cars, objects, route, form, group = join_case()
    form.begin(state(foot + cars, objectives=objects, step=4), route)
    lead = form._active[0]
    enemies = []
    if release == 'damage':
        lead.hp -= 10
    elif release.endswith('contact'):
        enemy = entity(999, 1, 0, lead.position[0]+3000, lead.position[1],
                       kind='BP_Base_UAV_C' if release == 'air_contact' else 'BP_BaseSoldier_C')
        lead.raw['agentPerception'] = [enemy.uid]
        enemies = [enemy]
    elif release == 'foot_inside':
        group[0].position[:] = objects[1].position
    else:
        lead.position[:] = objects[1].position
    s = state(foot + cars, enemies, objects, step=5)
    form.begin(s, route)
    assert all(form.choose(s, u, route) is None for u in group + cars)


def test_hq_guards_and_disconnected_units_do_not_replace_following_wave():
    foot, cars, objects, route, form, group = join_case()
    for f in foot:
        if f.uid in route._garrison:
            f.position[:2] = cars[0].position[:2]
    for f in group:
        f.position[:2] = cars[0].position[:2]
        f.raw['rSVD1'] = 'soldier;0'
    s = state(foot + cars, objectives=objects, step=4)
    form.begin(s, route)
    assert form.choose(s, form._active[0], route).command == 'Guard'


def test_no_armed_support_never_holds_infantry():
    foot, cars, objects, route, form, mobile = formation_case()
    for car in cars:
        car.raw['interaction'] = ['0']
    mobile.position[0] = objects[1].position[0] + 17500
    s = state(foot + cars, objectives=objects, step=4)
    form.begin(s, route)
    assert form.choose(s, mobile, route) is None


def test_empty_b_keeps_existing_advance_and_episode_clears_join():
    foot, cars, objects, route, form, group = join_case()
    objects[1].raw['blueteamNum'] = 0
    s = state(foot + cars, objectives=objects, step=4)
    form.begin(s, route)
    assert all(form.choose(s, car, route) is None for car in cars)
    objects[1].raw['blueteamNum'] = 7
    s = state(foot + cars, objectives=objects, step=0, episode=2)
    form.begin(s, route)
    gate = form._gates[objects[1].uid]
    assert gate['start'] == 0 and not gate['released']
