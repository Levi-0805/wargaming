"""Behaviour checks for the staggered arrivals and early UAV losses at 15:00."""
import pytest

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.ApproachCohesion import ApproachCohesion
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.UAVControl import UAVControl
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.FlightNavigation import FlightNavigation
from test_fslutk_rule import entity, state, algorithm, map_objectives
from test_air_and_departure import aircraft


def convoy():
    objs = map_objectives()
    x, y = objs[0].position[:2]
    units = [entity(100+i, 0, i, x + (30000 if i < 3 else 55000), y) for i in range(10)]
    units += [entity(200+i, 0, 10+i, x+50000, y, kind='BP_MNWS_Vehicle_Armored_C') for i in range(2)]
    return units, objs


def test_early_arrivals_wait_locally_but_rear_keeps_moving():
    units, objs = convoy()
    rule = algorithm(units)
    actions = rule.act(state(units, objectives=objs))
    assert all(a.command == 'Guard' and a.points == tuple(u.position) for u, a in zip(units[:3], actions[:3]))
    assert all(a.command == 'Moving' for a in actions[3:])


def test_main_body_and_two_vehicles_release_once_without_waiting_for_everyone():
    units, objs = convoy()
    gate, route = ApproachCohesion(), AssaultRoute()
    for step in range(3):
        if step == 1:
            for u in [*units[3:7], *units[-2:]]:
                u.position[0] = objs[0].position[0] + 34000
        if step == 2:
            for u in units[3:]:
                u.position[0] += 50000
        s = state(units, objectives=objs, step=step)
        route.update(s)
        gate.begin(s, route)
        assert (gate.choose(s, units[0], route) is None) == (step > 0)
    assert gate.reason == 'main_body_ready'


def test_stuck_rear_cannot_create_endless_or_repeated_wait():
    units, objs = convoy()
    gate, route = ApproachCohesion(), AssaultRoute()
    for step in range(120):
        s = state(units, objectives=objs, step=step)
        route.update(s)
        gate.begin(s, route)
        gate.begin(s, route)  # Duplicate calls must not change the deadline.
        a = gate.choose(s, units[0], route)
        assert (a is not None) == (step < 28)
    assert gate.reason == 'deadline'


@pytest.mark.parametrize('trigger', ['local', 'damage', 'captured', 'inside'])
def test_assembly_does_not_block_local_combat_or_recall_captured_units(trigger):
    units, objs = convoy()
    rule = algorithm(units)
    rule.act(state(units, objectives=objs))
    enemy = entity(900, 1, 0, units[0].position[0]-4000, units[0].position[1])
    if trigger == 'local':
        units[0].raw['agentPerception'] = [enemy.uid]
    elif trigger == 'damage':
        units[0].hp -= 10
    elif trigger == 'captured':
        rule._route.stage = 1
    else:
        units[0].position[:] = objs[0].position
    s = state(units, [enemy] if trigger == 'local' else [], objs, 1)
    rule._prepare(s)
    assert rule._cohesion.choose(s, units[0], rule._route) is None
    if trigger == 'local':
        assert rule.act(s)[0].command == 'NormalAttacking'


def test_episode_reset_restores_single_rendezvous():
    units, objs = convoy()
    rule = algorithm(units)
    rule.act(state(units, objectives=objs))
    rule.act(state(units, objectives=objs, step=40))
    assert rule._cohesion.released
    assert rule.act(state(units, objectives=objs, episode=2))[0].command == 'Guard'
    assert not rule._cohesion.released


def escort_case():
    obj = map_objectives()[0]
    x, y = obj.position[:2]
    soldiers = [entity(100+i, 0, i, x+60000, y) for i in range(10)]
    uavs = [aircraft(i+1, x=x+100000, y=y) for i in range(5)]
    return obj, soldiers, uavs, UAVControl()


def test_50_step_timeout_never_sends_wing_directly_to_unsupported_hq():
    obj, soldiers, uavs, control = escort_case()
    for g in soldiers:
        g.position[0] += 40000
    for step in (0, 50):
        for u in uavs:
            action = control.choose(state([*soldiers, *uavs], objectives=[obj], step=step), u, obj)
            if action.command == 'Moving':
                assert AssaultRoute.distance(action.points, obj.position) >= 94000
            assert action.command != 'SelfDestruct'


def test_isolated_front_soldier_cannot_pull_uavs_into_hq():
    obj, soldiers, uavs, control = escort_case()
    soldiers[0].position[:] = obj.position
    s = state([*soldiers, *uavs], objectives=[obj], step=50)
    for u in uavs:
        action = control.choose(s, u, obj)
        assert action.command == 'Moving'
        assert AssaultRoute.distance(action.points, obj.position) >= 54000


def test_station_stays_fixed_through_reports_casualties_and_small_ground_motion():
    obj, soldiers, uavs, control = escort_case()
    uav = uavs[0]
    initial = None
    for step in range(8):
        enemy = entity(900+step, 1, 0, obj.position[0]+(-1)**step * 10000, obj.position[1])
        if step == 3:
            soldiers[0].alive = False
        for u in soldiers:
            u.position[0] += 200  # Small retreat must not recall aircraft.
        control.choose(state([*soldiers, *uavs], [enemy], [obj], step), uav, obj)
        point = control._stations[uav.uid][1]
        initial = point if initial is None else initial
        assert point == initial


def test_main_body_advances_station_and_capture_changes_objective_without_old_patrol():
    obj, soldiers, uavs, control = escort_case()
    uav = uavs[0]
    control.choose(state([*soldiers, *uavs], objectives=[obj]), uav, obj)
    old = control._stations[uav.uid]
    for g in soldiers:
        g.position[0] -= 10000
    control.choose(state([*soldiers, *uavs], objectives=[obj], step=1), uav, obj)
    new = control._stations[uav.uid]
    assert new[0] < old[0] and new[2] == old[2] + 1
    second = map_objectives()[1]
    for step in (2, 3, 4):
        action = control.choose(state([*soldiers, *uavs], objectives=[second], step=step), uav, second)
    assert action.command == 'Moving' and action.points == control._stations[uav.uid][1]
    assert control._objective_uid == second.uid


def test_stable_local_lock_requires_ground_support_before_self_destruct():
    obj, soldiers, uavs, control = escort_case()
    uav = uavs[0]
    uav.position[:] = (obj.position[0]+1000, obj.position[1], 8000)
    enemy = entity(900, 1, 0, *obj.position[:2])
    uav.raw.update(agentPerception=[900], interaction=['1'])
    for step in range(3):
        action = control.choose(state([*soldiers, uav], [enemy], [obj], step), uav, obj)
        assert action.command != 'SelfDestruct'
    for g in soldiers[:4]:
        g.position[0] = obj.position[0] + 15000  # Outside friendly clearance.
    assert control.choose(state([*soldiers, uav], [enemy], [obj], 3), uav, obj).command != 'SelfDestruct'
    for g in soldiers[4:6]:
        g.position[0] = obj.position[0] + 15000
    assert control.choose(state([*soldiers, uav], [enemy], [obj], 4), uav, obj).command == 'SelfDestruct'


def test_empty_payload_observes_fixed_station_instead_of_chasing_contacts():
    obj, soldiers, uavs, control = escort_case()
    uav = uavs[0]
    uav.raw['interaction'] = ['0']
    for step in range(5):
        enemy = entity(900, 1, 0, obj.position[0]+step*1000, obj.position[1])
        uav.raw['agentPerception'] = [900]
        action = control.choose(state([*soldiers, uavs[0]], [enemy], [obj], step), uav, obj)
        assert action.command not in {'SelfDestruct', 'NormalAttacking'}
        assert action.points == control._stations[uav.uid][1]


def test_local_target_does_not_switch_on_each_frame_score_fluctuation():
    uav = aircraft()
    a, b = entity(900, 1, 0, 10000), entity(901, 1, 0, 11000)
    uav.raw['agentPerception'] = [900, 901]
    control = UAVControl()
    for step in range(3):
        a.position[0] = 10000 if step == 0 else 12000
        action = control.choose(state([uav], [a,b], step=step), uav, None)
        assert action.target_uid == a.uid


def test_unreachable_station_has_finite_retries_and_next_station_releases_it():
    uav, nav = aircraft(), FlightNavigation()
    for step in range(100):
        a = nav.apply(state([uav], step=step), uav, (20000, 0, 8000), (1, 'escort', 0), max_stall_resets=2)
        if step >= 28:
            assert a.command == 'Guard' and a.points == tuple(uav.position)
    assert nav.resets[uav.uid] == 2
    a = nav.apply(state([uav], step=100), uav, (30000, 0, 8000), (1, 'escort', 1), max_stall_resets=2)
    assert a.command == 'Moving' and a.points == (30000, 0, 8000)
