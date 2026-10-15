"""Observed failures in the 20261015_113304 match, without hidden enemy state."""
import pytest

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.UAVControl import UAVControl
from test_air_and_departure import aircraft
from test_fslutk_rule import entity, state, algorithm, map_objectives, guards_at


def lock_case():
    uav = aircraft()
    enemy = entity(20, 1, 0, 1000)
    uav.raw.update(agentPerception=[20], interaction=['1'])
    return uav, enemy, UAVControl()


def test_moving_target_never_creates_fixed_coordinate_dive():
    uav, enemy, control = lock_case()
    for step in range(20):
        enemy.position[0] = 1000 + step * 1500  # Recorded vehicle movement.
        action = control.choose(state([uav], [enemy], step=step), uav, None)
        assert action.command != 'SelfDestruct'
        if action.command == 'NormalAttacking':
            assert action.target_uid == enemy.uid


@pytest.mark.parametrize('change', ['moved', 'lost', 'dead', 'offline', 'friendly'])
def test_invalidated_lock_never_repeats_old_dive_point(change):
    uav, enemy, control = lock_case()
    units = [uav]
    for step in range(3):
        action = control.choose(state(units, [enemy], step=step), uav, None)
    assert action.command == 'SelfDestruct'
    if change == 'moved':
        enemy.position[0] += 1500
    elif change == 'lost':
        uav.raw['agentPerception'] = []
    elif change == 'dead':
        enemy.alive = False
    elif change == 'offline':
        uav.raw['rSVD1'] = 'uav;0'
    else:
        units.append(entity(30, 0, 1, 1000))
    for step in (3, 4):
        action = control.choose(state(units, [enemy], step=step), uav, None)
        assert action.command != 'SelfDestruct'
    assert not control._missions


def test_small_position_update_uses_current_point_not_launch_snapshot():
    uav, enemy, control = lock_case()
    for step in range(3):
        control.choose(state([uav], [enemy], step=step), uav, None)
    enemy.position[0] += 100
    action = control.choose(state([uav], [enemy], step=3), uav, None)
    assert action.command == 'SelfDestruct' and action.points == (1100, 0, 0)


def test_same_target_cannot_be_reserved_again_when_it_moves_out_of_old_cluster():
    uav, enemy, control = lock_case()
    second = aircraft(2)
    second.raw['agentPerception'] = [20]
    units = [uav, second]
    for step in range(3):
        control.choose(state(units, [enemy], step=step), uav, None)
        second_action = control.choose(state(units, [enemy], step=step), second, None)
    assert second_action.command != 'SelfDestruct'
    enemy.position[0] += 1500
    # The second UAV executes before the first can invalidate its mission.
    second_action = control.choose(state(units, [enemy], step=3), second, None)
    assert second_action.command != 'SelfDestruct'
    assert all(uid != second.uid for uid in control._missions)


@pytest.mark.parametrize('case', ['one_frame_repeated', 'gap', 'too_high', 'too_far', 'empty', 'friendly'])
def test_unsafe_or_unconfirmed_dive_is_rejected(case):
    uav, enemy, control = lock_case()
    units = [uav]
    steps = (0, 1, 2)
    if case == 'one_frame_repeated':
        steps = (0, 0, 0)
    elif case == 'gap':
        steps = (0, 2, 4)
    elif case == 'too_high':
        uav.position[2] = 15000
    elif case == 'too_far':
        enemy.position[0] = 10000
    elif case == 'empty':
        uav.raw['interaction'] = ['0']
    else:
        units.append(entity(30, 0, 1, 1000))
    for step in steps:
        action = control.choose(state(units, [enemy], step=step), uav, None)
        assert action.command != 'SelfDestruct'
        if case in ('empty', 'friendly'):
            assert action.command != 'NormalAttacking'


def test_opening_launch_uses_small_group_then_latches_without_recalling_uavs():
    obj = map_objectives()[0]
    ground = [entity(30+i, 0, i, obj.position[0]+100000, obj.position[1]) for i in range(6)]
    uavs = [aircraft(i+1, x=60000, y=35000) for i in range(5)]
    units = [*ground, *uavs]
    control = UAVControl()
    assert all(control.choose(state(units, objectives=[obj]), u, obj).command == 'Guard' for u in uavs)
    ground[0].position[0] = obj.position[0] + 10000  # One isolated scout is insufficient.
    assert control.choose(state(units, objectives=[obj], step=1), uavs[0], obj).command == 'Guard'
    for g in ground[:5]:
        g.position[0] = obj.position[0] + 70000
    assert control.choose(state(units, objectives=[obj], step=2), uavs[0], obj).command == 'Moving'
    assert control.choose(state(units, objectives=[obj], step=2), uavs[1], obj).command == 'Guard'
    for g in ground[:5]:
        g.position[0] = obj.position[0] + 55000
    for step in (3, 4, 5):
        for u in uavs:
            action = control.choose(state(units, objectives=[obj], step=step), u, obj)
            if step == 5:  # The scout transitions once to the next escort station.
                assert action.command == 'Moving'
    points = {u.uid: control._stations[u.uid][1] for u in uavs}
    for g in ground:
        g.position[0] = obj.position[0] + 100000
    assert all(control.choose(state(units, objectives=[obj], step=6), u, obj).command == 'Moving' for u in uavs)
    assert {u.uid: control._stations[u.uid][1] for u in uavs} == points


def test_opening_wait_cannot_stall_aircraft_for_entire_game_or_delay_ground():
    obj = map_objectives()[0]
    soldiers = [entity(30+i, 0, i, 60000, 35000) for i in range(6)]
    uav = aircraft(x=60000, y=35000)
    rule = algorithm([*soldiers, uav])
    actions = rule.act(state([*soldiers, uav], objectives=[obj]))
    assert all(a.command == 'Moving' for a in actions[:-1])
    assert actions[-1].command == 'Guard'
    for step in range(1, 51):
        action = rule._uav.choose(state([*soldiers, uav], objectives=[obj], step=step), uav, obj)
    assert action.command == 'Moving'


def test_local_threat_overrides_opening_wait_and_normal_attack_does_not_loop_forever():
    uav, enemy, control = lock_case()
    uav.position[2] = 15000  # Not a close dive opportunity.
    soldiers = [entity(30+i, 0, i, 60000, 35000) for i in range(6)]
    obj = map_objectives()[0]
    actions = [control.choose(state([uav, *soldiers], [enemy], [obj], step), uav, obj) for step in range(8)]
    assert all(a.command == 'NormalAttacking' for a in actions[:3])
    assert all(a.command != 'NormalAttacking' for a in actions[3:7])
    assert actions[7].command == 'NormalAttacking'


@pytest.mark.parametrize('kind', ['BP_MNWS_Vehicle_6x6UGV_C', 'BP_MNWS_Vehicle_Armored_C'])
def test_vehicle_engages_visible_enemy_vehicle_before_infantry_but_never_hidden_armor(kind):
    car = entity(1, 0, 0, 0, kind=kind)
    soldier = entity(20, 1, 0, 500)
    lynx = entity(21, 1, 1, 4000, kind='BP_MNWS_Vehicle_6x6UGV_C')
    car.raw['agentPerception'] = [20, 21]
    assert algorithm([car]).act(state([car], [soldier, lynx]))[0].target_uid == lynx.uid
    car.raw['agentPerception'] = [20]
    assert algorithm([car]).act(state([car], [soldier, lynx]))[0].target_uid == soldier.uid


def test_vehicle_burst_without_damage_repositions_early_and_switches_target():
    car = entity(1, 0, 0, 0, kind='BP_MNWS_Vehicle_Armored_C')
    blocked = entity(20, 1, 0, 4000, kind='BP_MNWS_Vehicle_6x6UGV_C')
    reachable = entity(21, 1, 1, 5000)
    car.raw['agentPerception'] = [20, 21]
    rule = algorithm([car])
    actions = []
    for step in range(4):
        car.raw['interaction'] = [str(200 - 15 * step)]
        actions.append(rule.act(state([car], [blocked, reachable], step=step))[0])
    assert [a.target_uid for a in actions[:2]] == [20, 20]
    assert actions[2].command == 'Moving'
    assert actions[3].target_uid == reachable.uid


def test_damaging_vehicle_fire_is_not_interrupted_by_short_trial_budget():
    car = entity(1, 0, 0, 0, kind='BP_MNWS_Vehicle_Armored_C')
    enemy = entity(20, 1, 0, 4000)
    car.raw['agentPerception'] = [20]
    rule = algorithm([car])
    for step in range(6):
        car.raw['interaction'] = [str(200 - 15 * step)]
        enemy.hp = 100 - step * 10
        assert rule.act(state([car], [enemy], step=step))[0].command == 'NormalAttacking'


def test_departing_vehicle_still_engages_locally_seen_exit_threat():
    objects = map_objectives()
    objects[0].raw.update(percent=1, redTeamNum=4)
    guards = guards_at(objects[0])
    car = entity(1, 0, 0, -37782, 24389, kind='BP_MNWS_Vehicle_Armored_C')
    enemy = entity(9000, 1, 0, -41929, 33078, kind='BP_MNWS_Vehicle_6x6UGV_C')
    units = [car, *guards]
    rule = algorithm(units)
    for step in range(4):
        rule.act(state(units, objectives=objects, step=step))
    assert rule._route.stage == 1
    car.raw.update(agentPerception=[enemy.uid], interaction=['200'])
    action = rule.act(state(units, [enemy], objects, 4))[0]
    assert action.command == 'NormalAttacking' and action.target_uid == enemy.uid
    car.raw['agentPerception'] = []
    assert rule.act(state(units, [enemy], objects, 5))[0].command != 'NormalAttacking'
