"""Stop-loss decisions for occupied B with a publicly empty alternative A."""
from dataclasses import replace

import pytest

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RemnantCapture import RemnantCapture
from test_fslutk_rule import algorithm, entity, guards_at, map_objectives, state


def battle():
    objects = map_objectives()
    hq, b, a = objects
    hq.raw.update(percent=1, redTeamNum=8, blueteamNum=0)
    b.raw.update(blueteamNum=15, scale={'x': 10500, 'y': 9500}, rotation={'yaw': -12.8125})
    units = guards_at(hq, count=8)
    units[1].entity_type = 'BP_RoboDog_C'
    units += [entity(200+i, 0, 8+i, b.position[0], b.position[1]+25000) for i in range(12)]
    rule = algorithm(units, enable_continuous_column=True)
    for step in range(8):
        rule.act(state(units, objectives=objects, step=step))
    assert rule._route.hq_secured and rule._route.objective(state(units, objectives=objects)).uid == b.uid
    return units, objects, rule


def reduce(units, count):
    for i, unit in enumerate(units):
        unit.alive = i < count
        if not unit.alive:
            unit.hp = 0


def step(rule, units, objects, index=8, enemies=()):
    s = state(units, enemies, objects, step=index)
    return s, rule.act(s)


def test_last_human_and_dog_both_leave_old_hq_reserve_for_a():
    units, objects, rule = battle()
    reduce(units, 2)
    s, actions = step(rule, units, objects)
    route = rule._route
    assert route._remnant.active and route.objective(s).uid == objects[2].uid
    assert not route._garrison
    assert all(a.command == 'Moving' and a.points for a in actions[:2])
    assert all(a.command == 'Idle' for a in actions[2:])
    assert rule._route._remnant.reason == 'last_capturers'


def test_half_force_lost_and_outnumbered_without_cars_triggers_before_last_two():
    units, objects, rule = battle()
    reduce(units, 10)
    s, _ = step(rule, units, objects)
    assert rule._route._remnant.active
    assert rule._route._remnant.reason == 'unsupported_collapse'
    assert rule._route.objective(s).uid == objects[2].uid
    assert list(rule._route._garrison.values()) == [objects[0].uid]


def test_intact_capture_force_is_not_redirected():
    units, objects, rule = battle()
    reduce(units, 12)
    s, _ = step(rule, units, objects)
    assert not rule._route._remnant.active
    assert rule._route.objective(s).uid == objects[1].uid


@pytest.mark.parametrize('ammo', [['20'], [], ['bad'], ['nan'], ['-1']])
def test_usable_or_unknown_vehicle_ammunition_prevents_collapse_assumption(ammo):
    units, objects, rule = battle()
    reduce(units, 2)
    car = entity(999, 0, 25, *objects[0].position[:2], kind='BP_MNWS_Vehicle_6x6UGV_C')
    car.raw['interaction'] = ammo
    units.append(car)
    step(rule, units, objects)
    assert not rule._route._remnant.active


def test_empty_vehicle_does_not_block_last_pair_diversion():
    units, objects, rule = battle()
    reduce(units, 2)
    car = entity(999, 0, 25, *objects[0].position[:2], kind='BP_MNWS_Vehicle_Armored_C')
    car.raw['interaction'] = ['0']
    units.append(car)
    step(rule, units, objects)
    assert rule._route._remnant.active


@pytest.mark.parametrize('value', [None, 'bad', -1, float('nan'), 6])
def test_unknown_or_defended_a_does_not_authorize_diversion(value):
    units, objects, rule = battle()
    reduce(units, 2)
    objects[2].raw['blueteamNum'] = value
    step(rule, units, objects)
    assert not rule._route._remnant.active


def test_recent_a_perimeter_contact_overrides_empty_count():
    units, objects, rule = battle()
    reduce(units, 2)
    enemy = entity(9000, 1, 0, objects[2].position[0] + 11000, objects[2].position[1])
    step(rule, units, objects, enemies=[enemy])
    for i in range(9, 25):
        step(rule, units, objects, i)
        assert not rule._route._remnant.active
    for i in range(25, 28):
        step(rule, units, objects, i)
    assert rule._route._remnant.active


def test_discontinuous_empty_evidence_needs_three_new_frames():
    units, objects, rule = battle()
    reduce(units, 2)
    for index in (20, 21):
        step(rule, units, objects, index)
        assert not rule._route._remnant.active
    step(rule, units, objects, 22)
    assert rule._route._remnant.active


def test_no_hq_first_bypass_even_with_only_two_and_empty_a():
    units, objects, rule = battle()
    reduce(units, 2)
    rule._route.hq_secured = False
    step(rule, units, objects)
    assert not rule._route._remnant.active


def test_empty_b_preserves_existing_attack_order():
    units, objects, rule = battle()
    reduce(units, 2)
    objects[1].raw['blueteamNum'] = 0
    step(rule, units, objects)
    assert not rule._route._remnant.active


def test_changed_a_report_pauses_forward_goal_and_requires_reconfirmation():
    units, objects, rule = battle()
    reduce(units, 2)
    step(rule, units, objects)
    safe = tuple(units[0].position)
    objects[2].raw['blueteamNum'] = 8
    s, actions = step(rule, units, objects, 9)
    assert rule._route.objective(s).uid == objects[2].uid
    assert actions[0].command == 'Guard' and actions[0].points == safe
    objects[2].raw['blueteamNum'] = 0
    for index in (10, 11):
        step(rule, units, objects, index)
        assert not rule._route._remnant.ready
    step(rule, units, objects, 12)
    assert rule._route._remnant.ready


def test_capturing_a_never_sends_last_pair_back_to_defended_b():
    units, objects, rule = battle()
    reduce(units, 2)
    step(rule, units, objects)
    for unit in units[:2]:
        unit.position[:] = objects[2].position
    objects[2].raw.update(percent=1, redTeamNum=2)
    objects[0].raw['redTeamNum'] = 0
    for index in range(9, 28):
        s, actions = step(rule, units, objects, index)
        assert rule._route.objective(s).uid == objects[2].uid
        assert set(rule._route._garrison) == {u.uid for u in units[:2]}
        assert all(a.command != 'Moving' or rule._route.inside_garrison(objects[2], a.points)
                   for a in actions[:2])


def test_local_b_contact_cannot_replace_withdrawal_with_more_attacks():
    units, objects, rule = battle()
    reduce(units, 2)
    for u in units[:2]:
        u.position[:] = objects[1].position
    enemy = entity(9000, 1, 0, objects[1].position[0]+1000, objects[1].position[1])
    for u in units[:2]:
        u.raw['agentPerception'] = [enemy.uid]
    _, actions = step(rule, units, objects, enemies=[enemy])
    assert rule._route._remnant.active
    assert all(a.command == 'Moving' and a.target_uid is None for a in actions[:2])


@pytest.mark.parametrize('origin', [(-41810,32275,6425), (-50063,5399,6425),
                                   (-65000,-30945,5600), (-45450,-55000,5600)])
def test_replanned_segments_do_not_enter_rotated_b_exclusion(origin):
    units, objects, rule = battle()
    route = rule._route
    remnant = RemnantCapture()
    path = remnant.detour(route, origin, objects[2].position, objects[1])
    assert path and path[-1] == tuple(objects[2].position)
    bounds = remnant.bounds(route, objects[1])
    for left, right in zip((origin, *path), path):
        assert not remnant.crosses(left, right, bounds)


def test_b_interior_exit_never_reenters_after_leaving():
    units, objects, rule = battle()
    route, remnant = rule._route, RemnantCapture()
    path = remnant.detour(route, objects[1].position, objects[2].position, objects[1])
    bounds = remnant.bounds(route, objects[1])
    assert not remnant.inside(path[0], bounds)
    assert all(not remnant.crosses(a,b,bounds) for a,b in zip(path,path[1:]))


def test_episode_reset_clears_remnant_goal_and_reserve_exception():
    units, objects, rule = battle()
    reduce(units, 2)
    step(rule, units, objects)
    rule.act(replace(state(units, objectives=objects), episode=2))
    assert not rule._route._remnant.active
    assert rule._route.stage == 0


def test_hq_lost_reinstates_hq_priority():
    units, objects, rule = battle()
    reduce(units, 2)
    step(rule, units, objects)
    objects[0].raw.update(percent=.4, blueteamNum=4)
    for index in (9,10):
        s, _ = step(rule, units, objects,index)
    assert not rule._route._remnant.active
    assert rule._route.objective(s).uid == objects[0].uid


def test_surplus_arrival_at_a_does_not_return_to_stale_hq_exit_waypoint():
    units, objects, rule = battle()
    reduce(units, 4)
    step(rule, units, objects)
    for unit in units[:4]:
        unit.position[:] = objects[2].position
    objects[2].raw.update(percent=1, redTeamNum=4)
    _, actions = step(rule, units, objects, 9)
    assert len(rule._route._garrison) == 2
    assert all(a.command == 'Guard' or a.points is not None
               and rule._route.inside_garrison(objects[2], a.points) for a in actions[:4])
    assert all(a.command == 'Guard' for u, a in zip(units[:4], actions[:4])
               if u.uid not in rule._route._garrison)


def test_last_pair_is_released_after_later_losses_in_existing_stop_loss_mode():
    units, objects, rule = battle()
    reduce(units, 10)
    step(rule, units, objects)
    assert len(rule._route._garrison) == 1
    reduce(units, 2)
    _, actions = step(rule, units, objects, 9)
    assert not rule._route._garrison
    assert all(a.command == 'Moving' for a in actions[:2])
