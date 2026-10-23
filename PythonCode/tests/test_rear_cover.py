import pytest

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RearCover import RearCover
from test_fslutk_rule import entity, guards_at, map_objectives, state


def captured_case():
    objects = map_objectives()
    hq = objects[0]
    hq.raw.update(percent=1, redTeamNum=6, blueteamNum=0,
                  scale={'x': 7700, 'y': 6500})
    foot = guards_at(hq, count=6)
    x, y = hq.position[:2]
    cars = [entity(200 + i, 0, 6 + i, x + 2000 + 800 * i, y,
                   kind='BP_MNWS_Vehicle_6x6UGV_C' if i == 0 else 'BP_MNWS_Vehicle_Armored_C')
            for i in range(3)]
    for car in cars:
        car.raw.update(interaction=['100'], rSVD1='vehicle;1')
    route = AssaultRoute(adaptive=True)
    for step in range(4):
        route.update(state(foot + cars, objectives=objects, step=step))
    assert route.hq_secured and route.stage == 1
    return foot, cars, objects, route, RearCover()


def frame(foot, cars, objects, step=4, episode=1):
    return state(foot + cars, objectives=objects, step=step, episode=episode)


def test_exactly_one_nearby_vehicle_covers_without_blocking_other_vehicles():
    foot, cars, objects, route, cover = captured_case()
    s = frame(foot, cars, objects)
    actions = [cover.choose_vehicle(s, car, route) for car in cars]
    assert sum(a is not None for a in actions) == 1
    assert actions[0].command == 'Guard'
    assert actions[0].points == tuple(cars[0].position)
    assert cover.choose_vehicle(s, foot[0], route) is None
    # No navigation to an invented firing point is used for initial coverage.
    assert all(a is None or a.target_uid is None for a in actions)


@pytest.mark.parametrize('invalid', ['dead', 'empty', 'unknown_ammo', 'disconnected', 'far'])
def test_ineligible_vehicle_is_not_selected_and_no_far_vehicle_is_recalled(invalid):
    foot, cars, objects, route, cover = captured_case()
    car = cars[0]
    for other in cars[1:]:
        other.position[0] += 60000
    if invalid == 'dead':
        car.alive = False
    elif invalid == 'empty':
        car.raw['interaction'] = ['0']
    elif invalid == 'unknown_ammo':
        car.raw.pop('interaction')
    elif invalid == 'disconnected':
        car.raw['rSVD1'] = 'vehicle;0'
    else:
        car.position[0] += 20000
    s = frame(foot, cars, objects)
    assert all(cover.choose_vehicle(s, unit, route) is None for unit in cars)


def test_vehicle_requires_actual_nearby_garrison_not_just_hq_proximity():
    foot, cars, objects, route, cover = captured_case()
    route._garrison.clear()
    s = frame(foot, cars, objects)
    assert all(cover.choose_vehicle(s, unit, route) is None for unit in cars)


@pytest.mark.parametrize('invalid', ['dead', 'empty', 'disconnected'])
def test_replacement_keeps_original_deadline(invalid):
    foot, cars, objects, route, cover = captured_case()
    cover.begin(frame(foot, cars, objects), route)
    deadline = cover._deadline
    if invalid == 'dead':
        cars[0].alive = False
    elif invalid == 'empty':
        cars[0].raw['interaction'] = ['0']
    else:
        cars[0].raw['rSVD1'] = 'vehicle;0'
    s = frame(foot, cars, objects, step=deadline - 1)
    assert cover.choose_vehicle(s, cars[1], route) is not None
    assert cover._deadline == deadline
    s = frame(foot, cars, objects, step=deadline)
    assert all(cover.choose_vehicle(s, car, route) is None for car in cars)
    assert all(cover.choose_vehicle(frame(foot, cars, objects, step=deadline + 8), car, route)
               is None for car in cars)


def test_cover_returns_only_to_its_own_reached_position():
    foot, cars, objects, route, cover = captured_case()
    cover.begin(frame(foot, cars, objects), route)
    point = tuple(cars[0].position)
    cars[0].position[0] += 1500
    action = cover.choose_vehicle(frame(foot, cars, objects, step=5), cars[0], route)
    assert action.command == 'Moving' and action.points == point


def test_hq_loss_cancels_cover_and_recapture_can_start_one_new_window():
    foot, cars, objects, route, cover = captured_case()
    cover.begin(frame(foot, cars, objects), route)
    route.hq_secured = False
    assert cover.choose_vehicle(frame(foot, cars, objects, step=5), cars[0], route) is None
    assert cover._deadline is None
    route.hq_secured = True
    assert cover.choose_vehicle(frame(foot, cars, objects, step=9), cars[0], route) is not None
    assert cover._deadline == 9 + cover.WINDOW


@pytest.mark.parametrize('episode,step', [(2, 0), (1, 0)])
def test_new_episode_or_step_rewind_clears_old_damage_and_plans(episode, step):
    foot, cars, objects, route, cover = captured_case()
    cover.begin(frame(foot, cars, objects), route)
    foot[0].hp -= 20
    cover.guard_reaction(frame(foot, cars, objects, step=5), foot[0], objects[0], route)
    assert cover._plans
    route.hq_secured = False
    cover.begin(frame(foot, cars, objects, step=step, episode=episode), route)
    assert not cover._plans and not cover._damaged and cover._cover_uid is None
    assert cover._deadline is None and not cover._scan_started


def test_unseen_damage_gets_fixed_bounded_in_box_movement_and_retains_new_slot():
    foot, cars, objects, route, cover = captured_case()
    guard, hq = foot[0], objects[0]
    guard.position[:2] = hq.position[:2] + [6900, 5700]
    old = route.guard_slot(guard, hq)
    cover.begin(frame(foot, cars, objects), route)
    guard.hp -= 45
    action = cover.guard_reaction(frame(foot, cars, objects, step=5), guard, hq, route)
    assert action.command == 'Moving' and action.target_uid is None
    assert route.inside_garrison(hq, action.points)
    assert 800 <= route.distance(guard.position, action.points) <= 1600
    goal = action.points
    for step in range(6, 11):
        guard.hp -= 1
        again = cover.guard_reaction(frame(foot, cars, objects, step=step), guard, hq, route)
        assert again.points == goal
        assert cover._plans[guard.uid][2] == 11
    # Simulate partial navigation, not UE physics or a battle outcome.
    guard.position[:2] = (guard.position[:2] + goal[:2]) / 2
    assert cover.guard_reaction(frame(foot, cars, objects, step=11), guard, hq, route) is None
    assert not cover._plans
    assert route.guard_slot(guard, hq) == tuple(guard.position)
    assert route.guard_slot(guard, hq) != old
    assert cover.guard_reaction(frame(foot, cars, objects, step=12), guard, hq, route) is None


def test_reaction_arrival_holds_reached_point_and_later_damage_can_try_again():
    foot, cars, objects, route, cover = captured_case()
    guard, hq = foot[0], objects[0]
    cover.begin(frame(foot, cars, objects), route)
    guard.hp -= 10
    action = cover.guard_reaction(frame(foot, cars, objects, step=5), guard, hq, route)
    guard.position[:] = action.points
    assert cover.guard_reaction(frame(foot, cars, objects, step=6), guard, hq, route) is None
    assert route.guard_slot(guard, hq) == tuple(guard.position)
    guard.hp -= 10
    again = cover.guard_reaction(frame(foot, cars, objects, step=7), guard, hq, route)
    assert again.command == 'Moving' and again.points != action.points


def test_unhurt_guard_never_moves_or_fires_blind_and_scans_after_four_steps():
    foot, cars, objects, route, cover = captured_case()
    guard, hq = foot[0], objects[0]
    directions = set()
    for step in range(4, 40):
        s = frame(foot, cars, objects, step=step)
        cover.begin(s, route)
        assert cover.guard_reaction(s, guard, hq, route) is None
        action = cover.scan(s, guard, hq)
        assert action.command == 'Guard' and action.target_uid is None
        if step < 8:
            assert action.points == tuple(guard.position)
        else:
            assert action.points is None
            directions.add(action.direction)
    assert directions == set(cover.DIRECTIONS)


def test_reaction_does_not_pull_outside_or_disconnected_guard_into_a_new_probe():
    foot, cars, objects, route, cover = captured_case()
    cover.begin(frame(foot, cars, objects), route)
    foot[0].position[0] += 30000
    foot[1].raw['rSVD1'] = 'soldier;0'
    for guard in foot[:2]:
        guard.hp -= 10
    s = frame(foot, cars, objects, step=5)
    assert all(cover.guard_reaction(s, guard, objects[0], route) is None for guard in foot[:2])
