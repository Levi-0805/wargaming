"""Actual rule regressions for ignored rear contacts and blind HQ damage."""
import pytest

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RearCover import RearCover
from test_fslutk_rule import algorithm, entity, map_objectives, state


def captured(kind='BP_BaseSoldier_C', count=8, point=None, cars=0):
    objects = map_objectives()
    hq = objects[0]
    hq.raw.update(percent=1, redTeamNum=count, blueteamNum=0,
                  scale={'x': 7700, 'y': 6500})
    point = point or (-41810, 32275, 6425)
    foot = [entity(100+i, 0, i, *point[:2], kind=kind) for i in range(count)]
    vehicles = [entity(200+i, 0, count+i, point[0]+2000+i*700, point[1],
                       kind='BP_MNWS_Vehicle_6x6UGV_C') for i in range(cars)]
    for unit in foot + vehicles:
        unit.position[2] = point[2]
        unit.raw.update(agentPerception=[], interaction=['250'], rSVD1='unit;1')
    rule = algorithm(foot + vehicles)
    for step in range(4):
        rule.act(state(foot + vehicles, objectives=objects, step=step))
    assert rule._route.hq_secured and rule._route.stage == 1
    return foot, vehicles, objects, rule


def contact(unit, distance=8000, kind='BP_MNWS_Vehicle_6x6UGV_C'):
    enemy = entity(999, 1, 0, unit.position[0], unit.position[1]+distance, kind=kind)
    enemy.position[2] = unit.position[2]
    return enemy


@pytest.mark.parametrize('kind', ['BP_BaseSoldier_C', 'BP_RoboDog_C'])
@pytest.mark.parametrize('enemy_kind', ['BP_MNWS_Vehicle_6x6UGV_C', 'BP_MNWS_Vehicle_Armored_C'])
def test_departing_foot_can_defend_against_a_local_vehicle_eighty_metres_behind(kind, enemy_kind):
    foot, _, objects, rule = captured(kind)
    unit = foot[-1]
    assert unit.uid not in rule._route._garrison
    enemy = contact(unit, kind=enemy_kind)
    unit.raw['agentPerception'] = [enemy.uid]
    action = rule.act(state(foot, [enemy], objects, step=4))[-1]
    assert action.command == 'NormalAttacking' and action.target_uid == enemy.uid
    assert rule._route.stage == 1


@pytest.mark.parametrize('distance,local,attacks', [(15000, True, True), (15001, True, False), (8000, False, False)])
def test_departure_self_defence_is_local_and_bounded(distance, local, attacks):
    foot, _, objects, rule = captured()
    unit = foot[-1]
    enemy = contact(unit, distance)
    unit.raw['agentPerception'] = [enemy.uid] if local else []
    action = rule.act(state(foot, [enemy], objects, step=4))[-1]
    assert (action.command == 'NormalAttacking') == attacks
    if not attacks:
        assert action.command == 'Moving' and action.target_uid is None


def test_departure_vehicle_threat_precedes_a_nearby_infantry_distraction():
    foot, _, objects, rule = captured()
    unit = foot[-1]
    armor, soldier = contact(unit), contact(unit, 1500, 'BP_BaseSoldier_C')
    soldier.uid = 998
    unit.raw['agentPerception'] = [armor.uid, soldier.uid]
    action = rule.act(state(foot, [soldier, armor], objects, step=4))[-1]
    assert action.command == 'NormalAttacking' and action.target_uid == armor.uid


def test_unrelated_distant_infantry_still_does_not_pin_departing_main_body():
    foot, _, objects, rule = captured()
    enemy = contact(foot[-1], kind='BP_BaseSoldier_C')
    for unit in foot:
        unit.raw['agentPerception'] = [enemy.uid]
    actions = rule.act(state(foot, [enemy], objects, step=4))
    assert all(action.command == 'Moving' for unit, action in zip(foot, actions)
               if unit.uid not in rule._route._garrison)


@pytest.mark.parametrize('offset', [7000, 9000])
def test_hq_guard_defends_from_its_position_even_at_or_outside_the_boundary(offset):
    foot, _, objects, rule = captured(count=4)
    guard = foot[0]
    guard.position[0] -= offset
    enemy = entity(999, 1, 0, guard.position[0]-8000, guard.position[1],
                   kind='BP_MNWS_Vehicle_6x6UGV_C')
    enemy.position[2] = guard.position[2]
    guard.raw['agentPerception'] = [enemy.uid]
    assert AssaultRoute.distance(enemy.position, objects[0].position) > 12000
    action = rule.act(state(foot, [enemy], objects, step=4))[0]
    assert rule._route.garrison_for(state(foot, objectives=objects), guard) is not None
    assert action.command == 'NormalAttacking' and action.target_uid == enemy.uid


def test_guard_reposition_while_fighting_rear_armor_stays_inside_capture_box():
    foot, _, objects, rule = captured(count=4, point=(-48810, 32275, 6425))
    guard = foot[0]
    enemy = entity(999, 1, 0, guard.position[0]-8000, guard.position[1],
                   kind='BP_MNWS_Vehicle_6x6UGV_C')
    enemy.position[2] = guard.position[2]
    guard.raw['agentPerception'] = [999]
    moves = []
    for step in range(4, 22):
        action = rule.act(state(foot, [enemy], objects, step=step))[0]
        if action.command == 'Moving':
            moves.append(action)
            assert rule._route.inside_garrison(objects[0], action.points)
    assert moves


def test_blind_damage_immediately_moves_guard_and_does_not_recall_it_to_old_slot():
    foot, _, objects, rule = captured('BP_RoboDog_C', count=4, point=(-41000, 26400, 6425))
    guard = foot[0]
    old = tuple(guard.position)
    guard.hp = 45.58  # Same observable trigger as recorded dog 386 at step160.
    action = rule.act(state(foot, objectives=objects, step=4))[0]
    assert action.command == 'Moving' and action.target_uid is None
    assert rule._route.inside_garrison(objects[0], action.points)
    assert 800 <= AssaultRoute.distance(old, action.points) <= 1600
    guard.position[:] = action.points
    action = rule.act(state(foot, objectives=objects, step=5))[0]
    assert action.command == 'Guard' and action.target_uid is None
    assert rule._route.guard_slot(guard, objects[0]) == tuple(guard.position)
    assert rule._route.guard_slot(guard, objects[0]) != old


def test_one_temporary_rear_vehicle_leaves_other_vehicles_and_main_body_free_to_advance():
    foot, cars, objects, rule = captured(cars=3)
    for step in range(4, 7):
        current = state(foot + cars, objectives=objects, step=step)
        actions = rule.act(current)
    vehicle_actions = actions[len(foot):]
    holding = [(u, a) for u, a in zip(cars, vehicle_actions)
               if a.command == 'Guard' and a.points is not None
               and AssaultRoute.distance(u.position, a.points) <= RearCover.HOLD_RADIUS]
    assert len(holding) == 1
    assert any(a.points is not None and AssaultRoute.distance(u.position, a.points) > 2000
               for u, a in zip(cars, vehicle_actions))
    assert all(a.command == 'Moving' for u, a in zip(foot, actions)
               if u.uid not in rule._route._garrison)
    selected = holding[0][0]
    for step in range(36, 39):
        actions = rule.act(state(foot + cars, objectives=objects, step=step))
    action = actions[(foot + cars).index(selected)]
    assert action.points is not None and AssaultRoute.distance(action.points, selected.position) > 2000


def test_rear_vehicle_uses_normal_local_fire_checks_before_cover_scanning():
    foot, cars, objects, rule = captured(cars=3)
    selected = min(cars, key=lambda u: AssaultRoute.distance(u.position, foot[0].position))
    enemy = contact(selected)
    # A team report alone does not authorize the rear vehicle to fire.
    for step in range(4, 6):
        actions = rule.act(state(foot + cars, [enemy], objects, step=step))
        assert all(a.command != 'NormalAttacking' for a in actions[len(foot):])
    selected.raw['agentPerception'] = [enemy.uid]
    action = rule.act(state(foot + cars, [enemy], objects, step=6))[len(foot)]
    assert action.command == 'Guard' and action.target_uid is None
    action = rule.act(state(foot + cars, [enemy], objects, step=7))[len(foot)]
    assert action.command == 'NormalAttacking' and action.target_uid == enemy.uid
