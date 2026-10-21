"""Clearance failures from the Oct20 evening batch, without hidden targets."""
import math

import pytest

from cssim.protocol import Action
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.UrbanContact import UrbanContact
from test_fslutk_rule import algorithm, entity, guards_at, map_objectives, objective, state


def box():
    obj = objective(901, 0, name='HQ', red=8, blue=2, percent=1)
    obj.raw.update(scale={'x': 7700, 'y': 6500}, rotation={'yaw': 31})
    return obj


def test_search_keeps_a_blocked_goal_long_enough_for_real_garrison_escape():
    obj = box()
    units = guards_at(obj, first=1, count=8)
    rule = algorithm(units)
    initial = rule.act(state(units, objectives=[obj]))[0]
    assert initial.command == 'Moving'
    unit = units[0]
    assert unit.uid in rule._route._garrison
    actions = [rule.act(state(units, objectives=[obj], step=step))[0]
               for step in range(1, 13)]
    assert all(a.points == initial.points for a in actions[:7])
    # Recovery at step 8 must move away from the old position/guard slot.
    escapes = [a for a in actions[7:] if a.points != initial.points]
    assert escapes and all(a.command == 'Moving' and a.points is not None for a in escapes)
    assert all(200 < rule._route.distance(unit.position, a.points) < 2000 for a in escapes)
    assert all(rule._route.inside_garrison(obj, a.points) for a in actions)
    assert rule._urban._plans[unit.uid][1] == initial.points


def test_search_explores_all_public_box_sectors_without_enemy_coordinates():
    obj, contact, route = box(), UrbanContact(), AssaultRoute()
    unit = entity(1, 0, 0, 0)
    goals = set()
    for step in range(contact.CLEAR_PLAN_STEPS * 8):
        s = state([unit], objectives=[obj], step=step)
        contact.begin(s)
        action = contact.search_point(s, unit, obj, route)
        assert action.command == 'Moving' and action.target_uid is None
        assert route.inside_garrison(obj, action.points)
        goals.add(action.points)
    assert len(goals) == 8
    locals_ = [route._local_point(obj, p) for p in goals]
    assert min(p[0] for p in locals_) < -5000 < 5000 < max(p[0] for p in locals_)
    assert min(p[1] for p in locals_) < -4000 < 4000 < max(p[1] for p in locals_)


def test_non_garrison_main_body_searches_contested_b():
    objects = map_objectives()
    hq, base, _ = objects
    hq.raw.update(percent=1, redTeamNum=4, blueteamNum=0)
    base.raw.update(percent=1, redTeamNum=12, blueteamNum=1)
    home = guards_at(hq, first=1, count=4)
    body = guards_at(base, first=10, count=12)
    rule = algorithm(home + body)
    for step in range(4):
        rule.act(state(home + body, objectives=objects, step=step))
    assert rule._route.objective(state(home + body, objectives=objects)).uid == base.uid
    rule._cohesion.released = True
    actions = rule.act(state(home + body, objectives=objects, step=4))
    body_ids = {u.uid for u in body}
    searchers = [(u, a) for u, a in zip(home + body, actions)
                 if u.uid in body_ids and u.uid not in rule._route._garrison]
    assert searchers
    for unit, action in searchers:
        assert action.command == 'Moving' and action.target_uid is None
        assert rule._urban._plans[unit.uid][0] == ('clear', base.uid)


def test_local_contact_immediately_interrupts_clearance_and_empty_box_stops_sweep():
    obj = box()
    units = guards_at(obj, first=1, count=8)
    rule = algorithm(units)
    rule.act(state(units, objectives=[obj]))
    enemy = entity(999, 1, 0, 500)
    units[0].raw.update(agentPerception=[999], interaction=['250'])
    action = rule.act(state(units, [enemy], [obj], step=1))[0]
    assert action.command == 'NormalAttacking' and action.target_uid == 999
    obj.raw['blueteamNum'] = 0
    units[0].raw['agentPerception'] = []
    rule.act(state(units, objectives=[obj], step=2))
    assert not rule._urban._sweeps
    assert not any(p[0][0] == 'clear' for p in rule._urban._plans.values())


@pytest.mark.parametrize('yaw,direction', [(0, '+X'), (90, '+Y')])
def test_escape_at_capture_boundary_keeps_a_real_inward_displacement(yaw, direction):
    obj = box()
    obj.raw['rotation']['yaw'] = yaw
    route, contact = AssaultRoute(), UrbanContact()
    extent = route._footprint(obj)[0]
    angle = math.radians(yaw)
    unit = entity(1, 0, 0, extent * math.cos(angle), extent * math.sin(angle))
    action = contact.constrain(unit, Action.move(direction), obj, route)
    assert 200 < route.distance(unit.position, action.points) <= 800.01
    assert route.inside_garrison(obj, action.points)


def test_sweep_state_resets_for_a_new_episode():
    obj, contact, route = box(), UrbanContact(), AssaultRoute()
    unit = entity(1, 0, 0, 0)
    s = state([unit], objectives=[obj])
    contact.begin(s)
    first = contact.search_point(s, unit, obj, route)
    s = state([unit], objectives=[obj], step=contact.CLEAR_PLAN_STEPS)
    contact.begin(s)
    assert contact.search_point(s, unit, obj, route).points != first.points
    s = state([unit], objectives=[obj], episode=2)
    contact.begin(s)
    assert contact.search_point(s, unit, obj, route).points == first.points
