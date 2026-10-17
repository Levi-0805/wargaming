"""Regression checks for routes and holdings observed in low-score replays."""
from test_adaptive_assault import advance, opening
from test_fslutk_rule import entity, guards_at, state


def test_a_approach_counts_b_but_not_when_already_past_b():
    units, objects, route = opening(approaching=True)
    objects[1].raw['scale'] = {'x': 10500, 'y': 9500}
    s = advance(route, units, objects, [0])
    assert objects[1] in route.transit_objectives(s, units[0].position, objects[2])
    assert objects[1] not in route.transit_objectives(s, objects[2].position, objects[2])
    objects[1].raw.update(percent=1, redTeamNum=4)
    assert objects[1] not in route.transit_objectives(s, units[0].position, objects[2])


def test_incidental_capture_reserves_guards_without_changing_attack_target():
    units, objects, route = opening()
    units += guards_at(objects[1])
    s = advance(route, units, objects, [0])
    route._reorder(s, (objects[2].uid, objects[1].uid, objects[0].uid))
    objects[1].raw.update(percent=1, redTeamNum=4)
    s = advance(route, units, objects, range(1, 4))
    assert route.stage == 0
    s = advance(route, units, objects, [4])
    assert route.stage == 1 and route.objective(s).uid == objects[2].uid
    assert list(route._garrison.values()).count(objects[1].uid) == 4


def test_incidental_capture_needs_consecutive_empty_observations():
    units, objects, route = opening()
    units += guards_at(objects[1])
    objects[1].raw.update(percent=1, redTeamNum=4)
    advance(route, units, objects, [0, 1, 10, 11])
    assert route.stage == 0
    objects[1].raw['blueteamNum'] = 1
    advance(route, units, objects, [12, 13, 14, 15])
    assert route.stage == 0


def test_replanned_mobile_unit_skips_old_a_leg_when_goal_is_hq():
    units, objects, route = opening()
    s = advance(route, units, objects, [0])
    route._reorder(s, (objects[1].uid, objects[2].uid, objects[0].uid), 2)
    unit = units[0]
    unit.position[:] = (-38797, -35688, 5620)
    route._cursor[unit.uid] = (1, 0)
    route.point(s, unit)
    assert route._cursor[unit.uid][0] == 2
    assert route._route_for(unit, 2)[-1] == tuple(objects[0].position)


def test_small_surviving_force_retains_mobile_capturers_until_threat_returns():
    _, objects, route = opening()
    units = guards_at(objects[1], count=2) + guards_at(objects[2], first=200, count=2)
    for obj in objects[1:]:
        obj.raw.update(percent=1, redTeamNum=2)
    s = advance(route, units, objects, range(8))
    route._reorder(s, (objects[1].uid, objects[2].uid, objects[0].uid), 2)
    route._assign_garrison(s)
    assert len(route._garrison) == 2
    assert all(route.garrison_needed(s, obj) == 1 for obj in objects[1:])
    enemy = entity(999, 1, 0, *objects[1].position[:2])
    s = advance(route, units, objects, [8], [enemy])
    assert route.garrison_needed(s, objects[1]) >= 4
