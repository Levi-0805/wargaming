"""Front-entry routing regressions; movement requests are not live match results."""
import math

import pytest

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.HQEntry import HQEntry
from test_fslutk_rule import algorithm, entity, guards_at, map_objectives, state


GROUND = ("BP_BaseSoldier_C", "BP_RoboDog_C",
          "BP_MNWS_Vehicle_Armored_C", "BP_MNWS_Vehicle_6x6UGV_C")
OLD_CORNER = (-40107, 36086)


def front_case(kind="BP_BaseSoldier_C", point=None, *, contested_bar=False):
    point = point or AssaultRoute.LOWER[-1]
    unit = entity(1, 0, 0, point[0], point[1], kind=kind)
    unit.position[:] = point
    unit.raw.update(agentPerception=[], interaction=["250"], rSVD1="unit;1")
    objects = map_objectives()
    objects[0].raw.update(blueteamNum=2, scale={"x": 7700, "y": 6500})
    if contested_bar:
        objects[0].raw.update(percent=1, redTeamNum=8)
    # Do not confuse the wheel vehicle's separate, bounded heading correction
    # with the first requested entrance. Its normal route must start at once.
    arrival = HQEntry.VEHICLE_ARRIVAL if kind in HQEntry.VEHICLES else HQEntry.ARRIVAL
    goal = next(p for p in AssaultRoute.HQ_FRONT
                if AssaultRoute.distance(unit.position, p) > arrival)
    unit.yaw = math.degrees(math.atan2(goal[1] - point[1], goal[0] - point[0]))
    return unit, objects, algorithm([unit])


def segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    t = 0 if denominator == 0 else max(0, min(1, (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator))
    return math.hypot(point[0] - start[0] - t * dx,
                      point[1] - start[1] - t * dy)


def on_front_corridor(point):
    # Vehicle and infantry height conventions intentionally differ; this
    # change concerns the horizontal entrance used by all four ground types.
    return point is not None and any(tuple(point[:2]) == tuple(p[:2])
                                     for p in AssaultRoute.HQ_FRONT)


def test_both_front_corridors_avoid_the_recorded_fatal_corner_including_segments():
    # Checking only individual waypoints can miss a straight segment crossing
    # the old bend. This validates requested geometry, not UE path feasibility.
    for corridor in (AssaultRoute.HQ_FRONT, AssaultRoute.HQ_FRONT_ALTERNATE):
        chain = (AssaultRoute.LOWER[-1], *corridor)
        assert all(segment_distance(OLD_CORNER, left, right) > 2000
                   for left, right in zip(chain, chain[1:]))


@pytest.mark.parametrize("kind", GROUND)
def test_first_hq_approach_uses_front_corridor_without_a_facade_stall(kind):
    unit, objects, rule = front_case(kind)
    current = state([unit], objectives=objects)
    action = rule.act(current)[0]
    assert action.command == ("Guard" if kind in HQEntry.VEHICLES else "Moving")
    assert on_front_corridor(action.points)
    assert action.points != tuple(objects[0].position)
    assert AssaultRoute.distance(action.points, OLD_CORNER) > 2000
    assert rule._hq_entry.maneuvering(current, unit, rule._route)
    assert rule._route.objective(current).uid == objects[0].uid


@pytest.mark.parametrize("kind", GROUND[:2])
def test_contested_garrison_can_finish_front_entry_before_interior_search(kind):
    unit, objects, rule = front_case(kind, (-43531, 26968, 6425), contested_bar=True)
    current = state([unit], objectives=objects)
    action = rule.act(current)[0]
    assert rule._route.inside_garrison(objects[0], unit.position)
    assert rule._route.garrison_for(current, unit).uid == objects[0].uid
    assert rule._hq_entry.maneuvering(current, unit, rule._route)
    # This reached position is inside the capture box, but entering from the
    # front requires an outside approach. The final guard clamp must not cut it.
    assert action.command == "Moving"
    assert on_front_corridor(action.points)
    assert not rule._route.inside_garrison(objects[0], action.points)
    assert not any(plan[0][0] == "clear" for plan in rule._urban._plans.values())


def test_completed_entry_restores_garrison_bounds_for_clearance_and_recovery():
    unit, objects, rule = front_case(point=AssaultRoute.HQ_FRONT[-1], contested_bar=True)
    for step in range(30):
        current = state([unit], objectives=objects, step=step)
        action = rule.act(current)[0]
        assert unit.uid in rule._hq_entry._completed
        assert not rule._hq_entry.maneuvering(current, unit, rule._route)
        assert rule._route.garrison_for(current, unit).uid == objects[0].uid
        if action.points is not None:
            assert rule._route.inside_garrison(objects[0], action.points)
    assert rule._urban._plans[unit.uid][0] == ("clear", objects[0].uid)


@pytest.mark.parametrize("kind", GROUND)
def test_blocked_front_entry_has_a_finite_navigation_budget_and_new_episode_retry(kind):
    unit, objects, rule = front_case(kind)
    limit = (len(AssaultRoute.HQ_FRONT_ALTERNATE) + 3) * (HQEntry.WAYPOINT_TRIALS + 1) + 40
    actions = []
    for step in range(limit):
        current = state([unit], objectives=objects, step=step)
        actions.append(rule.act(current)[0])
        assert rule._route.objective(current).uid == objects[0].uid
    assert unit.uid in rule._hq_entry._exhausted
    assert not rule._hq_entry.maneuvering(current, unit, rule._route)
    # Exhaustion releases navigation to the current objective; it must not turn
    # into a permanent stationary Guard or unlock a different objective.
    assert any(a.command == "Moving" or
               (a.command == "Guard" and a.points is not None
                and AssaultRoute.distance(a.points, unit.position) > 300)
               for a in actions[-24:])
    new_episode = state([unit], objectives=objects, episode=2)
    restarted = rule.act(new_episode)[0]
    assert unit.uid not in rule._hq_entry._exhausted
    assert not rule._hq_entry._alternate
    assert rule._hq_entry.maneuvering(new_episode, unit, rule._route)
    assert on_front_corridor(restarted.points)


@pytest.mark.parametrize("kind", GROUND)
def test_local_fire_interrupts_front_entry_then_resumes_the_same_corner(kind):
    unit, objects, rule = front_case(kind)
    initial = rule.act(state([unit], objectives=objects))[0]
    cursor = rule._hq_entry._routes[unit.uid]
    enemy = entity(20, 1, 0, unit.position[0] + 500, unit.position[1])
    enemy.position[2] = unit.position[2]
    unit.raw["agentPerception"] = [enemy.uid]
    fired = []
    for step in range(1, 7):
        enemy.hp = 100 - step
        unit.raw["interaction"] = [str(250 - step)]
        action = rule.act(state([unit], [enemy], objects, step))[0]
        fired.append(action)
        assert rule._hq_entry._routes[unit.uid] == cursor
    assert any(a.command == "NormalAttacking" and a.target_uid == enemy.uid for a in fired)
    unit.raw["agentPerception"] = []
    resumed = [rule.act(state([unit], objectives=objects, step=step))[0]
               for step in range(7, 12)]
    assert any(a.points == initial.points for a in resumed)
    assert rule._hq_entry._routes[unit.uid] == cursor


def test_front_entry_does_not_treat_a_contested_full_hq_bar_as_permission_for_ab():
    objects = map_objectives()
    objects[0].raw.update(percent=1, redTeamNum=8, blueteamNum=2,
                          scale={"x": 7700, "y": 6500})
    units = [entity(i + 1, 0, i, -41810, 32275) for i in range(8)]
    for unit in units:
        unit.position[2] = 6425
        unit.raw.update(agentPerception=[], interaction=["250"])
    rule = algorithm(units)
    for step in range(40):
        current = state(units, objectives=objects, step=step)
        rule.act(current)
        assert rule._route.objective(current).uid == objects[0].uid
        assert not rule._route.hq_secured


@pytest.mark.parametrize("kind", GROUND[:2])
def test_first_contact_of_provisional_guard_fires_before_returning_to_front_entry(kind):
    unit, objects, rule = front_case(kind, (-32000, 28000, 6425), contested_bar=True)
    enemy = entity(20, 1, 0, -33000, 28000)
    enemy.position[2] = 6425
    unit.raw["agentPerception"] = [enemy.uid]
    current = state([unit], [enemy], objects)
    action = rule.act(current)[0]
    assert rule._route.garrison_for(current, unit).uid == objects[0].uid
    assert not rule._route.inside_garrison(objects[0], unit.position)
    assert action.command == "NormalAttacking" and action.target_uid == enemy.uid
    assert rule._hq_entry.pending(current, unit, rule._route)
    # Previously the first local contact prevented creation of the entry cursor,
    # and _defend replaced combat with a move to the clamped wall (-34510,28000).
    unit.raw["agentPerception"] = []
    resumed = rule.act(state([unit], objectives=objects, step=1))[0]
    assert on_front_corridor(resumed.points)
    assert tuple(resumed.points[:2]) != (-34510, 28000)


@pytest.mark.parametrize("held_b", [False, True])
def test_lost_hq_reattack_retains_front_corridor_after_entry_exhaustion(held_b):
    objects = map_objectives()
    objects[0].raw.update(percent=1, redTeamNum=8, blueteamNum=0,
                          scale={"x": 7700, "y": 6500})
    units = guards_at(objects[0], first=1, count=8)
    if held_b:
        objects[1].raw.update(percent=1, redTeamNum=8, blueteamNum=0)
        units += guards_at(objects[1], first=20, count=8)
    for unit in units:
        unit.position[2] = 6425
        unit.raw.update(agentPerception=[], interaction=["250"])
    rule = algorithm(units)
    for step in range(8):
        rule.act(state(units, objectives=objects, step=step))
    assert rule._route.hq_secured
    assert rule._route.stage == (2 if held_b else 1)
    objects[0].raw.update(percent=0, redTeamNum=0, blueteamNum=2)
    for unit in units:
        unit.position[:] = AssaultRoute.LOWER[-1]
    for step in range(8, 10):
        rule.act(state(units, objectives=objects, step=step))
    current = state(units, objectives=objects, step=9)
    assert rule._route.objective(current).uid == objects[0].uid
    # When B is still owned, reordering retains it before HQ. HQ entry cannot
    # rely on being route leg zero.
    assert rule._route.stage == int(held_b)
    hq_path = rule._route._route_for(units[0], rule._route.stage)
    assert all(point in hq_path for point in AssaultRoute.HQ_FRONT)
    limit = 10 + (len(AssaultRoute.HQ_FRONT_ALTERNATE) + 3) * (HQEntry.WAYPOINT_TRIALS + 1)
    fallback = []
    for step in range(10, limit):
        current = state(units, objectives=objects, step=step)
        action = rule.act(current)[0]
        assert rule._route.objective(current).uid == objects[0].uid
        if units[0].uid in rule._hq_entry._exhausted:
            if action.points is not None:
                fallback.append(action.points)
            if len(fallback) == 3:
                break
    assert len(fallback) == 3
    assert all(on_front_corridor(point) for point in fallback)
    assert all(AssaultRoute.distance(point, objects[0].position) > 5000 for point in fallback)
    assert rule._hq_entry.pending(current, units[0], rule._route)
    assert not rule._route.hq_secured


def test_exhausted_provisional_guard_continues_front_route_without_clamping_or_search():
    unit, objects, rule = front_case(point=(-43531, 26968, 6425), contested_bar=True)
    limit = (len(AssaultRoute.HQ_FRONT_ALTERNATE) + 3) * (HQEntry.WAYPOINT_TRIALS + 1)
    fallback = []
    for step in range(limit):
        current = state([unit], objectives=objects, step=step)
        action = rule.act(current)[0]
        if unit.uid in rule._hq_entry._exhausted:
            assert rule._hq_entry.pending(current, unit, rule._route)
            assert unit.uid not in rule._hq_entry._completed
            assert rule._route.garrison_for(current, unit).uid == objects[0].uid
            assert not any(plan[0][0] == "clear" for plan in rule._urban._plans.values())
            if action.points is not None:
                fallback.append(action.points)
            if len(fallback) == 3:
                break
    assert len(fallback) == 3
    assert all(on_front_corridor(point) for point in fallback)
    # The same pending entrance exemption must survive exhausting the dedicated
    # route; clamping this particular next turn would cut it at y=26175.
    assert all(not rule._route.inside_garrison(objects[0], point) for point in fallback)


def test_established_hq_guards_do_not_restart_ingress_while_main_force_attacks_b():
    objects = map_objectives()
    objects[0].raw.update(percent=1, redTeamNum=8, blueteamNum=0,
                          scale={"x": 7700, "y": 6500})
    guards = guards_at(objects[0], first=1, count=8)
    for guard in guards:
        # Inside the held footprint but outside the return entrance's finish
        # radius, so a new ingress would wrongly recall these defenders.
        guard.position[:] = (-43531, 26968, 6425)
        guard.raw.update(agentPerception=[], interaction=["250"])
    rule = algorithm(guards)
    for step in range(4):
        current = state(guards, objectives=objects, step=step)
        rule.act(current)
    assert rule._route.hq_secured and rule._route.stage == 1
    objects[0].raw["blueteamNum"] = 1
    for step in range(4, 9):
        current = state(guards, objectives=objects, step=step)
        actions = rule.act(current)
        for guard, action in zip(guards, actions):
            if rule._route.garrison_for(current, guard) is not None:
                assert not rule._hq_entry.pending(current, guard, rule._route)
                assert guard.uid not in rule._hq_entry._routes
                if action.points is not None:
                    assert rule._route.inside_garrison(objects[0], action.points)
