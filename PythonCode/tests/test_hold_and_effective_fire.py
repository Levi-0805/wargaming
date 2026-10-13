"""Observed failures in 20261013_195259: lost HQ, idle UAVs, ineffective fire."""
import math

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.UAVControl import UAVControl
from test_air_and_departure import aircraft
from test_fslutk_rule import algorithm, entity, guards_at, map_objectives, state


def test_guards_outside_capture_core_return_before_main_force_departs():
    objects = map_objectives()
    objects[0].raw.update(percent=1, redTeamNum=8, blueteamNum=4)
    guards = guards_at(objects[0], count=8)
    for guard in guards:
        guard.position[1] += 7000  # Actual first round: assigned guards 60-75m away.
    rule = algorithm(guards)
    for step in range(6):
        actions = rule.act(state(guards, objectives=objects, step=step))
    assert rule._route.stage == 0
    assert len(rule._route._garrison) == 7
    for guard, action in zip(guards, actions):
        if guard.uid in rule._route._garrison:
            assert action.command == "Moving"
            assert AssaultRoute.distance(action.points, objects[0].position) < 2500
    for guard in guards:
        guard.position[1] -= 7000
    rule.act(state(guards, objectives=objects, step=6))
    assert rule._route.stage == 1
    assert len(rule._route._garrison) == 7  # Remaining unit can advance immediately.


def test_real_hq_footprint_holds_reached_positions_without_a_centre_arrival_gate():
    objects = map_objectives()
    obj = objects[0]
    obj.raw.update(scale={"x": 7700, "y": 6500, "z": 2500},
                   percent=1, redTeamNum=11, blueteamNum=4)
    guards = guards_at(obj, count=8)
    for i, guard in enumerate(guards):
        guard.position[0] += (i - 4) * 200
        guard.position[1] += 5500
    rule = algorithm(guards)
    for step in range(4):
        actions = rule.act(state(guards, objectives=objects, step=step))
    assert rule._route.stage == 1
    assert len(rule._route._garrison) == 7
    for guard, action in zip(guards, actions):
        if guard.uid in rule._route._garrison:
            assert action.command == "Guard"
            assert action.points == tuple(guard.position)
    assert not rule._route.inside_garrison(obj, (obj.position[0], obj.position[1] + 7000, 0))


def test_scaled_rotated_hold_footprint_clamps_reposition_and_preserves_height():
    obj = map_objectives()[0]
    obj.raw.update(scale={"x": 7700, "y": 6500, "z": 2500}, rotation={"yaw": 90})
    route = AssaultRoute()
    outside = (obj.position[0] + 10000, obj.position[1] + 10000, 6400)
    clamped = route.clamp_guard_point(obj, outside)
    assert abs(clamped[0] - obj.position[0]) <= 6100.1
    assert abs(clamped[1] - obj.position[1]) <= 7300.1
    assert clamped[2] == 6400
    assert route.inside_garrison(obj, clamped)


def test_guard_reserve_is_not_cut_to_leave_two_attackers_and_dead_guards_are_replaced():
    objects = map_objectives()
    objects[0].raw.update(percent=1, redTeamNum=6, blueteamNum=2)
    guards = guards_at(objects[0], count=6)
    route = AssaultRoute()
    for step in range(4):
        route.update(state(guards, objectives=objects, step=step))
    assert route.stage == 1 and len(route._garrison) == 5
    dead = next(unit for unit in guards if unit.uid in route._garrison)
    dead.alive = False
    objects[0].raw["redTeamNum"] = 5
    route.update(state(guards, objectives=objects, step=4))
    assert len(route._garrison) == 5 and dead.uid not in route._garrison


def test_guard_attacking_and_recovery_cannot_chase_out_of_capture_core():
    objects = map_objectives()
    objects[0].raw.update(percent=1, redTeamNum=5)
    guards = guards_at(objects[0], count=5)
    enemy = entity(9000, 1, 0, -41810, 40000)
    for guard in guards:
        guard.position[0] += 2400
        guard.raw.update(agentPerception=[enemy.uid], interaction=["250"])
    rule = algorithm(guards)
    moves = 0
    for step in range(30):
        actions = rule.act(state(guards, [enemy], objects, step))
        for guard, action in zip(guards, actions):
            if guard.uid in rule._route._garrison and action.command == "Moving":
                moves += 1
                assert action.points is not None
                assert AssaultRoute.distance(action.points, objects[0].position) <= 2100.1
    assert moves > 0


def test_guard_does_not_follow_a_distant_public_contact():
    objects = map_objectives()
    objects[0].raw.update(percent=1, redTeamNum=4)
    guards = guards_at(objects[0])
    enemy = entity(9000, 1, 0, -41810, 52275)
    rule = algorithm(guards)
    actions = rule.act(state(guards, [enemy], objects))
    assert all(a.command in {"Guard", "Moving"} for a in actions)
    assert all(AssaultRoute.distance(a.points, objects[0].position) < 2500 for a in actions)


def test_limited_survivors_still_fight_without_releasing_the_capture_point():
    objects = map_objectives()
    objects[0].raw.update(percent=1, redTeamNum=3, blueteamNum=2)
    guards = guards_at(objects[0], count=3)
    enemy = entity(9000, 1, 0, -41810, 34275)
    rule = algorithm(guards)
    for step in range(4):
        for guard in guards:
            guard.raw["interaction"] = [str(250 - step)]
        actions = rule.act(state(guards, [enemy], objects, step))
    assert rule._route.stage == 0 and len(rule._route._garrison) == 3
    assert all(a.command == "NormalAttacking" for a in actions)


def test_ammunition_consumed_without_damage_eventually_changes_target():
    soldier = entity(1, 0, 0, 0)
    blocked, reachable = entity(20, 1, 0, 4000), entity(21, 1, 1, 5000)
    soldier.raw["agentPerception"] = [20, 21]
    rule = algorithm([soldier])
    actions = []
    for step in range(10):
        soldier.raw["interaction"] = [str(250 - step)]
        actions.append(rule.act(state([soldier], [blocked, reachable], step=step))[0])
    assert [a.target_uid for a in actions[:8]] == [20] * 8
    assert actions[8].command == "Moving"
    assert actions[9].target_uid == 21


def test_local_target_has_priority_over_more_popular_shared_target():
    agents = [entity(i, 0, i, 0) for i in range(3)]
    shared = entity(20, 1, 0, 4000)
    local = entity(21, 1, 1, 2000, kind="BP_RoboDog_C")
    for agent in agents:
        agent.raw["agentPerception"] = [20]
    agents[0].raw["agentPerception"] = [21]
    assert algorithm(agents).act(state(agents, [shared, local]))[0].target_uid == 21


def test_empty_vehicle_moves_to_objective_instead_of_repeating_dry_fire():
    car = entity(1, 0, 0, -17959, 27728, kind="BP_MNWS_Vehicle_Armored_C")
    car.raw.update(interaction=["0.0"], agentPerception=[20])
    enemy = entity(20, 1, 0, -18000, 31000)
    action = algorithm([car]).act(state([car], [enemy], map_objectives()))[0]
    assert action.command == "Moving"


def test_five_low_uavs_all_get_forward_destinations_at_actual_spawn_height():
    objectives = map_objectives()
    objectives[0].position[2] = 6425
    uavs = [aircraft(i + 1, x=61000 + i * 100, y=36000, z=6000) for i in range(5)]
    actions = algorithm(uavs).act(state(uavs, objectives=objectives))
    for uav, action in zip(uavs, actions):
        assert action.command == "Moving" and action.points is not None
        assert action.points[0] < uav.position[0] - 50000
        assert action.points[2] > uav.position[2]


def test_uav_does_not_retarget_every_frame_when_contact_scores_change():
    uav = aircraft(z=6000)
    first, second = entity(20, 1, 0, 10000), entity(21, 1, 1, 12000)
    control = UAVControl()
    assert control.choose(state([uav], [first, second]), uav, None).points[0] == 10000
    second.position[0] = -9000
    action = control.choose(state([uav], [first, second], step=1), uav, None)
    assert action.points[0] == 10000
    for step in range(2, 7):
        uav.position[0] += 300
        action = control.choose(state([uav], step=step), uav, None)
        assert action.command == "Moving"  # Bounded memory of the old report.
    assert control.choose(state([uav], step=8), uav, None).command == "Guard"


def test_low_uav_local_target_can_start_strike_immediately():
    uav = aircraft(z=6000)
    enemy = entity(20, 1, 0, 1000)
    enemy.position[2] = 6000
    uav.raw["agentPerception"] = [20]
    assert UAVControl().choose(state([uav], [enemy]), uav, None).command == "SelfDestruct"
