"""Regressions for units 481/486/186/196 in 20261013_202842 episode 1."""
from cssim.protocol import Action
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.MotionRecovery import MotionRecovery
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from test_fslutk_rule import entity, state, map_objectives


def stuck_car():
    car = entity(481, 0, 0, 33934.21, 31598.87, kind="BP_MNWS_Vehicle_Armored_C")
    car.position[2] = 6335.344
    car.yaw = -28.48333
    car.raw['rSVD1'] = 'car;1'
    return car


def test_turn_correction_cannot_starve_reverse_and_side_recovery():
    car = stuck_car()
    recovery = MotionRecovery()
    goal = Action.move_at(AssaultRoute.LOWER[5])
    actions = [recovery.apply(state([car], step=step), car, goal) for step in range(14)]
    assert all(a.command == 'Guard' for a in actions[:2])
    assert all(a.points == goal.points for a in actions[2:8])
    assert actions[8].points is None and actions[9].points is None
    assert actions[10].points == actions[11].points == actions[12].points
    assert actions[10].points != goal.points
    assert actions[13] is goal  # Always return to the requested route.


def test_jittering_goal_and_interleaved_fire_preserve_failed_move_evidence():
    car = stuck_car()
    recovery = MotionRecovery()
    escapes = []
    for step in range(22):
        if step % 2:
            action = Action.attack(20)
            assert recovery.apply(state([car], step=step), car, action) is action
        else:
            goal = Action.move_at((car.position[0] - 800, car.position[1] + (step % 3) * 800, car.position[2]))
            result = recovery.apply(state([car], step=step), car, goal)
            if result.command == 'Moving' and result.points is None:
                escapes.append(step)
    assert escapes and escapes[0] <= 16


def test_small_destination_jitter_does_not_cancel_active_vehicle_escape():
    car = stuck_car()
    recovery = MotionRecovery()
    goal = Action.move_at(AssaultRoute.LOWER[5])
    for step in range(9):
        first = recovery.apply(state([car], step=step), car, goal)
    jitter = Action.move_at((goal.points[0] + 50, goal.points[1], goal.points[2]))
    assert recovery.apply(state([car], step=9), car, jitter).direction == first.direction
    assert recovery.apply(state([car], step=10), car, jitter).points != jitter.points


def test_successful_fast_vehicle_motion_is_not_interrupted_by_yaw_correction():
    car = stuck_car()
    recovery = MotionRecovery()
    goal = Action.move_at((20000, 31598, 6335))
    recovery.apply(state([car]), car, goal)
    for step in range(1, 8):
        car.position[0] -= 800
        assert recovery.apply(state([car], step=step), car, goal) is goal


def test_stationary_guarding_does_not_trigger_escape_on_first_move():
    car = stuck_car()
    recovery = MotionRecovery()
    for step in range(20):
        guard = Action.guard_position(car.position)
        assert recovery.apply(state([car], step=step), car, guard) is guard
    car.yaw = 180
    move = Action.move_at((20000, car.position[1], car.position[2]))
    assert recovery.apply(state([car], step=20), car, move) is move


def test_link_loss_clears_previous_stationary_vehicle_evidence():
    car = stuck_car()
    recovery = MotionRecovery()
    move = Action.move_at(AssaultRoute.LOWER[5])
    for step in range(7):
        recovery.apply(state([car], step=step), car, move)
    car.raw['rSVD1'] = 'car;0'
    assert recovery.apply(state([car], step=7), car, move) is move
    car.raw['rSVD1'] = 'car;1'
    car.yaw = 180
    assert recovery.apply(state([car], step=8), car, move) is move


def transit(unit):
    route = AssaultRoute()
    objects = map_objectives()
    route.update(state([unit], objectives=objects))
    route._cursor[unit.uid] = (0, 1)
    return route, objects


def test_recorded_infantry_loop_eventually_bypasses_only_the_near_transit_point():
    soldier = entity(186, 0, 0, 65700, 34415)
    route, objects = transit(soldier)
    points = []
    for step in range(26):
        soldier.position[:2] = (65700, 34415) if step % 2 else (66160.63, 34736.86)
        s = state([soldier], objectives=objects, step=step)
        route.update(s)
        points.append(route.point(s, soldier))
    assert points[0] == AssaultRoute.LOWER[1]
    assert points[-1] == AssaultRoute.LOWER[2]
    assert route.stage == 0


def test_car_near_a_later_road_sample_can_rejoin_without_returning_to_old_one():
    car = entity(486, 0, 0, 59531, 34431, kind='BP_MNWS_Vehicle_Armored_C')
    route, objects = transit(car)
    for step in range(22):
        s = state([car], objectives=objects, step=step)
        route.update(s)
        point = route.point(s, car)
    assert point == AssaultRoute.LOWER[3]


def test_nearby_objective_is_never_skipped_by_stall_bypass():
    soldier = entity(186, 0, 0, -41810, 34275)
    route, objects = transit(soldier)
    route._cursor[soldier.uid] = (0, len(route._routes[0]) - 1)
    for step in range(50):
        s = state([soldier], objectives=objects, step=step)
        route.update(s)
        assert route.point(s, soldier) == tuple(objects[0].position)
        assert route.stage == 0


def test_distant_road_point_is_not_skipped_when_unit_stalls_off_route():
    soldier = entity(186, 0, 0, 66000, 20000)
    route, objects = transit(soldier)
    for step in range(45):
        s = state([soldier], objectives=objects, step=step)
        route.update(s)
        assert route.point(s, soldier) == AssaultRoute.LOWER[1]


def test_long_combat_gap_does_not_count_as_failed_waypoint_approaches():
    soldier = entity(186, 0, 0, 65700, 34415)
    route, objects = transit(soldier)
    for step in [*range(15), *range(100, 110)]:
        s = state([soldier], objectives=objects, step=step)
        route.update(s)
        assert route.point(s, soldier) == AssaultRoute.LOWER[1]


def test_offline_unit_does_not_skip_waypoints_while_link_is_down():
    car = entity(486, 0, 0, 59531, 34431, kind='BP_MNWS_Vehicle_Armored_C')
    car.raw['rSVD1'] = 'car;0'
    route, objects = transit(car)
    for step in range(30):
        s = state([car], objectives=objects, step=step)
        route.update(s)
        assert route.point(s, car) == AssaultRoute.LOWER[1]
