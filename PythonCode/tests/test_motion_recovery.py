from cssim.protocol import Action
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.MotionRecovery import MotionRecovery
from test_command_links import unit, view


def test_repeated_move_without_displacement_has_bounded_escape_then_retries_goal():
    soldier = unit(1)
    recovery = MotionRecovery()
    action = Action.move_at((20000, 0, 0))
    for step in range(9):
        result = recovery.apply(view(soldier, step=step), soldier, action)
    assert result.direction == "+X+Y"
    assert recovery.apply(view(soldier, step=9), soldier, action).direction == "+X+Y"
    assert recovery.apply(view(soldier, step=10), soldier, action) is action


def test_moving_in_circles_is_detected_as_no_net_progress():
    soldier = unit(1)
    recovery = MotionRecovery()
    action = Action.move_at((20000, 0, 0))
    for step in range(9):
        soldier.position[1] = 1000 if step % 2 else -1000
        result = recovery.apply(view(soldier, step=step), soldier, action)
    assert result.points is None  # 20m each step still made zero forward progress.


def test_deliberate_waiting_and_firing_are_not_movement_failures():
    soldier = unit(1)
    recovery = MotionRecovery()
    for step in range(20):
        action = Action.guard_position(soldier.position) if step % 2 else Action.attack(99)
        assert recovery.apply(view(soldier, step=step), soldier, action) is action


def test_ground_navigation_preserves_full_path_and_map_elevation():
    dog = unit(1, z=7500, kind="BP_RoboDog_C")
    path = Action.move_to([(20000, 0, 2000), (30000, 5000, 2500)])
    assert MotionRecovery().apply(view(dog), dog, path) is path


def test_actual_forward_progress_does_not_trigger_escape():
    soldier = unit(1)
    recovery = MotionRecovery()
    for step in range(20):
        soldier.position[0] = step * 100
        result = recovery.apply(view(soldier, step=step), soldier, Action.move_at((20000, 0, 0)))
        assert result.points is not None


def test_disconnected_unit_and_bomb_action_are_not_redirected():
    uav = unit(1, kind="BP_Base_UAV_C", comm=0)
    recovery = MotionRecovery()
    move = Action.move_at((20000, 0, 2000))
    for step in range(12):
        assert recovery.apply(view(uav, step=step), uav, move) is move
    bomb = Action.self_destruct((2000, 0, 0))
    assert recovery.apply(view(uav, step=12), uav, bomb) is bomb


def test_new_episode_clears_old_escape_state():
    soldier = unit(1)
    recovery = MotionRecovery()
    move = Action.move_at((20000, 0, 0))
    for step in range(9):
        recovery.apply(view(soldier, step=step), soldier, move)
    assert recovery.apply(view(soldier, episode=2), soldier, move) is move
