from cssim.protocol import Action
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.MotionRecovery import MotionRecovery
from test_command_links import unit, view


def test_repeated_move_without_displacement_changes_to_direction_control():
    soldier = unit(1)
    recovery = MotionRecovery()
    action = Action.move_at((20_000, 0, 0))
    for step in range(9):
        result = recovery.apply(view(soldier, step=step), soldier, action)
    assert result.command == "Moving"
    assert result.direction == "+X"
    assert result.points is None
    for step in range(9, 17):
        result = recovery.apply(view(soldier, step=step), soldier, action)
    assert result.direction == "+X+Y"


def test_deliberate_waiting_and_firing_are_not_movement_failures():
    soldier = unit(1)
    recovery = MotionRecovery()
    guard = Action.guard_position(tuple(soldier.position))
    attack = Action.attack(99)
    for step in range(20):
        action = guard if step % 2 else attack
        assert recovery.apply(view(soldier, step=step), soldier, action) is action


def test_ground_steps_preserve_elevation_and_forward_progress():
    dog = unit(1, z=7500, kind="BP_RoboDog_C")
    result = MotionRecovery().apply(view(dog), dog, Action.move_at((20_000, 0, 2000)))
    assert result.points == (2000, 0, 7500)


def test_actual_progress_does_not_trigger_an_escape():
    soldier = unit(1)
    recovery = MotionRecovery()
    for step in range(20):
        soldier.position[0] = step * 100
        result = recovery.apply(view(soldier, step=step), soldier, Action.move_at((20_000, 0, 0)))
        assert result.points is not None


def test_disconnected_unit_and_bomb_action_are_not_redirected():
    uav = unit(1, kind="BP_Base_UAV_C", comm=0)
    recovery = MotionRecovery()
    move = Action.move_at((20_000, 0, 2000))
    for step in range(12):
        assert recovery.apply(view(uav, step=step), uav, move) is move
    bomb = Action.self_destruct((2000, 0, 0))
    assert recovery.apply(view(uav, step=12), uav, bomb) is bomb


def test_new_episode_clears_old_escape_state():
    soldier = unit(1)
    recovery = MotionRecovery()
    move = Action.move_at((20_000, 0, 0))
    for step in range(9):
        recovery.apply(view(soldier, step=step), soldier, move)
    assert recovery.apply(view(soldier, episode=2), soldier, move).points is not None
