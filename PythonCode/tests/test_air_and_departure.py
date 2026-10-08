import pytest
from cssim.protocol import Action
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.UAVControl import UAVControl
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.MotionRecovery import MotionRecovery
from test_fslutk_rule import entity, state, algorithm, map_objectives


def aircraft(uid=1, x=0, y=0):
    result = entity(uid, 0, uid, x, y, kind="BP_Base_UAV_C")
    result.raw.update(rSVD1="uav;1", agentPerception=[])
    return result


def test_detected_ground_target_100m_away_starts_coordinate_strike():
    uav = aircraft()
    enemy = entity(20, 1, 0, 10000)
    uav.raw["agentPerception"] = [20]
    result = UAVControl().choose(state([uav], [enemy]), uav, None)
    assert result.command == "SelfDestruct" and result.points == (10000, 0, 0)


def test_strike_is_not_cancelled_by_movement_next_frame_or_brief_loss_of_contact():
    uav = aircraft()
    enemy = entity(20, 1, 0, 10000)
    uav.raw["agentPerception"] = [20]
    rule = algorithm([uav])
    first = rule.act(state([uav], [enemy]))[0]
    uav.raw["agentPerception"] = []
    for step in range(1, 12):
        result = rule.act(state([uav], step=step))[0]
        assert result == first
    assert rule.act(state([uav], step=12))[0].command != "SelfDestruct"


def test_second_aircraft_selects_another_cluster():
    uavs = [aircraft(1), aircraft(2)]
    enemies = [entity(20, 1, 0, 8000), entity(21, 1, 1, 8000, 4000)]
    for uav in uavs:
        uav.raw["agentPerception"] = [20, 21]
    actions = algorithm(uavs).act(state(uavs, enemies))
    assert all(action.command == "SelfDestruct" for action in actions)
    assert actions[0].points != actions[1].points


def test_ground_target_has_priority_over_air_target():
    uav = aircraft()
    enemy_uav = entity(20, 1, 0, 1000, kind="BP_Base_UAV_C")
    soldier = entity(21, 1, 1, 9000)
    uav.raw["agentPerception"] = [20, 21]
    action = UAVControl().choose(state([uav], [enemy_uav, soldier]), uav, None)
    assert action.points == (9000, 0, 0)


def test_shared_contact_guides_scout_but_does_not_authorize_strike():
    uav = aircraft()
    enemy = entity(20, 1, 0, 10000)
    action = UAVControl().choose(state([uav], [enemy]), uav, None)
    assert action.command == "Moving" and action.points == (10000, 0, 2500)


def test_surviving_aircraft_reports_unconfirmed_strike_and_tries_normal_attack():
    uav = aircraft()
    enemy = entity(20, 1, 0, 10000)
    uav.raw["agentPerception"] = [20]
    control = UAVControl()
    for step in range(12):
        assert control.choose(state([uav], [enemy], step=step), uav, None).command == "SelfDestruct"
    assert control.choose(state([uav], [enemy], step=12), uav, None).command == "NormalAttacking"


def test_disconnected_uav_cannot_start_strike():
    uav = aircraft()
    uav.raw.update(rSVD1="uav;0", agentPerception=[20])
    enemy = entity(20, 1, 0, 10000)
    assert UAVControl().choose(state([uav], [enemy]), uav, None).command != "SelfDestruct"


def test_aircraft_stagnation_corrects_toward_goal_without_random_sideways_rotation():
    uav = aircraft(x=60000)
    objectives = map_objectives()
    control = UAVControl()
    for step in range(7):
        action = control.choose(state([uav], objectives=objectives, step=step), uav, objectives[0])
    assert action.command == "Moving" and "-X" in action.direction
    assert "+X" not in action.direction and "-Y" not in action.direction


def test_new_episode_clears_old_strike_coordinates():
    uav = aircraft()
    uav.raw["agentPerception"] = [20]
    enemy = entity(20, 1, 0, 10000)
    control = UAVControl()
    control.choose(state([uav], [enemy]), uav, None)
    uav.raw["agentPerception"] = []
    assert control.choose(state([uav], episode=2), uav, None).command == "Guard"


def test_confirmed_capture_releases_main_force_despite_distant_hq_contacts():
    objectives = map_objectives()
    agents = [entity(i, 0, i, -41810 + i * 100, 32275) for i in range(6)]
    enemy = entity(20, 1, 0, -41810, 40000)
    for agent in agents:
        agent.raw["agentPerception"] = [20]
    rule = algorithm(agents)
    for step in range(3):
        rule.act(state(agents, [enemy], objectives, step))
    assert rule._route.stage == 0  # No occupation, no early departure.
    objectives[0].raw.update(percent=1, redTeamNum=6, blueteamNum=2)
    rule.act(state(agents, [enemy], objectives, 3))
    actions = rule.act(state(agents, [enemy], objectives, 4))
    assert rule._route.stage == 1
    assert sum(action.command == "Moving" for action in actions) == 3
    assert sum(action.command == "NormalAttacking" for action in actions) == 3
    # Clearing defenders releases two of the three garrison units as well.
    objectives[0].raw["blueteamNum"] = 0
    rule.act(state(agents, [], objectives, 5))
    assert len(rule._route._garrison) == 1


def test_single_capture_sample_does_not_release_main_force():
    agent = entity(1, 0, 0, -41810, 32275)
    objectives = map_objectives()
    rule = algorithm([agent])
    objectives[0].raw.update(percent=1, redTeamNum=1)
    rule.act(state([agent], objectives=objectives))
    objectives[0].raw["percent"] = 0
    rule.act(state([agent], objectives=objectives, step=1))
    assert rule._route.stage == 0


def test_wheeled_vehicle_backs_away_then_bypasses_and_retries_original_waypoint():
    car = entity(1, 0, 0, 0, kind="BP_MNWS_Vehicle_Armored_C")
    car.yaw = 180
    move = Action.move_at((-10000, 0, 0))
    recovery = MotionRecovery()
    for step in range(9):
        action = recovery.apply(state([car], step=step), car, move)
    assert action.direction == "+X"  # Opposite actual heading, not another turn into the wall.
    assert recovery.apply(state([car], step=9), car, move).direction == "+X"
    side = recovery.apply(state([car], step=10), car, move)
    assert side.points[0] > 0 and abs(side.points[1]) == pytest.approx(2500)
    assert recovery.apply(state([car], step=13), car, move) is move


def test_vehicle_recovery_never_overrides_attack_or_link_loss():
    car = entity(1, 0, 0, 0, kind="BP_MNWS_Vehicle_Armored_C")
    move = Action.move_at((-10000, 0, 0))
    recovery = MotionRecovery()
    for step in range(9):
        recovery.apply(state([car], step=step), car, move)
    attack = Action.attack(20)
    assert recovery.apply(state([car], step=9), car, attack) is attack
    car.raw["rSVD1"] = "car;0"
    assert recovery.apply(state([car], step=10), car, move) is move
