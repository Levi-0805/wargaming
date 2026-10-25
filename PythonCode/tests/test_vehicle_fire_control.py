import math

import pytest

from cssim.protocol import Action
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.VehicleFireControl import VehicleFireControl
from test_fslutk_rule import entity, state


def case(kind="BP_MNWS_Vehicle_Armored_C", distance=800):
    car = entity(1, 0, 0, 0, kind=kind)
    enemy = entity(20, 1, 0, distance)
    car.raw.update(agentPerception=[enemy.uid], interaction=["200"], rSVD1="vehicle;1")
    return car, enemy, VehicleFireControl()


@pytest.mark.parametrize("kind", sorted(VehicleFireControl.VEHICLES))
def test_vehicle_aims_with_guard_before_uid_fire_and_encodes_supported_protocol(kind):
    car, enemy, control = case(kind)
    aim = control.choose(state([car], [enemy]), car)
    assert aim.to_ue_string(car.uid, [car, enemy]) == "ActionSet2::Guard;+X;1"
    fire = control.choose(state([car], [enemy], step=1), car)
    assert fire.to_ue_string(car.uid, [car, enemy]) == "ActionSet2::NormalAttacking;20;1"


@pytest.mark.parametrize("reported_range", [None, 0, -1, "bad", math.nan, math.inf])
def test_invalid_or_missing_range_never_infers_range_from_perception(reported_range):
    car, enemy, control = case()
    car.properties["fireRange"] = reported_range
    car.properties["perceptionRange"] = 20000
    assert control.choose(state([car], [enemy]), car) is None


def test_out_of_range_contact_does_not_interrupt_patrol_and_range_is_three_dimensional():
    car, enemy, control = case(distance=2001)
    car.properties["fireRange"] = 2000
    assert control.choose(state([car], [enemy]), car) is None
    enemy.position[0], enemy.position[2] = 1600, 1400
    assert control.choose(state([car], [enemy], step=1), car) is None
    enemy.position[2] = 1200  # Exactly on a non-placeholder reported boundary.
    assert control.choose(state([car], [enemy], step=2), car).command == "Guard"


def test_known_generic_range_does_not_reintroduce_ten_metre_never_fire_regression():
    car, enemy, control = case(distance=9500)
    assert car.properties["fireRange"] == 1000
    assert control.choose(state([car], [enemy]), car).command == "Guard"
    assert control.choose(state([car], [enemy], step=1), car).command == "NormalAttacking"
    enemy.position[0] = 15001
    assert control.choose(state([car], [enemy], step=2), car) is None


def test_placeholder_envelope_includes_recorded_vehicle_kill_distances_but_not_full_sight_range():
    car, enemy, control = case(distance=13400)
    car.properties['perceptionRange'] = 20000
    assert control.choose(state([car], [enemy]), car).command == 'Guard'
    assert control.choose(state([car], [enemy], step=1), car).command == 'NormalAttacking'
    enemy.position[0] = 18000
    assert control.choose(state([car], [enemy], step=2), car) is None


def test_locally_seen_lynx_has_priority_over_wheeled_armor_and_infantry():
    car, soldier, control = case()
    wheeled = entity(30,1,1,500,kind='BP_MNWS_Vehicle_Armored_C',hp=20)
    lynx = entity(31,1,2,5000,kind='BP_MNWS_Vehicle_6x6UGV_C')
    car.raw['agentPerception']=[soldier.uid,wheeled.uid,lynx.uid]
    enemies=[soldier,wheeled,lynx]
    control.choose(state([car],enemies),car)
    assert control.choose(state([car],enemies,step=1),car).target_uid==lynx.uid


@pytest.mark.parametrize("invalid", ["shared", "missing_local", "dead", "zero_hp", "friendly", "air", "empty", "disconnected"])
def test_invalid_contact_never_causes_explicit_vehicle_fire(invalid):
    car, enemy, control = case()
    if invalid == "shared":
        car.raw["agentPerception"] = []
    elif invalid == "missing_local":
        car.raw.pop("agentPerception")
    elif invalid == "dead":
        enemy.alive = False
    elif invalid == "zero_hp":
        enemy.hp = 0
    elif invalid == "friendly":
        enemy.team = 0
    elif invalid == "air":
        enemy.entity_type = "BP_Base_UAV_C"
    elif invalid == "empty":
        car.raw["interaction"] = ["0"]
    else:
        car.raw["rSVD1"] = "vehicle;0"
    assert control.choose(state([car], [enemy]), car) is None


def test_no_damage_burst_stops_even_when_rounds_are_spent_and_cooldown_allows_travel():
    car, enemy, control = case()
    actions = []
    for step in range(6):
        car.raw["interaction"] = [str(200 - 15 * step)]
        actions.append(control.choose(state([car], [enemy], step=step), car))
    assert [a.command if a else None for a in actions] == [
        "Guard", "NormalAttacking", "NormalAttacking", "Guard", None, None]
    assert control.choose(state([car], [enemy], step=10), car) is None
    assert control.choose(state([car], [enemy], step=11), car).command == "Guard"


def test_confirmed_damage_with_ammo_spent_keeps_useful_fire_but_damage_alone_does_not():
    car, enemy, control = case()
    for step in range(6):
        car.raw["interaction"] = [str(200 - 10 * step)]
        enemy.hp = 100 - step * 5
        action = control.choose(state([car], [enemy], step=step), car)
        assert action.command == ("Guard" if step == 0 else "NormalAttacking")
    car, enemy, control = case()
    for step in range(4):
        enemy.hp -= 5  # Damage by a teammate must not justify our unchanged ammo.
        action = control.choose(state([car], [enemy], step=step), car)
    assert action.command == "Guard"


def test_hidden_or_out_of_range_enemy_vehicle_cannot_displace_local_in_range_infantry():
    car, enemy, control = case()
    armor = entity(30, 1, 1, 18000, kind="BP_MNWS_Vehicle_6x6UGV_C")
    car.raw["agentPerception"] = [enemy.uid, armor.uid]
    control.choose(state([car], [enemy, armor]), car)
    assert control.choose(state([car], [enemy, armor], step=1), car).target_uid == enemy.uid


def test_different_target_requires_fresh_aim_and_same_frame_calls_do_not_consume_burst():
    car, enemy, control = case()
    frame = state([car], [enemy])
    assert all(control.choose(frame, car).command == "Guard" for _ in range(5))
    frame = state([car], [enemy], step=1)
    assert all(control.choose(frame, car).command == "NormalAttacking" for _ in range(5))
    assert control.choose(state([car], [enemy], step=2), car).command == "NormalAttacking"
    enemy.uid = 25
    car.raw["agentPerception"] = [25]
    assert control.choose(state([car], [enemy], step=3), car).command == "Guard"


@pytest.mark.parametrize("kind", sorted(VehicleFireControl.VEHICLES))
@pytest.mark.parametrize("contested_b", [False, True])
def test_same_direction_priority_change_reuses_aim_only_in_contested_b(kind, contested_b):
    car, soldier, control = case(kind)
    soldier.position[:2] = (0, -5000)
    lynx = entity(30, 1, 1, 100, -6000, kind="BP_MNWS_Vehicle_6x6UGV_C")
    enemies = [soldier, lynx]
    first = control.choose(state([car], enemies), car, advance_on_stall=contested_b)
    assert first.command == "Guard" and first.direction == "-Y"
    car.raw["agentPerception"].append(lynx.uid)
    second = control.choose(state([car], enemies, step=1), car, advance_on_stall=contested_b)
    assert second.command == ("NormalAttacking" if contested_b else "Guard")
    if contested_b:
        assert second.target_uid == lynx.uid
        # Repeated decisions cannot spend additional shots in the same frame.
        assert control.choose(state([car], enemies, step=1), car,
                              advance_on_stall=True) is second
        assert control._shots[(car.uid, lynx.uid)]["count"] == 1


@pytest.mark.parametrize("break_aim", ["direction", "gap", "contact_lost", "position_guard"])
def test_contested_b_target_change_requires_new_aim_when_direction_not_preserved(break_aim):
    car, soldier, control = case()
    lynx = entity(30, 1, 1, 900, kind="BP_MNWS_Vehicle_6x6UGV_C")
    if break_aim == "position_guard":
        soldier.position[:] = car.position
        lynx.position[:] = car.position
    control.choose(state([car], [soldier, lynx]), car, advance_on_stall=True)
    step = 1
    if break_aim == "direction":
        lynx.position[:2] = (0, 900)
    elif break_aim == "gap":
        step = 2
    elif break_aim == "contact_lost":
        car.raw["agentPerception"] = []
        assert control.choose(state([car], [soldier, lynx], step=1), car,
                              advance_on_stall=True) is None
        step = 2
    car.raw["agentPerception"] = [lynx.uid]
    action = control.choose(state([car], [soldier, lynx], step=step), car,
                            advance_on_stall=True)
    assert action.command == "Guard"


@pytest.mark.parametrize("invalid", ["shared", "out_of_range", "dead", "friendly", "air", "empty", "disconnected"])
def test_reusing_b_aim_never_bypasses_current_target_eligibility(invalid):
    car, soldier, control = case()
    target = entity(30, 1, 1, 900, kind="BP_MNWS_Vehicle_6x6UGV_C")
    control.choose(state([car], [soldier, target]), car, advance_on_stall=True)
    car.raw["agentPerception"] = [target.uid]
    if invalid == "shared":
        car.raw["agentPerception"] = []
    elif invalid == "out_of_range":
        target.position[0] = 15001
    elif invalid == "dead":
        target.alive = False
    elif invalid == "friendly":
        target.team = car.team
    elif invalid == "air":
        target.entity_type = "BP_Base_UAV_C"
    elif invalid == "empty":
        car.raw["interaction"] = ["0"]
    else:
        car.raw["rSVD1"] = "vehicle;0"
    assert control.choose(state([car], [soldier, target], step=1), car,
                          advance_on_stall=True) is None


def test_loss_of_local_contact_or_range_cancels_fire_and_new_episode_clears_burst():
    car, enemy, control = case()
    for step in range(3):
        control.choose(state([car], [enemy], step=step), car)
    car.raw["agentPerception"] = []
    assert control.choose(state([car], [enemy], step=3), car) is None
    car.raw["agentPerception"] = [enemy.uid]
    assert control.choose(state([car], [enemy], episode=2), car).command == "Guard"
    assert control.choose(state([car], [enemy], step=1, episode=2), car).command == "NormalAttacking"
    enemy.position[0] = 15001
    assert control.choose(state([car], [enemy], step=2, episode=2), car) is None


def test_patrol_converts_waypoints_but_preserves_escape_and_nonmovement_actions():
    path = Action.move_to(((10, 20, 30), (40, 50, 60)))
    assert VehicleFireControl.patrol(path) == Action.guard_at(path.points)
    for action in (Action.move("-X+Y"), Action.attack(20), Action.guard("+X"), Action.idle(), Action.move("+Z")):
        assert VehicleFireControl.patrol(action) is action


@pytest.mark.parametrize("kind", sorted(VehicleFireControl.VEHICLES))
def test_blocked_vehicle_escape_reaches_wire_then_resumes_guard_patrol(kind):
    from test_fslutk_rule import algorithm, objective

    car = entity(1, 0, 0, 0, kind=kind)
    objects = [objective(901, 100000, name='HQ')]
    rule = algorithm([car])
    road = Action.move_at((10000, 0, 0))
    rule._combat_action = lambda s, unit: road
    actions = [rule.choose_action(state([car], objectives=objects, step=step), car)
               for step in range(14)]
    assert actions[0] == Action.guard_at(road.points)
    # Keep position fixed to reproduce an obstacle. Both Lynx and armored
    # vehicles must serialize a real movement escape, not Guard aiming.
    for action in actions[8:10]:
        assert action.command == 'Moving' and action.points is None
        assert action.to_ue_string(car.uid, [car]) == f'ActionSet2::Moving;{action.direction};1'
    resumed = 13 if kind == 'BP_MNWS_Vehicle_Armored_C' else 10
    assert actions[resumed] == Action.guard_at(road.points)
