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


def test_patrol_preserves_waypoints_directions_and_nonmovement_actions():
    path = Action.move_to(((10, 20, 30), (40, 50, 60)))
    assert VehicleFireControl.patrol(path) == Action.guard_at(path.points)
    assert VehicleFireControl.patrol(Action.move("-X+Y")) == Action.guard("-X+Y")
    for action in (Action.attack(20), Action.guard("+X"), Action.idle(), Action.move("+Z")):
        assert VehicleFireControl.patrol(action) is action
