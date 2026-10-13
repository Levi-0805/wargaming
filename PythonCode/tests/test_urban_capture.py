"""HQ courtyard regression cases from 20261014_010705 public observations."""
import pytest
from dataclasses import replace

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.UrbanContact import UrbanContact
from test_fslutk_rule import algorithm, entity, guards_at, map_objectives, state


def test_shared_contact_rotation_cannot_produce_endless_attack_requests():
    soldier = entity(1, 0, 0, 0)
    soldier.raw.update(agentPerception=[], interaction=["250"])
    enemies = [entity(20, 1, 0, 4000), entity(21, 1, 1, 5000)]
    rule = algorithm([soldier])
    actions = []
    for step in range(24):
        enemies[0].hp, enemies[1].hp = ((100, 90) if step % 2 else (90, 100))
        actions += rule.act(state([soldier], enemies, step=step))
    assert [i for i, action in enumerate(actions) if action.command == "NormalAttacking"] == [0, 8, 16]
    assert all(action.command in {"NormalAttacking", "Moving", "Guard"} for action in actions)


@pytest.mark.parametrize("spent,damage,expected", [(True, True, "NormalAttacking"),
                                                  (True, False, "Moving"),
                                                  (False, True, "Moving")])
def test_shared_trial_requires_ammunition_and_damage_feedback(spent, damage, expected):
    soldier, enemy = entity(1, 0, 0, 0), entity(20, 1, 0, 4000)
    soldier.raw.update(agentPerception=[], interaction=["250"])
    rule = algorithm([soldier])
    assert rule.act(state([soldier], [enemy]))[0].command == "NormalAttacking"
    soldier.raw["interaction"] = ["249" if spent else "250"]
    enemy.hp = 90 if damage else 100
    assert rule.act(state([soldier], [enemy], step=1))[0].command == expected


def test_local_detection_interrupts_a_shared_contact_maneuver():
    soldier = entity(1, 0, 0, 0)
    soldier.raw.update(agentPerception=[], interaction=["250"])
    shared, local = entity(20, 1, 0, 4000), entity(21, 1, 1, 3000)
    rule = algorithm([soldier])
    rule.act(state([soldier], [shared]))
    assert rule.act(state([soldier], [shared], step=1))[0].command == "Moving"
    soldier.raw["agentPerception"] = [local.uid]
    action = rule.act(state([soldier], [shared, local], step=2))[0]
    assert action.command == "NormalAttacking" and action.target_uid == local.uid


def test_ineffective_local_contact_seeks_a_view_instead_of_cycling_shared_targets():
    soldier = entity(1, 0, 0, 0)
    blocked, shared = entity(20, 1, 0, 4000), entity(21, 1, 1, 5000)
    soldier.raw["agentPerception"] = [20]
    rule = algorithm([soldier])
    for step in range(12):
        soldier.raw["interaction"] = [str(250 - step)]
        action = rule.act(state([soldier], [blocked, shared], step=step))[0]
        assert action.target_uid != shared.uid
        if step >= 8:
            assert action.command in {"Moving", "Guard"}


def test_fixed_maneuver_arrives_scans_briefly_then_resumes_search():
    soldier, enemy = entity(1, 0, 0, 0), entity(20, 1, 0, 4000)
    contact = UrbanContact()
    first = contact.reposition(state([soldier], [enemy]), soldier, enemy)
    soldier.position[0] += 500
    assert contact.reposition(state([soldier], [enemy], step=1), soldier, enemy).points == first.points
    soldier.position[:] = first.points
    for step in (2, 3):
        action = contact.reposition(state([soldier], [enemy], step=step), soldier, enemy)
        assert action.command == "Guard" and action.points is None
    action = contact.reposition(state([soldier], [enemy], step=4), soldier, enemy)
    assert action.command == "Moving" and action.points != first.points


def test_reached_observer_positions_have_bounded_reservations_and_ground_height():
    blocked = [entity(i + 1, 0, i, 0) for i in range(3)]
    observer = entity(10, 0, 10, 2000, 1000)
    flyer = entity(11, 0, 11, 1300, kind="BP_Base_UAV_C")
    high = entity(12, 0, 12, 1500)
    high.position[2] = 1000
    enemy = entity(20, 1, 0, 4000)
    for unit in blocked:
        unit.raw["agentPerception"] = []
    for unit in (observer, flyer, high):
        unit.raw["agentPerception"] = [enemy.uid]
    contact = UrbanContact()
    current = state([*blocked, observer, flyer, high], [enemy])
    actions = [contact.reposition(current, unit, enemy) for unit in blocked]
    assert sum(action.points == tuple(observer.position) for action in actions) == 2
    assert all(action.points[2] == 0 for action in actions)
    assert actions[-1].points not in (tuple(flyer.position), tuple(high.position))


def test_occupied_hq_is_searched_without_inventing_an_enemy_target():
    obj = map_objectives()[0]
    obj.raw.update(scale={"x": 7700, "y": 6500}, percent=1, redTeamNum=6, blueteamNum=2)
    guards = guards_at(obj, count=6)
    guards[1].position[0] += 2500  # A reached courtyard position.
    for unit in guards:
        unit.raw["agentPerception"] = []
    rule = algorithm(guards)
    actions = rule.act(state(guards, objectives=[obj]))
    assert all(action.command == "Moving" and action.target_uid is None for action in actions)
    assert any(action.points == tuple(guards[1].position) for action in actions)
    for unit, action in zip(guards, actions):
        if unit.uid in rule._route._garrison:
            assert rule._route.inside_garrison(obj, action.points)


def test_guard_maneuver_clamps_plan_and_can_finish_scanning_inside_point():
    obj = map_objectives()[0]
    soldier = guards_at(obj, count=1)[0]
    soldier.position[0] += 2000
    enemy = entity(20, 1, 0, obj.position[0], obj.position[1] + 7000)
    route, contact = AssaultRoute(), UrbanContact()
    current = state([soldier], [enemy])
    move = contact.constrain(soldier, contact.reposition(current, soldier, enemy), obj, route)
    assert route.inside_garrison(obj, move.points)
    soldier.position[:] = move.points
    scan = contact.constrain(soldier, contact.reposition(state([soldier], [enemy], step=1), soldier, enemy), obj, route)
    assert scan.command == "Guard" and scan.points is None


def test_hq_requires_four_consecutive_clear_and_physically_held_frames():
    objectives = map_objectives()
    obj = objectives[0]
    obj.raw.update(percent=1, redTeamNum=6, blueteamNum=2)
    units = guards_at(obj, count=6)
    rule = algorithm(units)
    for step in range(6):
        actions = rule.act(state(units, objectives=objectives, step=step))
        assert rule._route.stage == 0
        assert all(action.command != "Idle" for action in actions)
    # Three clear samples followed by a returning occupant restart confirmation.
    for step, blue in ((6, 0), (7, 0), (8, 0), (9, 1), (10, 0), (11, 0), (12, 0)):
        obj.raw["blueteamNum"] = blue
        rule.act(state(units, objectives=objectives, step=step))
        assert rule._route.stage == 0
    actions = rule.act(state(units, objectives=objectives, step=13))
    assert rule._route.stage == 1 and len(rule._route._garrison) == 4
    assert all(action.command == "Moving" for unit, action in zip(units, actions)
               if unit.uid not in rule._route._garrison)


def test_clear_hq_with_outside_guards_does_not_accumulate_departure_confirmation():
    objectives = map_objectives()
    obj = objectives[0]
    obj.raw.update(percent=1, redTeamNum=4, blueteamNum=0)
    units = guards_at(obj)
    units[-1].position[0] += 7000
    route = AssaultRoute()
    for step in range(6):
        route.update(state(units, objectives=objectives, step=step))
    units[-1].position[:] = obj.position
    for step in range(6, 9):
        route.update(state(units, objectives=objectives, step=step))
        assert route.stage == 0
    route.update(state(units, objectives=objectives, step=9))
    assert route.stage == 1


@pytest.mark.parametrize("occupants", [None, "bad", -1])
def test_owner_label_does_not_override_partial_capture_or_missing_occupants(occupants):
    objectives = map_objectives()
    obj = objectives[0] = replace(objectives[0], team=0)
    obj.raw.update(percent=.5, redTeamNum=4)
    units, route = guards_at(obj), AssaultRoute()
    for step in range(4):
        route.update(state(units, objectives=objectives, step=step))
    assert route.stage == 0
    obj.raw["percent"] = 1
    obj.raw["blueteamNum"] = occupants
    for step in range(4, 8):
        route.update(state(units, objectives=objectives, step=step))
    assert route.stage == 0


def test_episode_reset_removes_shared_trial_cooldown_and_maneuver():
    soldier, enemy = entity(1, 0, 0, 0), entity(20, 1, 0, 4000)
    soldier.raw["agentPerception"] = []
    rule = algorithm([soldier])
    rule.act(state([soldier], [enemy]))
    rule.act(state([soldier], [enemy], step=1))
    assert rule.act(state([soldier], [enemy], episode=2))[0].command == "NormalAttacking"


def test_uncontested_road_approach_is_not_replaced_by_hq_sideways_search():
    objectives = map_objectives()
    objectives[0].raw.update(redTeamNum=0, blueteamNum=12)
    soldier = entity(1, 0, 3, -29782, 25170)
    soldier.raw["agentPerception"] = []
    action = algorithm([soldier]).act(state([soldier], objectives=objectives))[0]
    assert action.command == "Moving" and action.points[0] < soldier.position[0] - 5000
