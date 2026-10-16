"""Public-state strategy tests; synthetic positions never model battle scores."""
from dataclasses import replace

import pytest

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from test_fslutk_rule import entity, guards_at, map_objectives, state


def opening(*, approaching=False):
    x, y = (-6470, 30114) if approaching else (60000, 12000)
    units = [entity(i, 0, i, x, y) for i in range(12)]
    return units, map_objectives(), AssaultRoute(adaptive=True)


def advance(route, units, objects, steps, enemies=()):
    for step in steps:
        current = state(units, enemies, objects, step=step)
        route.update(current)
    return current


def test_concentrated_hq_defence_changes_only_after_consecutive_evidence():
    units, objects, route = opening(approaching=True)
    objects[0].raw['blueteamNum'] = 14
    for step in (0, 1):
        s = advance(route, units, objects, [step])
        assert route.objective(s).uid == objects[0].uid
    s = advance(route, units, objects, [2])
    assert route.objective(s).uid in {objects[1].uid, objects[2].uid}
    assert route.hq_uid == objects[0].uid
    assert route.is_hq(objects[0]) and not route.is_hq(route.objective(s))


def test_split_ab_defence_preserves_high_value_hq_and_original_road():
    units, objects, route = opening()
    objects[1].raw['blueteamNum'] = objects[2].raw['blueteamNum'] = 8
    for step in range(20):
        s = advance(route, units, objects, [step])
        assert route.objective(s).uid == objects[0].uid
        assert route.point(s, units[0]) == route.LOWER[1]
    assert not route._replanned


def test_missing_enemy_data_is_unknown_and_does_not_stop_approach():
    units, objects, route = opening()
    for obj in objects:
        obj.raw.pop('blueteamNum')
    for step in range(80):
        s = advance(route, units, objects, [step])
        assert route.objective(s).uid == objects[0].uid
        assert route.point(s, units[0]) == route.LOWER[1]
    assert route.recon_pending(s)
    assert route._strategy.pressure(s, route, objects[0]) > 0


@pytest.mark.parametrize('count', [None, 'unavailable', -1, float('nan')])
def test_invalid_occupant_count_does_not_prove_empty(count):
    units, objects, route = opening()
    objects[0].raw['blueteamNum'] = count
    s = advance(route, units, objects, [0])
    assert route._strategy.occupants(s, objects[0]) is None
    assert route.recon_pending(s)


def test_public_perimeter_contacts_override_zero_box_count():
    units, objects, route = opening(approaching=True)
    enemies = [entity(100 + i, 1, i, objects[0].position[0] + 9000,
                      objects[0].position[1]) for i in range(12)]
    s = advance(route, units, objects, range(3), enemies)
    assert route.objective(s).uid != objects[0].uid
    # A moment of lost sight does not instantly make the perimeter empty.
    s = advance(route, units, objects, [3])
    assert route._strategy.pressure(s, route, objects[0]) == 12
    s = advance(route, units, objects, [20])
    assert route._strategy.pressure(s, route, objects[0]) == 0


def test_hidden_raw_state_does_not_change_decision():
    units, objects, route = opening()
    for step in range(3):
        s = replace(state(units, objectives=objects, step=step), raw={
            'enemies': [{'uId': 900, 'agentLocation': list(objects[0].position), 'agentHp': 100}],
            'blueteamNum': 1000,
        })
        route.update(s)
    assert route.objective(s).uid == objects[0].uid
    assert route._strategy.pressure(s, route, objects[0]) == 0


def test_observation_gap_restarts_candidate_confirmation():
    units, objects, route = opening(approaching=True)
    objects[0].raw['blueteamNum'] = 14
    s = advance(route, units, objects, [0, 1, 10, 11])
    assert route.objective(s).uid == objects[0].uid
    s = advance(route, units, objects, [12])
    assert route.objective(s).uid != objects[0].uid


def test_short_pressure_reversal_cannot_flip_committed_target():
    units, objects, route = opening(approaching=True)
    objects[0].raw['blueteamNum'] = 14
    s = advance(route, units, objects, range(3))
    chosen = route.objective(s)
    objects[0].raw['blueteamNum'] = 0
    chosen.raw['blueteamNum'] = 20
    s = advance(route, units, objects, range(3, 20))
    assert route.objective(s).uid == chosen.uid
    s = advance(route, units, objects, range(20, 35))
    assert route.objective(s).uid == objects[0].uid


def test_reordered_opening_keeps_lower_road_and_omits_enemy_hq_centre():
    units, objects, route = opening(approaching=True)
    units[0].position[:] = (60000, 12000, 0)  # Rear unit still joins the road.
    objects[0].raw['blueteamNum'] = 14
    s = advance(route, units, objects, range(3))
    assert route.point(s, units[0]) == route.LOWER[1]
    path = route._route_for(units[0], route.stage)
    assert tuple(objects[0].position) not in path
    assert path[-1] == tuple(route.objective(s).position)
    assert all(point in (*route.LOWER, *route.HQ_B, *route.B_A) for point in path[:-1])


def test_replanning_near_hq_does_not_order_return_to_lower_road():
    units, objects, route = opening()
    route.update(state(units, objectives=objects))
    route._reorder(state(units, objectives=objects), (objects[1].uid, objects[0].uid, objects[2].uid))
    units[0].position[:] = (objects[0].position[0] + 3000, objects[0].position[1], 0)
    route._reorder(state(units, objectives=objects), (objects[0].uid, objects[1].uid, objects[2].uid))
    assert route.point(state(units, objectives=objects), units[0]) == tuple(objects[0].position)


def test_confirmed_hq_releases_surplus_after_clear_observations():
    units, objects, route = opening()
    units = guards_at(objects[0], count=8)
    objects[0].raw.update(percent=1, redTeamNum=8)
    s = advance(route, units, objects, range(4))
    assert route.stage == 1 and len(route._garrison) == 4
    s = advance(route, units, objects, range(4, 8))
    assert route.stage == 1 and len(route._garrison) == 3
    assert all(route.inside_garrison(objects[0], unit.position) for unit in units
               if unit.uid in route._garrison)
    # A new public threat restores the normal reserve immediately.
    enemy = entity(999, 1, 0, objects[0].position[0] + 11000, objects[0].position[1])
    advance(route, units, objects, [8], [enemy])
    assert len(route._garrison) == 4


def test_after_hq_capture_selects_weaker_secondary_without_recalling_guards():
    units, objects, route = opening()
    units = guards_at(objects[0], count=8)
    objects[0].raw.update(percent=1, redTeamNum=8)
    objects[1].raw['blueteamNum'] = 12
    s = advance(route, units, objects, range(4))
    assert route.stage == 1 and route.objective(s).uid == objects[2].uid
    assert set(route._garrison.values()) == {objects[0].uid}


def test_lost_hq_does_not_force_remaining_column_back_into_stronghold():
    units, objects, route = opening()
    units = guards_at(objects[0], count=8)
    objects[0].raw.update(percent=1, redTeamNum=8)
    advance(route, units, objects, range(4))
    objects[0].raw.update(percent=.5, redTeamNum=0, blueteamNum=20)
    s = advance(route, units, objects, [4, 5])
    assert route.objective(s).uid != objects[0].uid
    assert objects[0].uid not in route._garrison.values()
    assert set(route.order) == {obj.uid for obj in objects}
    assert len(route.order) == 3 and route.hq_uid == objects[0].uid


def test_scout_keeps_destination_until_arrival_then_checks_other_objectives():
    units, objects, route = opening()
    scout = entity(1000, 0, 12, 60000, 12000, kind='BP_Scout_UAV_C')
    first = None
    for step in range(40):
        s = advance(route, units, objects, [step])
        chosen = route.recon_objective(s, scout)
        first = first or chosen.uid
        assert chosen.uid == first
    scout.position[:] = chosen.position
    for step in range(40, 43):
        s = advance(route, units, objects, [step])
        chosen = route.recon_objective(s, scout)
    assert chosen.uid != first
    # Perimeter proximity schedules the next survey; it never creates an enemy.
    assert not route._strategy._contacts


def test_episode_reset_clears_old_contacts_and_objective_commitment():
    units, objects, route = opening(approaching=True)
    objects[0].raw['blueteamNum'] = 14
    advance(route, units, objects, range(3))
    objects[0].raw['blueteamNum'] = 0
    s = state(units, objectives=objects, step=0, episode=2)
    route.update(s)
    assert route.order == tuple(obj.uid for obj in objects)
    assert not route._replanned and not route._strategy._contacts


def test_identical_spawn_deployment_is_not_treated_as_final_blue_strategy():
    units, objects, route = opening()
    objects[0].raw['blueteamNum'] = 16
    s = advance(route, units, objects, range(80))
    assert route.objective(s).uid == objects[0].uid
    assert not route._replanned
    for unit in units:
        unit.position[:] = (-6470, 30114, 0)
    objects[0].raw['blueteamNum'] = 0
    objects[1].raw['blueteamNum'] = objects[2].raw['blueteamNum'] = 10
    s = advance(route, units, objects, range(80, 84))
    assert route.objective(s).uid == objects[0].uid
    assert not route._replanned


def test_real_forward_uav_contact_allows_early_assessment():
    units, objects, route = opening()
    scout = entity(1000, 0, 12, objects[0].position[0] + 10000,
                   objects[0].position[1], kind='BP_Base_UAV_C')
    enemies = [entity(2000+i, 1, i, objects[0].position[0], objects[0].position[1]) for i in range(12)]
    scout.raw['agentPerception'] = [enemy.uid for enemy in enemies]
    s = advance(route, [*units, scout], objects, range(3), enemies)
    assert route.objective(s).uid != objects[0].uid
