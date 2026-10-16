"""Reconnaissance survives failed fire and stale reports in the public view."""
from types import SimpleNamespace

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.UAVControl import UAVControl
from test_air_and_departure import aircraft
from test_fslutk_rule import entity, map_objectives, state


def test_missing_current_perception_list_cannot_become_a_range_only_fire_lock():
    uav = aircraft()
    uav.raw.pop('agentPerception')
    enemy = entity(900, 1, 0, 1000)
    control = UAVControl()
    for step in range(12):
        action = control.choose(state([uav], [enemy], step=step), uav, None)
        assert action.command not in {'SelfDestruct', 'NormalAttacking'}
    assert not control._observations


def test_ammunition_expenditure_without_damage_has_a_finite_trial_budget():
    uav = aircraft(z=15000)
    enemy = entity(900, 1, 0, 10000)
    uav.raw['agentPerception'] = [enemy.uid]
    control = UAVControl()
    commands = []
    for step in range(40):
        uav.raw['interaction'] = [str(100 - step)]
        commands.append(control.choose(state([uav], [enemy], step=step), uav, None).command)
    assert commands.count('NormalAttacking') == 6
    assert all(command != 'NormalAttacking' for command in commands[10:])
    assert 'Moving' in commands[10:]
    enemy.hp -= 10
    assert control.choose(state([uav], [enemy], step=40), uav, None).command == 'NormalAttacking'


def test_alternating_local_contacts_do_not_reset_failed_fire_trials():
    uav = aircraft(z=15000)
    enemies = [entity(900, 1, 0, 10000), entity(901, 1, 1, 14000)]
    control = UAVControl()
    for step in range(40):
        uav.raw['agentPerception'] = [enemies[step % 2].uid]
        action = control.choose(state([uav], enemies, step=step), uav, None)
        if step >= 12:
            assert action.command != 'NormalAttacking'


def test_observed_approach_can_reopen_an_exhausted_fire_trial():
    uav = aircraft(z=15000)
    enemy = entity(900, 1, 0, 10000)
    uav.raw['agentPerception'] = [enemy.uid]
    control = UAVControl()
    for step in range(15):
        control.choose(state([uav], [enemy], step=step), uav, None)
    uav.position[0] = 7000
    assert control.choose(state([uav], [enemy], step=15), uav, None).command == 'NormalAttacking'


def test_multiple_aircraft_do_not_trial_the_same_ground_cluster_together():
    uavs = [aircraft(i + 1, z=15000) for i in range(3)]
    enemy = entity(900, 1, 0, 10000)
    control = UAVControl()
    for step in range(20):
        for uav in uavs:
            uav.raw['agentPerception'] = [enemy.uid]
        actions = [control.choose(state(uavs, [enemy], step=step), uav, None) for uav in uavs]
        assert sum(action.command == 'NormalAttacking' for action in actions) <= 1


def test_surviving_failed_strike_does_not_loop_again_after_its_cooldown():
    uav = aircraft()
    enemy = entity(900, 1, 0, 1000)
    uav.raw['agentPerception'] = [enemy.uid]
    control = UAVControl()
    actions = [control.choose(state([uav], [enemy], step=step), uav, None) for step in range(50)]
    assert sum(action.command == 'SelfDestruct' for action in actions) == control.STRIKE_STEPS
    enemy.hp -= 10
    assert control.choose(state([uav], [enemy], step=50), uav, None).command == 'SelfDestruct'


def test_dead_attacker_is_not_proof_of_a_hit_or_permission_for_a_second_dive():
    uavs = [aircraft(1), aircraft(2)]
    enemy = entity(900, 1, 0, 1000)
    for uav in uavs:
        uav.raw['agentPerception'] = [enemy.uid]
    control = UAVControl()
    for step in range(3):
        for uav in uavs:
            control.choose(state(uavs, [enemy], step=step), uav, None)
    assert 1 in control._missions
    uavs[0].alive = False
    for step in range(3, 40):
        action = control.choose(state(uavs, [enemy], step=step), uavs[1], None)
        assert action.command != 'SelfDestruct'


def recon_case():
    objectives = map_objectives()
    hq = objectives[0]
    soldiers = [entity(100 + i, 0, i, hq.position[0] + 15000, hq.position[1]) for i in range(6)]
    uavs = [aircraft(i + 1, x=hq.position[0] + 1000, y=hq.position[1]) for i in range(3)]
    route = SimpleNamespace(adaptive=True, _garrison={}, recon_objective=lambda current, agent: objectives[2])
    return objectives, soldiers, uavs, route, UAVControl()


def test_observer_keeps_side_objective_separate_and_is_replaced_after_loss():
    objectives, soldiers, uavs, route, control = recon_case()
    enemy = entity(900, 1, 0, *objectives[0].position[:2])
    units = [*soldiers, *uavs]
    for step in range(3):
        for uav in uavs:
            uav.raw['agentPerception'] = [enemy.uid]
        actions = [control.choose(state(units, [enemy], objectives, step), uav, objectives[0], route) for uav in uavs]
        assert actions[0].command not in {'NormalAttacking', 'SelfDestruct'}
        assert control._objectives[uavs[0].uid] == objectives[2].uid
        assert control._objectives[uavs[2].uid] == objectives[0].uid
    first_station = control._stations[uavs[0].uid]
    control.choose(state(units, [enemy], objectives, 3), uavs[2], objectives[0], route)
    assert control._stations[uavs[0].uid] == first_station
    uavs[0].alive = False
    # Aircraft 2 is already diving; aircraft 3 must preserve observation.
    action = control.choose(state(units, [enemy], objectives, 4), uavs[2], objectives[0], route)
    assert action.command not in {'NormalAttacking', 'SelfDestruct'}
    assert control._objectives[uavs[2].uid] == objectives[2].uid


def test_side_recon_does_not_leave_a_distant_observed_parent():
    objectives, soldiers, uavs, route, control = recon_case()
    parent = soldiers[0]
    parent.position[0] = 100000
    uavs[0].raw['indexInTeam'] = parent.uid
    current = state([*soldiers, *uavs], objectives=objectives)
    assert current.current_commander(uavs[0]) is parent
    control.choose(current, uavs[0], objectives[0], route)
    assert control._objectives[uavs[0].uid] == objectives[0].uid


def test_small_parent_motion_does_not_alternate_secondary_and_main_flight_legs():
    objectives, soldiers, uavs, route, control = recon_case()
    parent, uav, target = soldiers[0], uavs[0], objectives[2]
    uav.raw['indexInTeam'] = parent.uid
    for step, distance in enumerate((79000, 81000, 79500, 82000)):
        parent.position[:] = (target.position[0] + distance, target.position[1], 0)
        current = state([*soldiers, *uavs], objectives=objectives, step=step)
        control.choose(current, uav, objectives[0], route)
        assert control._objectives[uav.uid] == target.uid


def test_new_episode_discards_old_failed_fire_and_strike_evidence():
    uav = aircraft()
    enemy = entity(900, 1, 0, 1000)
    uav.raw['agentPerception'] = [enemy.uid]
    control = UAVControl()
    for step in range(20):
        control.choose(state([uav], [enemy], step=step), uav, None)
    for step in range(3):
        action = control.choose(state([uav], [enemy], step=step, episode=2), uav, None)
    assert action.command == 'SelfDestruct'
