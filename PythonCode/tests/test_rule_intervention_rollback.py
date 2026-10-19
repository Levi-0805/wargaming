"""Intervention removal, not a synthetic claim about match score or survival."""
import pytest

from cssim.algorithms import AlgorithmContext
from cssim.protocol import ActionSetFactory
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RLAgentAlgorithm import (
    AlgorithmConfig, ReinforceAgentAlgorithm as EntryPoint,
)
from test_fslutk_rule import algorithm, entity, state, map_objectives


SWITCHES = ('enable_preservation_pause', 'enable_shared_contact_maneuver',
            'enable_forced_entry_under_contact', 'enable_continuous_column',
            'enable_entry_waves')


@pytest.mark.parametrize('enabled', [None, *SWITCHES])
def test_platform_entry_defaults_off_and_allows_one_intervention_at_a_time(enabled):
    config = {} if enabled is None else {enabled: True}
    rule = EntryPoint(AlgorithmContext(0, (), (), ActionSetFactory.standard(), 'cpu', config))
    for name in SWITCHES:
        assert getattr(rule, name) is (name == enabled)


@pytest.mark.parametrize('name', SWITCHES)
def test_string_false_cannot_accidentally_enable_a_retired_rule(name):
    with pytest.raises(ValueError, match='JSON boolean'):
        AlgorithmConfig(**{name: 'false'})


@pytest.mark.parametrize('kind', ['BP_BaseSoldier_C', 'BP_RoboDog_C',
                                  'BP_MNWS_Vehicle_Armored_C', 'BP_MNWS_Vehicle_6x6UGV_C'])
def test_shared_reports_do_not_replace_ground_navigation_with_probe_or_orbit(kind):
    unit = entity(1, 0, 0, -29782, 25170, kind=kind)
    unit.raw.update(agentPerception=[], interaction=['250'])
    obj = map_objectives()
    reported, silent = algorithm([unit]), algorithm([unit])
    changes = []
    for step in range(40):
        # Changing reports behind the facade must not keep choosing new tiny
        # lateral destinations; ordinary route and obstruction recovery remain.
        enemy = entity(20 + step % 2, 1, 0, -39000 + step % 3 * 100, 33000)
        current = state([unit], [enemy], obj, step)
        actual = reported.act(current)[0]
        expected = silent.act(state([unit], objectives=obj, step=step))[0]
        assert actual == expected
        assert actual.command != 'NormalAttacking'
        changes.append(actual)
    expected_command = 'Guard' if kind in {'BP_MNWS_Vehicle_Armored_C', 'BP_MNWS_Vehicle_6x6UGV_C'} else 'Moving'
    assert any(action.command == expected_command and action.points is not None
               and action.points != tuple(unit.position) for action in changes)


@pytest.mark.parametrize('kind', ['BP_BaseSoldier_C', 'BP_RoboDog_C',
                                  'BP_MNWS_Vehicle_Armored_C', 'BP_MNWS_Vehicle_6x6UGV_C'])
def test_active_entrance_route_yields_to_local_fire_and_resumes_its_cursor(kind):
    unit = entity(1, 0, 0, -35000, 28500, kind=kind)
    unit.position[2] = 6425
    unit.raw.update(agentPerception=[], interaction=['250'])
    objects = map_objectives()
    objects[0].raw['blueteamNum'] = 5
    rule = algorithm([unit])
    for step in range(25):
        rule.act(state([unit], objectives=objects, step=step))
    cursor = rule._hq_entry._routes[unit.uid]
    enemy = entity(20, 1, 0, -36000, 29000)
    unit.raw['agentPerception'] = [enemy.uid]
    for step in range(25, 40):
        enemy.hp = 100 - (step - 25)
        unit.raw['interaction'] = [str(250 - (step - 25))]
        action = rule.act(state([unit], [enemy], objects, step))[0]
        if kind in rule.VEHICLES and step == 25:
            assert action.command == 'Guard'  # Aim before the first explicit shot.
        else:
            assert action.command == 'NormalAttacking' and action.target_uid == enemy.uid
        assert rule._hq_entry._routes[unit.uid] == cursor
    unit.raw['agentPerception'] = []
    actions = [rule.act(state([unit], objectives=objects, step=step))[0] for step in range(40, 43)]
    # Wheeled cars may spend two frames aligning their heading first.
    assert actions[-1].command == ('Guard' if kind in rule.VEHICLES else 'Moving')
    assert actions[-1].points is not None
    assert rule._hq_entry._routes[unit.uid] == cursor


def test_casualty_threshold_no_longer_orders_exposed_survivors_to_wait():
    objects = map_objectives()
    obj = objects[0]
    obj.raw['blueteamNum'] = 16
    units = [entity(1+i, 0, i, obj.position[0] - 27000, obj.position[1]) for i in range(12)]
    for unit in units:
        unit.raw.update(agentPerception=[], interaction=['250'])
    default = algorithm(units)
    legacy = algorithm(units, enable_preservation_pause=True)
    for rule in (default, legacy):
        rule.act(state(units, objectives=objects))
    for unit in units:
        unit.position[0] += 8000
    for unit in units[4:]:
        unit.alive = False
    current = state(units, objectives=objects, step=1)
    old = legacy.act(current)
    new = default.act(current)
    assert all(a.command == 'Guard' and a.points == tuple(u.position) for u, a in zip(units[:4], old[:4]))
    assert all(a.command == 'Moving' for a in new[:4])
    assert all(a.command == 'Idle' for a in new[4:])
