"""HQ entry repair must restore movement without relaxing capture geometry."""
from dataclasses import replace
import math

import pytest

from cssim.algorithms import AlgorithmContext
from cssim.environment.models import KeyObject
from cssim.protocol import ActionSetFactory
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.ChampionGuardHold import HQGuardEntryRoute, TiedHQGuardRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.Champion6cb5a79.MotionRecovery import MotionRecovery
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.Champion6cb5a79.RuleAgentAlgorithm import ReinforceAgentAlgorithm as HistoricalRule
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RLAgentAlgorithm import ReinforceAgentAlgorithm
from test_champion_baseline import context
from test_fslutk_rule import entity, state


def objective(*, uid=32769, yaw=0, scale=(7700, 6500)):
    return KeyObject.from_raw({
        'uId': uid,
        'location': {'x': -41810.42, 'y': 32275.18, 'z': 6326.022},
        'rotation': {'yaw': yaw},
        'scale': {'x': scale[0], 'y': scale[1], 'z': 2500},
        'redTeamNum': 4, 'blueteamNum': 0, 'percent': 1,
        'rSVD1': 'HQ',
    })


def world(obj, x, y, z=6429):
    yaw = math.radians(obj.raw['rotation']['yaw'])
    return (obj.position[0] + x * math.cos(yaw) - y * math.sin(yaw),
            obj.position[1] + x * math.sin(yaw) + y * math.cos(yaw), z)


def setup_guard(*, yaw=0, scale=(7700, 6500), local=None):
    obj = objective(yaw=yaw, scale=scale)
    route = HQGuardEntryRoute(adaptive=True)
    route.hq_uid = obj.uid
    extent = route._footprint(obj)
    position = world(obj, *(local or (0, -extent[1] - 2.18)))
    guard = entity(1431, 0, 0, position[0], position[1])
    guard.position[:] = position
    guard.raw['rSVD1'] = 'guard;1'
    route._garrison[guard.uid] = obj.uid
    old = route.clamp_guard_point(obj, guard.position)
    route._guard_slots[guard.uid, obj.uid] = old
    return route, obj, guard, old


@pytest.mark.parametrize('yaw', [0, 37, -90])
def test_outside_near_boundary_guard_receives_stable_interior_goal(yaw):
    route, obj, guard, old = setup_guard(yaw=yaw)
    assert not route.inside_garrison(obj, guard.position)
    assert route.distance(guard.position, old) < 3
    goal = route.guard_slot(guard, obj)
    x, y, _ = route._local_point(obj, goal)
    assert x == pytest.approx(0, abs=.01)
    assert y == pytest.approx(-5300, abs=.01)
    assert goal[2] == old[2]
    assert route.distance(guard.position, goal) > 800
    assert route.inside_garrison(obj, goal)
    assert not route.inside_garrison(obj, guard.position)
    # A stationary replay cannot manufacture a new target each frame.
    for _ in range(12):
        assert route.guard_slot(guard, obj) == goal
    guard.position[:] = goal
    assert route.guard_slot(guard, obj) == goal


def test_corner_clamps_each_local_axis_without_changing_elevation():
    route, obj, guard, old = setup_guard(yaw=24, local=(7302, -6102))
    goal = route.guard_slot(guard, obj)
    x, y, _ = route._local_point(obj, goal)
    assert (x, y) == pytest.approx((6500, -5300), abs=.01)
    assert goal[2] == old[2]
    assert route.inside_garrison(obj, goal)


@pytest.mark.parametrize('case', ['inside', 'other_objective', 'unassigned', 'far_slot',
                                 'dead', 'zero_hp', 'disconnected', 'normal_interior_slot'])
def test_unaffected_guards_preserve_the_historical_slot(case):
    route, obj, guard, old = setup_guard()
    if case == 'inside':
        guard.position[:] = world(obj, 0, -6000)
    elif case == 'other_objective':
        route.hq_uid = obj.uid + 1
    elif case == 'unassigned':
        route._garrison.clear()
    elif case == 'far_slot':
        guard.position[:] = world(obj, 0, -6600)
    elif case == 'dead':
        guard.alive = False
    elif case == 'zero_hp':
        guard.hp = 0
    elif case == 'disconnected':
        guard.raw['rSVD1'] = 'guard;0'
    elif case == 'normal_interior_slot':
        old = world(obj, 0, -5000)
        route._guard_slots[guard.uid, obj.uid] = old
    assert route.guard_slot(guard, obj) == old
    assert route._guard_slots[guard.uid, obj.uid] == old


def test_unknown_communication_does_not_exclude_an_existing_guard():
    route, obj, guard, old = setup_guard()
    guard.raw.pop('rSVD1')
    assert guard.communication_ok() is None
    assert route.guard_slot(guard, obj) != old


def test_missing_footprint_preserves_the_original_fallback():
    route, obj, guard, _ = setup_guard()
    obj.raw.pop('scale')
    original = TiedHQGuardRoute(adaptive=True)
    original.hq_uid = obj.uid
    original._garrison[guard.uid] = obj.uid
    assert route._footprint(obj) is None
    assert route.guard_slot(guard, obj) == original.guard_slot(guard, obj)


def test_small_capture_box_keeps_a_positive_interior_region():
    route, obj, guard, _ = setup_guard(scale=(500, 600), local=(102, -202))
    goal = route.guard_slot(guard, obj)
    x, y, _ = route._local_point(obj, goal)
    assert 0 < x < 100
    assert -200 < y < 0
    assert route.inside_garrison(obj, goal)
    assert goal[2] == guard.position[2]


def test_real_boundary_dead_zone_now_reaches_movement_recovery():
    route, obj, guard, old = setup_guard()
    assert guard.position[1] == pytest.approx(26173, abs=.01)
    assert old[1] == pytest.approx(26175.18, abs=.01)
    ctx = AlgorithmContext(0, (guard,), (), ActionSetFactory.standard(), 'cpu', {})
    policy = HistoricalRule(ctx, enable_parent_assignment=False)
    policy._route = route
    recovery = MotionRecovery()
    requested, recovered = [], []
    for step in range(recovery.STALL_STEPS + 1):
        current = state([guard], objectives=[obj], step=step)
        action = policy._defend(current, guard, obj, [], [])
        requested.append(action)
        recovered.append(recovery.apply(current, guard, action))
    assert all(a.command == 'Moving' and a.points[1] == pytest.approx(26975.18, abs=.01)
               for a in requested)
    assert all(a.points == requested[0].points for a in requested)
    assert recovered[0].points == requested[0].points
    assert recovered[-1].command == 'Moving' and recovered[-1].direction != 'N/A'
    # The original two-centimetre request never entered watchdog progress.
    from cssim.protocol import Action
    original_recovery = MotionRecovery()
    for step in range(original_recovery.STALL_STEPS + 1):
        result = original_recovery.apply(state([guard], objectives=[obj], step=step),
                                         guard, Action.move_at(old))
        assert result.points == old and result.direction == 'N/A'
    assert guard.uid not in original_recovery._progress


def test_factory_keeps_new_entry_experiment_separate_from_hold_baseline():
    import hashlib
    from pathlib import Path

    ctx = context()
    policy = ReinforceAgentAlgorithm(replace(ctx, config={**ctx.config,
        'policy_baseline': '6cb5a79_hq_guard_entry'}))
    assert isinstance(policy._route, HQGuardEntryRoute)
    metadata = policy.policy_metadata()
    assert metadata['policy_baseline'] == '6cb5a79_hq_guard_entry'
    assert metadata['policy_overlay'] == 'hq_tied_guard_retention_and_entry_margin'
    source = Path(__file__).resolve().parents[1] / 'TeamAlg/FsLutk7wD2RkNgbs/Algorithm/ChampionGuardHold.py'
    assert metadata['policy_overlay_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert metadata['policy_snapshot_sha256'] == '7d7e619118e5c7c22235deca80d49d0ab1d959d8172f5e8fd62f03f72ecdcac5'
    prior = ReinforceAgentAlgorithm(replace(ctx, config={**ctx.config,
        'policy_baseline': '6cb5a79_hq_hold'}))
    assert isinstance(prior._route, TiedHQGuardRoute)
    assert not isinstance(prior._route, HQGuardEntryRoute)
