from dataclasses import replace
import hashlib
from pathlib import Path
import pytest

from test_adaptive_assault import opening, advance
from test_fslutk_rule import guards_at, state
from test_champion_baseline import context
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.ChampionGuardHold import TiedHQGuardRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RLAgentAlgorithm import ReinforceAgentAlgorithm


def established():
    _, objects, _ = opening()
    units = guards_at(objects[0], count=8)
    route = TiedHQGuardRoute(adaptive=True)
    objects[0].raw.update(percent=1, redTeamNum=8, blueteamNum=0)
    advance(route, units, objects, range(4))
    assert route.stage == 1 and len(route._garrison) == 4
    objects[0].raw.update(percent=1, redTeamNum=4, blueteamNum=4)
    return units, objects, route


def test_existing_guards_hold_a_tied_hq_without_asserting_ownership():
    units, objects, route = established()
    before = dict(route._garrison)
    route._assign_garrison(state(units, objectives=objects, step=4))
    assert route._garrison == before
    assert not route.owned(state(units, objectives=objects), objects[0])
    assert route.stage == 1


@pytest.mark.parametrize('case', ['below_full', 'enemy_owner', 'no_previous', 'enemy_advantage',
                                  'dead', 'disconnected', 'far', 'remnant'])
def test_retention_does_not_override_loss_or_ineligible_guards(case):
    units, objects, route = established()
    uid = next(iter(route._garrison))
    if case == 'below_full': objects[0].raw['percent'] = .9
    if case == 'enemy_owner': objects[0] = replace(objects[0], team=1)
    if case == 'no_previous': route._garrison.clear()
    if case == 'enemy_advantage': objects[0].raw['blueteamNum'] = 5
    if case == 'dead': units = [replace(u, alive=False) if u.uid == uid else u for u in units]
    if case == 'disconnected':
        units = [replace(u, raw={**u.raw, 'rSVD1': 'guard;0'}) if u.uid == uid else u for u in units]
    if case == 'far':
        next(u for u in units if u.uid == uid).position[:] = (999999,999999,0)
    if case == 'remnant':
        # Explicit stop-loss must remain independent of ordinary guard retention.
        route._remnant.active = True
        route._remnant.target_uid = objects[1].uid
    current = state(units, objectives=objects, step=4)
    if case == 'disconnected':
        assert next(u for u in units if u.uid == uid).communication_ok() is False
    route._assign_garrison(current)
    assert len(route._garrison) == (3 if case in {'dead','disconnected','far'} else 0)


def test_tie_does_not_create_a_garrison_or_unlock_initial_departure():
    _, objects, _ = opening()
    route = TiedHQGuardRoute(adaptive=True)
    units = guards_at(objects[0], count=8)
    objects[0].raw.update(percent=1,redTeamNum=8,blueteamNum=8)
    advance(route, units, objects, range(30))
    assert route.stage == 0 and not route._garrison


def test_factory_binds_overlay_separately_from_immutable_snapshot():
    ctx = context()
    policy = ReinforceAgentAlgorithm(replace(ctx, config={**ctx.config,'policy_baseline':'6cb5a79_hq_hold'}))
    assert isinstance(policy._route,TiedHQGuardRoute)
    meta = policy.policy_metadata()
    assert meta['policy_baseline'] == '6cb5a79_hq_hold'
    assert meta['policy_overlay'] == 'hq_tied_guard_retention'
    file = Path(__file__).resolve().parents[1]/'TeamAlg/FsLutk7wD2RkNgbs/Algorithm/ChampionGuardHold.py'
    assert meta['policy_overlay_sha256'] == hashlib.sha256(file.read_bytes()).hexdigest()
    assert meta['policy_snapshot_sha256'] == '7d7e619118e5c7c22235deca80d49d0ab1d959d8172f5e8fd62f03f72ecdcac5'
