import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from cssim.algorithms import AlgorithmContext
from cssim.protocol import ActionSetFactory
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RLAgentAlgorithm import AlgorithmConfig, ReinforceAgentAlgorithm
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.Champion6cb5a79.RLAgentAlgorithm import ReinforceAgentAlgorithm as Original
from test_fslutk_rule import entity, map_objectives, state

def context():
    unit = entity(1, 0, 0, 60000, 12000)
    return AlgorithmContext(0, (unit,), (), ActionSetFactory.standard(), 'cpu',
                            dict(mode='hybrid_iql', policy_baseline='6cb5a79', iql_shadow=False))

def test_historical_hybrid_resolves_model_from_original_team_root():
    policy = ReinforceAgentAlgorithm(context())
    meta = policy.policy_metadata()
    assert meta['mode'] == 'hybrid_iql' and meta['shadow'] is False
    assert meta['model_sha256'] == '3e36fb18875258f58065cfa748b57c8947675cded629654ce3b79696a8863f4f'
    assert meta['policy_baseline'] == '6cb5a79'
    assert meta['policy_source_commit'] == '6cb5a79ced2cf5b2e9d282a8349ff3a5416e6694'
    assert meta['defense_profiles_applied'] is False

def test_bridge_preserves_original_actions_across_frames():
    ctx = context()
    bridge = ReinforceAgentAlgorithm(ctx)
    original_options = dict(ctx.config)
    original_options.pop('policy_baseline')
    team_root = Path(__file__).resolve().parents[1] / 'TeamAlg/FsLutk7wD2RkNgbs'
    original_options['iql_model'] = str(team_root / 'Models/target_iql_v1.json')
    original = Original(replace(ctx, config=original_options))
    for step in range(8):
        current = state(ctx.agents, objectives=map_objectives(), step=step)
        assert [a.to_ue_string(u.uid, ctx.agents) for u,a in zip(ctx.agents,bridge.act(current))] == [a.to_ue_string(u.uid, ctx.agents) for u,a in zip(ctx.agents,original.act(current))]
        assert bridge.take_selected_special_commands() == original.take_selected_special_commands()

def test_unknown_baseline_is_rejected():
    with pytest.raises(ValueError, match='baseline'):
        AlgorithmConfig(policy_baseline='unreviewed')

def test_snapshot_tampering_is_rejected_before_instantiation(monkeypatch):
    from TeamAlg.FsLutk7wD2RkNgbs.Algorithm import ChampionPolicyBridge as bridge
    class Wrong:
        def hexdigest(self): return '0' * 64
    monkeypatch.setattr(bridge.hashlib, 'sha256', lambda value: Wrong())
    with pytest.raises(ValueError, match='snapshot changed'):
        ReinforceAgentAlgorithm(context())
