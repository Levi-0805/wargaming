import json
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.TargetFeatures import (
    VERSION, STATE_DIM, TARGET_DIM, state_features, target_features)
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.LearnedTargeting import TargetPolicy, HybridIQLAgent
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RLAgentAlgorithm import AlgorithmConfig
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.VehicleFireControl import VehicleFireControl
from test_fslutk_rule import entity, state, map_objectives, algorithm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from train_target_iql import episode_transitions
from train_target_iql import aligned_actions
from train_target_iql import FEATURE_SOURCE


def model_file(tmp_path, **updates):
    w = np.zeros((1, STATE_DIM+TARGET_DIM)); w[0, STATE_DIM+4] = 8
    data = dict(algorithm='target_iql_v1', feature_version=VERSION, state_dim=STATE_DIM, target_dim=TARGET_DIM,
                features_sha256=hashlib.sha256(FEATURE_SOURCE.read_bytes()).hexdigest(),
                actor=[dict(weight=w.tolist(), bias=[0]), dict(weight=[[1]], bias=[0]), dict(weight=[[1]], bias=[0])],
                type_support=np.full((4, 4), 100).tolist())
    data.update(updates)
    p = tmp_path / 'model.json'; p.write_text(json.dumps(data), encoding='utf-8')
    return p


def case():
    u = entity(1, 0, 0, 0)
    low = entity(20, 1, 0, 2000, hp=20)
    high = entity(21, 1, 1, 3000, hp=100)
    u.raw.update(agentPerception=[20, 21], interaction=['100'])
    return u, low, high, state([u], [low, high], map_objectives())


def test_features_do_not_include_uids_terminal_score_or_unobserved_global_state():
    u, low, high, s = case()
    before = state_features(s, u), target_features(s, u, high)
    s.raw['Score'] = {'cheat': 6000}
    s.raw['hiddenEnemies'] = [{'hp': 1, 'position': [1, 2, 3]}]
    u.uid, low.uid, high.uid = 501, 520, 521
    u.raw['agentPerception'] = [520, 521]
    np.testing.assert_array_equal(before[0], state_features(s, u))
    np.testing.assert_array_equal(before[1], target_features(s, u, high))


@pytest.mark.parametrize('shadow', [False, True])
def test_model_ranks_only_current_candidates_and_shadow_preserves_default(tmp_path, shadow):
    u, low, high, s = case()
    policy = TargetPolicy(model_file(tmp_path), shadow=shadow)
    assert policy.choose(s, u, [low, high], low) is (low if shadow else high)
    assert policy.applied_changes == int(not shadow)
    u.raw['agentPerception'] = [low.uid]
    assert policy.choose(s, u, [low, high], low) is low
    assert policy.choose(s, u, [low], low) is low


def test_unrepresented_type_pair_falls_back(tmp_path):
    u, low, high, s = case()
    policy = TargetPolicy(model_file(tmp_path, type_support=np.zeros((4, 4), int).tolist()), shadow=False)
    assert policy.choose(s, u, [low, high], low) is low


@pytest.mark.parametrize('change', [{'feature_version': 999}, {'features_sha256': 'wrong'}, {'algorithm': 'unknown'},
                                  {'actor': []}, {'type_support': [[-1]]}])
def test_invalid_model_refuses_loading(tmp_path, change):
    with pytest.raises(ValueError):
        TargetPolicy(model_file(tmp_path, **change))


def test_vehicle_learned_rank_never_skips_aim_or_current_range_filter(tmp_path):
    u, low, high, s = case()
    u.entity_type = 'BP_MNWS_Vehicle_6x6UGV_C'
    control = VehicleFireControl()
    policy = TargetPolicy(model_file(tmp_path), shadow=False)
    control.target_selector = policy.choose
    assert control.choose(s, u).command == 'Guard'
    assert control.choose(state([u], [low, high], step=1), u).target_uid == high.uid
    high.position[0] = 20000
    assert control.choose(state([u], [low, high], step=2), u).command == 'Guard'
    assert control.choose(state([u], [low, high], step=3), u).target_uid == low.uid
    # A callback cannot inject a target that failed eligibility.
    control.target_selector = lambda *_: high
    assert control.choose(state([u], [low, high], step=4), u).target_uid == low.uid


def test_hybrid_shadow_keeps_rule_commands_and_records_predictions(tmp_path):
    u, low, high, s = case()
    base = algorithm([u]); cfg = AlgorithmConfig(mode='hybrid_iql', iql_model=str(model_file(tmp_path)),
                                               enable_parent_assignment=False, iql_shadow=True)
    learned = HybridIQLAgent(base.context, cfg)
    for step in range(3):
        s = state([u], [low, high], map_objectives(), step=step)
        assert learned.act(s) == base.act(s)
    assert learned.training_metrics()['iql_online_training'] == 0
    assert learned.policy_metadata()['mode'] == 'hybrid_iql'
    assert len(learned.policy_metadata()['model_sha256']) == 64
    assert learned.policy_metadata()['shadow'] is True


def test_terminal_score_appears_once_and_elapsed_steps_control_bootstrap():
    rows = []
    for step in (3, 7, 10):
        rows.append(dict(step=step, s=np.zeros(STATE_DIM), a=np.zeros((16, TARGET_DIM)),
                         mask=np.ones(16, bool), label=0, kind=0, target_kind=0))
    samples = list(episode_transitions({1: rows}, 12, 3000, .5))
    assert [r['reward'] for r in samples] == [0, 0, .125]
    assert [r['discount'] for r in samples] == [.0625, .125, 0]
    assert [r['terminal'] for r in samples] == [False, False, True]
    assert samples[1]['next_s'] is rows[2]['s']


def test_iql_shadow_requires_real_boolean():
    with pytest.raises(ValueError):
        AlgorithmConfig(iql_shadow='false')


@pytest.mark.parametrize('bad', [dict(DataCmd='reset'), dict(TimeStep=2),
                              dict(StringActions=['ActionSet2::NormalAttacking;20;1']*2),
                              dict(StringActions=['invalid']), dict(StringActions=[123])])
def test_stale_or_duplicate_recorded_actions_are_rejected(bad):
    packet = dict(DataCmd='step', TimeStep=3, StringActions=['ActionSet2::NormalAttacking;20;1'])
    assert aligned_actions(packet, 3)
    packet.update(bad)
    assert not aligned_actions(packet, 3)


def test_offline_training_uses_real_td_loss_and_never_updates_on_holdout_rewards(tmp_path):
    import hashlib
    from train_target_iql import train
    rng = np.random.default_rng(7)
    size = 280
    arrays = dict(s=rng.normal(size=(size, STATE_DIM)).astype('float32'),
                  a=rng.normal(size=(size, 16, TARGET_DIM)).astype('float32'),
                  mask=np.ones((size, 16), bool), label=np.zeros(size, np.int64),
                  next_s=np.zeros((size, STATE_DIM), 'float32'), reward=np.full(size, .5, 'float32'),
                  discount=np.zeros(size, 'float32'), kind=np.zeros(size, np.int64),
                  target_kind=np.zeros(size, np.int64),
                  split=np.asarray([0]*256+[1]*12+[2]*12, np.int8))
    def write_data():
        np.savez_compressed(tmp_path / 'decisions.npz', **arrays)
        (tmp_path / 'manifest.json').write_text(json.dumps(dict(
            feature_version=VERSION, features_sha256=hashlib.sha256(FEATURE_SOURCE.read_bytes()).hexdigest(),
            dataset_sha256=hashlib.sha256((tmp_path/'decisions.npz').read_bytes()).hexdigest())), encoding='utf-8')
    write_data()
    first = tmp_path / 'first.json'
    report = train(tmp_path, first, steps=2, seed=12)
    assert report['training_trace'][0]['q_loss'] > 0
    arrays['reward'][256:] = 99  # Neither validation nor test targets enter an update.
    write_data()
    second = tmp_path / 'second.json'
    train(tmp_path, second, steps=2, seed=12)
    assert json.loads(first.read_text())['actor'] == json.loads(second.read_text())['actor']
    TargetPolicy(first)  # Exported numeric model is accepted by live inference.


def test_training_refuses_old_feature_schema_before_reading_samples(tmp_path):
    from train_target_iql import train
    (tmp_path/'manifest.json').write_text(json.dumps(dict(feature_version=999)), encoding='utf-8')
    with pytest.raises(ValueError, match='feature schema'):
        train(tmp_path, tmp_path/'never-created.json', steps=1)
    assert not (tmp_path/'never-created.json').exists()
