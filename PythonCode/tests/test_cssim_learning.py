from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import cssim_annealing as anneal
import cssim_learning as learning
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.DefenseDoctrine import DefenseDoctrine, validate


def test_calendar_cooling_and_reheat_do_not_use_rolling_72_hours():
    anchor='2026-10-31'
    times=[datetime(2026,10,31,0,tzinfo=anneal.CHINA),datetime(2026,11,1,0,tzinfo=anneal.CHINA),
           datetime(2026,11,2,23,59,tzinfo=anneal.CHINA),datetime(2026,11,3,0,tzinfo=anneal.CHINA)]
    clocks=[anneal.temperature(anchor,t) for t in times]
    assert [c['temperature'] for c in clocks]==[800,300,100,800]
    assert [c['window'] for c in clocks]==[0,0,0,1]


def test_tail_energy_prevents_two_high_games_masking_catastrophic_round():
    stable=anneal.metrics([5100]*10)
    unstable=anneal.metrics([6000]*8+[870,2190])
    assert unstable['mean']>stable['mean']
    assert unstable['energy']>stable['energy']
    incumbent=dict(metrics=stable,checkpoint={})
    observed=dict(metrics=unstable,checkpoint={})
    result=anneal.assess({'current':incumbent,'champion':incumbent},observed,
                         round_number=1,anchor='2026-10-31')
    assert not result['last_acceptance']['accepted']
    assert result['champion']==incumbent
    assert not unstable['stability_target_met_in_sample']


def test_proposals_are_small_reproducible_and_preserve_unknown_fallback():
    profiles=DefenseDoctrine().config
    now=datetime(2026,10,31,tzinfo=timezone.utc)
    for number in range(1,41):
        proposed, change=anneal.propose(profiles,round_number=number,anchor='2026-10-31',now=now)
        validate(proposed)
        assert proposed['profiles']['unknown']==profiles['profiles']['unknown']
        differences=[(k,f) for k in profiles['profiles'] for f in profiles['profiles'][k]
                     if proposed['profiles'][k][f]!=profiles['profiles'][k][f]]
        assert differences==[(change['kind'],change['field'])]
        assert anneal.propose(profiles,round_number=number,anchor='2026-10-31',now=now)[0]==proposed


def test_unobserved_tactical_branch_cannot_be_accepted_as_a_score_gain():
    incumbent=dict(metrics=anneal.metrics([4000]*10),checkpoint={})
    observed=dict(metrics=anneal.metrics([6000]*10),checkpoint={},tested_kind='hq_b',
                  defense_groups={'hq_only':dict(count=10)})
    result=anneal.assess({'current':incumbent,'champion':incumbent},observed,
                        round_number=2,anchor='2026-10-31')
    assert result['current']==incumbent and result['champion']==incumbent
    assert result['last_acceptance']['reason']=='insufficient_observed_defence_exposure'


def test_four_round_reports_use_terminal_totals_and_keep_source_versions(tmp_path):
    def summary(number):
        return dict(round_number=number,source_commit=str(number)*40,model_sha256='a'*64,
            profile_sha256='b'*64,games=[dict(red=dict(total=5000+number)) for _ in range(10)],
            metrics=anneal.metrics([5000+number]*10),actual_exit_codes=[1,0],
            problems=['test problem'],plan=['test proposal'],defense_groups={})
    state=dict(reports_dir=str(tmp_path),rounds=[summary(i) for i in (1,2,3)])
    learning.write_reports(state,summary(4))
    report=(tmp_path/'四轮大报-0004.txt').read_text(encoding='utf-8-sig')
    assert '40局' in report and '5002.5' in report
    assert all(str(i)*40 in report for i in (1,2,3,4))
    assert (tmp_path/'第0004轮.txt').exists()
    assert not list(tmp_path.glob('*.json'))


def test_release_cannot_start_without_review_or_with_stop(tmp_path,monkeypatch):
    monkeypatch.setattr(learning,'STOP',tmp_path/'STOP')
    with pytest.raises(ValueError,match='reviewed'):
        learning.release(dict(phase='collecting'), 'x'*40)
    learning.STOP.touch()
    with pytest.raises(ValueError,match='STOP'):
        learning.release(dict(phase='awaiting_review',next_expected={}), 'x'*40)


def test_stop_also_protects_the_window_before_collector_directory_exists(tmp_path,monkeypatch):
    monkeypatch.setattr(learning,'ROOT',tmp_path)
    monkeypatch.setattr(learning,'STOP',tmp_path/'logs/learning/STOP')
    learning.request_stop(dict(active_campaign='new-round'))
    assert learning.STOP.exists()
    assert (tmp_path/'logs/campaigns/new-round/STOP').exists()


def test_failed_integration_does_not_modify_published_artifacts(tmp_path,monkeypatch):
    model,profile=tmp_path/'model.json',tmp_path/'profile.json'
    model.write_bytes(b'original model');profile.write_bytes(b'original profile')
    candidate=tmp_path/'candidate.json';candidate.write_bytes(b'trial')
    review=tmp_path/'review.json'
    review.write_text(json.dumps(dict(candidate_model=str(candidate),candidate_model_sha256=learning.batches.sha(candidate))))
    monkeypatch.setattr(learning,'MODEL',model);monkeypatch.setattr(learning,'PROFILE',profile)
    monkeypatch.setattr(learning,'STOP',tmp_path/'STOP')
    monkeypatch.setattr(learning,'checked_git',lambda *a:'')
    def fail(*a): raise RuntimeError('integration failed')
    monkeypatch.setattr(learning,'child',fail)
    state=dict(phase='awaiting_review',round_number=1,search=dict(current=dict(checkpoint=dict(model=str(model),profiles=str(profile)))),
        pending=dict(proposed_change={},candidate_review=str(review),games=[dict(red=dict(total=1),protocol='p',episode=1)]))
    with pytest.raises(RuntimeError,match='integration failed'): learning.prepare_next(state,use_candidate=True)
    assert model.read_bytes()==b'original model' and profile.read_bytes()==b'original profile'


def test_score_inbox_exists_before_any_candidate_is_ready(tmp_path,monkeypatch):
    monkeypatch.setattr(learning,'STATE',tmp_path/'state.json')
    games=[dict(red=dict(kill_score=1200,capture_score=1600,total=2800)) for _ in range(10)]
    summary=dict(campaign_id='cohort',source_commit='a'*40,games=games,metrics=dict(count=10))
    signal=learning.announce_scores(dict(round_number=1),summary,'collection.json')
    assert not signal['delivered_to_chat'] and signal['candidate_status']=='not_yet_trained'
    assert learning.read(tmp_path/'report-ready.json')==signal
    assert learning.read(signal['summary_path'])==summary
    summary['games']=games[:9]
    with pytest.raises(ValueError,match='ten'):learning.announce_scores(dict(round_number=1),summary,'x')
    summary['games']=games;games[0]['red']['total']=9999
    with pytest.raises(ValueError,match='destruction'):learning.announce_scores(dict(round_number=1),summary,'x')


def rollback_fixture(tmp_path,monkeypatch):
    model=tmp_path/'live-model.json';model.write_bytes(b'unchanged model')
    checkpoint=tmp_path/'saved-model.json';checkpoint.write_bytes(model.read_bytes())
    profile=tmp_path/'live-profile.json';profile.write_text('{}')
    monkeypatch.setattr(learning,'MODEL',model);monkeypatch.setattr(learning,'PROFILE',profile)
    monkeypatch.setattr(learning,'ROOT',tmp_path);monkeypatch.setattr(learning,'STATE',tmp_path/'state.json')
    monkeypatch.setattr(learning,'STOP',tmp_path/'STOP');monkeypatch.setattr(learning,'checked_git',lambda *a:'')
    state=dict(phase='awaiting_review',round_number=1,active_campaign='cohort',
        search=dict(current=dict(checkpoint=dict(model=str(checkpoint),profiles=str(profile)))),pending={})
    restored=tmp_path/'restored.json'
    restored.write_text(json.dumps(DefenseDoctrine().config))
    return state,restored,model,profile


def test_valid_profile_rollback_is_separate_from_weight_trial(tmp_path,monkeypatch):
    state,restored,model,profile=rollback_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='separate'):
        learning.prepare_next(state,use_candidate=True,restore_profiles=restored)
    review=learning.prepare_next(state,restore_profiles=restored)
    assert review['chosen']=='restore_historical_parameters'
    assert review['proposal']['scope']=='all_defense_profiles'
    assert model.read_bytes()==b'unchanged model' and profile.read_bytes()==restored.read_bytes()
    assert state['next_expected']['profile_sha256']==learning.batches.sha(restored)


def test_invalid_restore_does_not_change_either_live_artifact(tmp_path,monkeypatch):
    state,restored,model,profile=rollback_fixture(tmp_path,monkeypatch)
    restored.write_text('{"schema":99}')
    with pytest.raises(ValueError):learning.prepare_next(state,restore_profiles=restored)
    assert model.read_bytes()==b'unchanged model' and profile.read_text()=='{}'


def test_observer_stop_and_existing_identity_prevent_duplicate_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(learning,'ROOT',tmp_path);monkeypatch.setattr(learning,'STOP',tmp_path/'STOP')
    learning.STOP.touch()
    assert learning.ensure_observer(dict(rounds=[])) is None
    learning.STOP.unlink()
    identity=dict(pid=123,created_ticks=456,exe_path='pythonw.exe')
    learning.batches.write_json(tmp_path/'logs/adaptive-observer/launch.json',dict(identity=identity))
    monkeypatch.setattr(learning,'live',lambda value:value==identity)
    assert learning.ensure_observer(dict(rounds=[]))==identity
    assert learning.ensure_observer(dict(rounds=[{}]*10)) is None
