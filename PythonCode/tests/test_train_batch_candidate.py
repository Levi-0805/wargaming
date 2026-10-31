import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import train_batch_candidate as candidate
from batch_training import sha, write_json
from parse_match_score import parse_output

COMMIT = 'a' * 40


def protocol(path, episodes, worth=100):
    rows = []
    for ep in range(episodes):
        rows.extend([dict(propertyArr=[1]), dict(dataArr=[1], dataGlobal=dict(timeCnt=500, rSVD1='10;1'))])
        for side in ('red', 'blue'):
            rows.append(dict(Score={'u': {'Name': 'u', 'SumWorth': worth}}, Reward=1600,
                             Events=[{'摧毁方': side, '被摧毁方': 'blue' if side == 'red' else 'red',
                                      '得分': ('红方' if side == 'red' else '蓝方') + str(worth)}]))
    path.write_text(''.join(json.dumps(dict(direction='ue_to_python', payload=row), ensure_ascii=False)+'\n'
                            for row in rows), encoding='utf-8')
    return path


def finished(path, source, *, commit=COMMIT):
    shared = dict(batch_id=path.stem, source_commit=commit, config_sha256='1' * 64)
    result = dict(**shared, finished_at='2026-10-31T00:00:00+00:00',
                  status=dict(state='complete', ready_to_finish=True, expected_episodes=10, complete_episodes=10,
                              done_episodes=list(range(1,11)), ue_game_config='10;1', source_commit=commit,
                              protocol=str(source)),
                  match=parse_output(source, commit),
                  runtime_receipt=dict(**shared, pid=42, exit_code=0, log_directory=str(source.parent)),
                  runtime_identity=dict(**shared, process_id=42))
    write_json(path, result)
    return path


@pytest.fixture
def sources(tmp_path):
    old = [protocol(tmp_path / f'history-{i}.jsonl', 1, worth=100+i) for i in range(2)]
    live = protocol(tmp_path / 'new-batch.jsonl', 10, worth=200)
    batch = finished(tmp_path / 'finished.json', live)
    return tmp_path, old, live, batch


def test_valid_batch_is_bound_to_commit_history_remains_unknown(sources):
    _, old, live, batch = sources
    selected = candidate.select_sources([batch], old)
    assert [r['path'] for r in selected['sources']] == [str(p) for p in old] + [str(live)]
    assert [r['source_commit'] for r in selected['sources']] == [None, None, COMMIT]
    assert selected['sources'][-1]['batch_report_sha256'][str(batch)] == sha(batch)
    assert len(selected['sources'][-1]['rounds']) == 10


def cumulative(path, chunks):
    games, attempts = [], []
    for index, source in enumerate(chunks):
        batch=f'attempt-{index}'
        parsed=parse_output(source,COMMIT)
        manifest=path.parent/f'{batch}.json'
        write_json(manifest,dict(batch_id=batch,source_commit=COMMIT,model_sha256='b'*64))
        attempts.append(dict(batch_id=batch,manifest_path=str(manifest),
            status=dict(protocol=str(source),done_episodes=[r['episode'] for r in parsed['rounds']]),
            runtime_receipt=dict(exit_code=1)))
        for row in parsed['rounds']:
            games.append(dict(ordinal=len(games)+1,episode=row['episode'],batch_id=batch,
                protocol=str(source),protocol_sha256=parsed['source_sha256'],source_commit=COMMIT,model_sha256='b'*64,
                **{side:{k:row['scores'][side][k] for k in ('kill_score','capture_score','total')} for side in ('red','blue')}))
    write_json(path,dict(classification='ten_cumulative_games_complete',completed_games=10,
        source_commit=COMMIT,model_sha256='b'*64,games=games,attempts=attempts))
    return path


def test_cumulative_7_plus_3_with_failed_process_exits_retains_verified_complete_games(tmp_path):
    chunks=[protocol(tmp_path/'seven.jsonl',7),protocol(tmp_path/'three.jsonl',3)]
    report=cumulative(tmp_path/'cumulative.json',chunks)
    records=candidate.completed_report_sources(report)
    assert [len(r['rounds']) for r in records]==[7,3]
    assert all(r['source_commit']==COMMIT for r in records)


def test_recorded_runtime_commit_is_retained_without_guessing_from_filename(tmp_path):
    source=protocol(tmp_path/'old.jsonl',1)
    write_json(tmp_path/'run.json',dict(source_commit=COMMIT))
    record=candidate.historical_source(source)
    assert record['source_commit']==COMMIT
    assert record['source_commit_evidence']=='runtime_run_json'
    assert record['runtime_identity_sha256']==sha(tmp_path/'run.json')


@pytest.mark.parametrize('corruption', ['duplicate','score','source','sha','done'])
def test_cumulative_inconsistent_evidence_cannot_enter_training(tmp_path,corruption):
    source=protocol(tmp_path/'ten.jsonl',10)
    path=cumulative(tmp_path/'cumulative.json',[source])
    report=candidate.read_json(path)
    if corruption=='duplicate': report['games'][1]=dict(report['games'][0])
    if corruption=='score': report['games'][0]['red']['total']+=1
    if corruption=='source': report['games'][0]['source_commit']='f'*40
    if corruption=='sha': report['games'][0]['protocol_sha256']='0'*64
    if corruption=='done': report['attempts'][0]['status']['done_episodes'].remove(1)
    write_json(path,report)
    with pytest.raises(ValueError): candidate.completed_report_sources(path)


def test_duplicate_source_is_not_an_independent_split_and_retains_binding(sources):
    _, old, live, batch = sources
    selected = candidate.select_sources([batch, batch], [*old, live, old[0]])
    assert len(selected['sources']) == 3
    assert selected['sources'][-1]['source_commit'] == COMMIT
    assert len(selected['rejected']) == 3
    with pytest.raises(ValueError, match='three distinct'):
        candidate.select_sources([batch], [live, old[0], old[0]])


@pytest.mark.parametrize('mutation', ['partial', 'crash', 'commit', 'pid', 'changed_protocol',
                                     'stop_reason', 'error', 'log_directory', 'missing_log_directory'])
def test_unfinished_changed_or_ambiguous_batches_cannot_enter_training(sources, mutation):
    _, old, live, batch = sources
    report = candidate.read_json(batch)
    if mutation == 'partial': report['status']['complete_episodes'] = 9
    elif mutation == 'crash': report['runtime_receipt']['exit_code'] = 1
    elif mutation == 'commit': report['match']['source_commit'] = 'b'*40
    elif mutation == 'pid': report['runtime_identity']['process_id'] = 77
    elif mutation == 'stop_reason': report['runtime_receipt']['stop_reason'] = 'timeout'
    elif mutation == 'error': report['runtime_receipt']['error'] = 'connection failed'
    elif mutation == 'log_directory': report['runtime_receipt']['log_directory'] = str(live.parent / 'another')
    elif mutation == 'missing_log_directory': report['runtime_receipt'].pop('log_directory')
    elif mutation == 'changed_protocol':
        with live.open('ab') as stream: stream.write(b'\n')
    write_json(batch, report)
    with pytest.raises(ValueError, match='verified finished'):
        candidate.select_sources([batch], old)


def test_invalid_batch_is_reported_but_does_not_remove_another_valid_batch(sources):
    root, old, _, batch = sources
    bad = root / 'partial.json'
    write_json(bad, {'status': {'state': 'partial'}})
    selected = candidate.select_sources([bad, batch], old)
    assert len(selected['sources']) == 3
    assert selected['rejected'][0]['path'] == str(bad)


def test_conflicting_commit_bindings_fail_closed(sources):
    root, old, live, batch = sources
    conflict = finished(root / 'conflict.json', live, commit='b'*40)
    with pytest.raises(ValueError, match='conflicting'):
        candidate.select_sources([batch, conflict], old)


def test_candidate_trains_separate_artifact_and_records_actual_report(sources, monkeypatch):
    root, old, live, batch = sources
    baseline = root / candidate.BASELINE
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b'baseline policy')
    monkeypatch.setattr(candidate.subprocess, 'check_output', lambda *a, **kw: COMMIT)
    calls = []

    def build(paths, output):
        calls.append(paths)
        output.mkdir()
        (output / 'decisions.npz').write_bytes(b'unchanged dataset')
        manifest = dict(sources=[dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size, source_commit=None) for p in paths],
                        dataset_sha256=sha(output/'decisions.npz'), split_sessions=dict(train=[0], validation=1, test=2))
        write_json(output/'manifest.json', manifest)
        return manifest

    def train(dataset, output, *, steps, seed):
        manifest = candidate.read_json(dataset/'manifest.json')
        assert manifest['sources'][2]['source_commit'] == COMMIT
        assert manifest['sources'][2]['source_commit_evidence'] == 'verified_finished_batch'
        assert (dataset/'decisions.npz').read_bytes() == b'unchanged dataset'
        output.write_bytes(b'candidate numeric policy')
        report = dict(model_sha256=sha(output), offline_metrics={'validation': {'nll': .5}})
        write_json(dataset/'training-report.json', report)
        return report

    monkeypatch.setattr(candidate, 'build_dataset', build)
    monkeypatch.setattr(candidate, 'train', train)
    review = candidate.train_candidate([batch], old, 'candidate-1', repo=root, steps=10)
    assert calls == [[*old, live]]
    assert baseline.read_bytes() == b'baseline policy'
    assert review['state'] == 'trained_unpublished' and review['promoted'] is False
    assert review['new_match_outcomes'] is False
    assert review['training_report_sha256'] == sha(review['training_report'])
    assert review['candidate_model_sha256'] == sha(review['candidate_model'])
    assert Path(review['review_path']).is_file()
    with pytest.raises(ValueError, match='already exists'):
        candidate.train_candidate([batch], old, 'candidate-1', repo=root)


def test_manifest_snapshot_mismatch_does_not_get_false_commit(sources):
    root, old, _, batch = sources
    selection = candidate.select_sources([batch], old)
    dataset = root / 'dataset'
    dataset.mkdir()
    write_json(dataset/'manifest.json', {'sources': [dict(path='wrong', sha256='f'*64, bytes=1)]})
    with pytest.raises(ValueError, match='snapshot'):
        candidate.bind_manifest(dataset, selection)


@pytest.mark.parametrize('identifier', ['../escape', '', 'a/b'])
def test_candidate_id_cannot_escape_output_root(sources, identifier):
    root, old, _, batch = sources
    with pytest.raises(ValueError, match='Candidate ID'):
        candidate.train_candidate([batch], old, identifier, repo=root)
