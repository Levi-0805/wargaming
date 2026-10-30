from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import batch_training as batch


def config():
    return {
        'config.py->GlobalConfig': dict(num_worker_threads=1, device='cpu', max_episode=10, seed=10098,
            AlgorithmConfig_Red=batch.RED + '.py->AlgorithmConfig',
            AlgorithmConfig_Blue=batch.BLUE + '.py->AlgorithmConfig'),
        'wrapper.py->ScenarioConfig': dict(MaxStepsPerEpisode=500, StepGameTime=.5, TimeDilation=1,
            Algorithm_Names=[batch.RED + '->ReinforceAgentAlgorithm', batch.BLUE + '->ReinforceAgentAlgorithm']),
        batch.RED + '.py->AlgorithmConfig': dict(mode='hybrid_iql', iql_shadow=False,
                                                iql_model='Models/target_iql_v1.json')}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    repo, cssim = tmp_path / 'repo', tmp_path / 'CSSIM'
    model = repo / 'PythonCode' / batch.MODEL
    model.parent.mkdir(parents=True)
    model.write_text('{}')
    cfg = repo / 'effective.json'
    batch.write_json(cfg, config())
    monkeypatch.setattr(batch, 'verify_sources', lambda *a, **kw: 'commit-123')
    monkeypatch.setattr(batch, 'verify_remote', lambda *a: dict(remote='origin', ref='refs/heads/main', commit='commit-123'))
    manifest = batch.prepare(repo, cssim, cfg, 'test-10')
    return Path(manifest['manifest']), manifest


def packet(side):
    return dict(Score={'unit': {'Name': 'unit', 'SumWorth': 120}}, Reward=1600,
                Events=[{'摧毁方': side, '被摧毁方': 'blue' if side == 'red' else 'red',
                         '得分': '红方120' if side == 'red' else '蓝方120'}])


def add_run(manifest, count, *, ue='10;1', model=None, partial=False, run_end=True):
    directory = Path(manifest['log_root']) / 'new-run'
    directory.mkdir(parents=True, exist_ok=True)
    batch.write_json(directory / 'run.json', dict(batch_id=manifest['batch_id'],
        source_commit=manifest['source_commit'], config_sha256=manifest['config_sha256'], process_id=4321))
    protocol, simulation = [], []
    for episode in range(1, count + 1):
        for payload in ({'propertyArr': [1]}, {'dataArr': [1], 'dataGlobal': {'timeCnt': 500, 'rSVD1': ue}},
                        packet('red'), packet('blue')):
            protocol.append(dict(direction='ue_to_python', payload=payload))
        simulation.extend([
            dict(kind='reset', episode=episode, training=True, policies=[
                dict(mode='hybrid_iql', model_sha256=model or manifest['model_sha256'], shadow=False),
                dict(mode='rule', **{'class': 'TeamAlg.BlueModel.Algorithm.RLAgentAlgorithm.Agent'})]),
            dict(episode=episode, step=500, done=True)])
    if run_end:
        simulation.append(dict(kind='run_end', status='returned', error_type=None))
    for name, rows in (('protocol.jsonl', protocol), ('simulation.jsonl', simulation)):
        (directory / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    if partial:
        with (directory / 'simulation.jsonl').open('a', encoding='utf-8') as stream:
            stream.write('{"episode":')
    return directory


def receipt(manifest, directory, exit_code=0, **changes):
    start = datetime.fromisoformat(manifest['created_at']) + timedelta(seconds=1)
    value = dict(batch_id=manifest['batch_id'], source_commit=manifest['source_commit'],
                 config_sha256=manifest['config_sha256'], log_directory=str(directory), pid=4321,
                 started_at=start.isoformat(), ended_at=(start + timedelta(minutes=40)).isoformat(), exit_code=exit_code)
    value.update(changes)
    batch.write_json(manifest['runtime_exit_path'], value)


def test_prepare_records_config_and_preexisting_logs(prepared):
    path, manifest = prepared
    assert path.is_file()
    assert manifest['source_commit'] == 'commit-123'
    assert manifest['remote_verification']['commit'] == manifest['source_commit']
    assert manifest['runtime_config'] == config()
    assert manifest['config_sha256'] == batch.sha(manifest['config_path'])
    assert batch.status(path)['state'] == 'awaiting_runtime'


@pytest.mark.parametrize('change', ['episodes', 'mode', 'shadow', 'cpu', 'algorithm'])
def test_invalid_effective_configuration_rejected(tmp_path, change):
    value = config()
    g = value['config.py->GlobalConfig']
    red = value[batch.RED + '.py->AlgorithmConfig']
    if change == 'episodes': g['max_episode'] = 1
    elif change == 'mode': red['mode'] = 'rule'
    elif change == 'shadow': red['iql_shadow'] = True
    elif change == 'cpu': g['device'] = 'cuda'
    elif change == 'algorithm': value['wrapper.py->ScenarioConfig']['Algorithm_Names'][0] = 'Other->Agent'
    path = tmp_path / 'config.json'
    batch.write_json(path, value)
    with pytest.raises(ValueError): batch.validate_config(path)


def test_one_terminal_game_and_crash_are_partial_not_success(prepared):
    path, manifest = prepared
    directory = add_run(manifest, 1)
    assert batch.status(path)['complete_episodes'] == 1
    receipt(manifest, directory, exit_code=1)
    assert batch.status(path)['state'] == 'partial'
    with pytest.raises(ValueError, match='not complete'): batch.finish(path)


def test_ten_scores_need_actual_runtime_exit_receipt(prepared):
    path, manifest = prepared
    directory = add_run(manifest, 10)
    assert not batch.status(path)['ready_to_finish']
    receipt(manifest, directory)
    result = batch.finish(path)
    assert result['complete_episodes'] == 10
    saved = batch.read_json(result['result_path'])
    assert [r['scores']['red']['total'] for r in saved['match']['rounds']] == [1720] * 10
    assert saved['source_commit'] == 'commit-123'
    assert batch.finish(path)['result_path'] == result['result_path']


def test_ten_scores_do_not_hide_failed_process_exit(prepared):
    path, manifest = prepared
    directory = add_run(manifest, 10)
    receipt(manifest, directory, exit_code=2)
    assert batch.status(path)['state'] == 'partial'
    with pytest.raises(ValueError): batch.finish(path)


def test_ue_episode_override_and_loaded_model_mismatch_rejected(prepared):
    path, manifest = prepared
    add_run(manifest, 1, ue='1;1')
    with pytest.raises(ValueError, match='UE overrode'): batch.status(path)
    add_run(manifest, 1, model='wrong-model')
    with pytest.raises(ValueError, match='loaded red model'): batch.status(path)


def test_receipt_cannot_bind_another_runtime(prepared):
    path, manifest = prepared
    directory = add_run(manifest, 10)
    receipt(manifest, directory, config_sha256='wrong')
    with pytest.raises(ValueError, match='receipt config_sha256'): batch.finish(path)


def test_duplicate_done_or_missing_final_simulation_record_rejected(prepared):
    path, manifest = prepared
    directory = add_run(manifest, 10, partial=True)
    receipt(manifest, directory)
    assert not batch.status(path)['ready_to_finish']


def test_incremental_monitor_keeps_partial_line_until_complete(tmp_path):
    path = tmp_path / 'simulation.jsonl'
    path.write_bytes(b'{"episode":1,"step":2,"done":false}\n{"episode":1,')
    cache = batch.incremental_simulation(path, {})
    assert cache['done'] == [] and cache['partial_tail']
    with path.open('ab') as stream: stream.write(b'"step":3,"done":true}\n')
    cache = batch.incremental_simulation(path, cache)
    assert cache['done'] == [1] and not cache['partial_tail']
    assert batch.incremental_simulation(path, cache)['done'] == [1]
    path.write_bytes(b'')
    with pytest.raises(ValueError, match='truncated'): batch.incremental_simulation(path, cache)


def test_multiple_new_protocols_and_config_changes_fail_closed(prepared):
    path, manifest = prepared
    add_run(manifest, 1)
    other = Path(manifest['log_root']) / 'another-run'
    other.mkdir()
    (other / 'protocol.jsonl').write_text('')
    with pytest.raises(ValueError, match='Multiple new'): batch.status(path)
    Path(manifest['config_path']).write_text('{}')
    with pytest.raises(ValueError, match='config changed'): batch.status(path)


def test_prepare_cannot_overwrite_existing_batch(prepared):
    _, manifest = prepared
    with pytest.raises(ValueError, match='already exists'):
        batch.prepare(manifest['repo'], manifest['cssim_root'], manifest['config_path'], manifest['batch_id'])


def test_absolute_model_must_resolve_to_verified_deployment(tmp_path):
    root = tmp_path / 'PythonCode'
    value = config()
    value[batch.RED + '.py->AlgorithmConfig']['iql_model'] = str(root / batch.MODEL)
    cfg = tmp_path / 'config.json'
    batch.write_json(cfg, value)
    assert batch.validate_config(cfg, pythoncode_root=root) == value
    with pytest.raises(ValueError, match='model path'):
        batch.validate_config(cfg, pythoncode_root=tmp_path / 'Other')


def test_runtime_identity_cannot_be_from_another_process(prepared):
    path, manifest = prepared
    directory = add_run(manifest, 10)
    receipt(manifest, directory, pid=9999)
    with pytest.raises(ValueError, match='actual run.json process'):
        batch.finish(path)


def test_crash_before_any_protocol_is_partial(prepared):
    path, manifest = prepared
    receipt(manifest, None, exit_code=1, log_directory=None)
    result = batch.status(path)
    assert result['state'] == 'partial'
    assert result['complete_episodes'] == 0 and result['runtime_exit_code'] == 1


@pytest.mark.parametrize('change', [dict(stop_reason='supervisor_error'), dict(error='monitor failed')])
def test_supervisor_error_blocks_success_even_if_child_exited_zero(prepared, change):
    path, manifest = prepared
    directory = add_run(manifest, 10)
    receipt(manifest, directory, **change)
    assert not batch.status(path)['ready_to_finish']


@pytest.mark.parametrize('ending', ['missing', 'failed', 'duplicate', 'record_after_end'])
def test_only_one_final_successful_run_end_allows_completion(prepared, ending):
    path, manifest = prepared
    directory = add_run(manifest, 10, run_end=False)
    rows = []
    if ending == 'failed': rows = [dict(kind='run_end', status='failed', error_type='RuntimeError')]
    elif ending == 'duplicate': rows = [dict(kind='run_end', status='returned')] * 2
    elif ending == 'record_after_end': rows = [dict(kind='run_end', status='returned'), dict(episode=10, step=500)]
    with (directory / 'simulation.jsonl').open('a', encoding='utf-8') as stream:
        for row in rows: stream.write(json.dumps(row) + '\n')
    receipt(manifest, directory)
    assert not batch.status(path)['ready_to_finish']


def test_source_guard_covers_all_tools(tmp_path, monkeypatch):
    calls = []
    def fake_git(repo, *arguments):
        calls.append(arguments)
        return 'commit-123' if arguments[0] == 'rev-parse' else ' M tools/autotrain_runtime.py'
    monkeypatch.setattr(batch, 'git', fake_git)
    with pytest.raises(ValueError, match='not clean'):
        batch.verify_sources(tmp_path, tmp_path)
    assert calls[-1] == ('status', '--porcelain', '--', 'PythonCode', 'tools')


@pytest.mark.parametrize('remote', ['different\trefs/heads/main', '', 'commit-123\trefs/heads/other'])
def test_prepare_remote_guard_rejects_unpublished_head(tmp_path, monkeypatch, remote):
    monkeypatch.setattr(batch, 'git', lambda _, *args: 'main' if args[0] == 'branch' else remote)
    with pytest.raises(ValueError, match='origin/main'):
        batch.verify_remote(tmp_path, 'commit-123')


def test_prepare_records_one_remote_verification(tmp_path, monkeypatch):
    calls = []
    def fake_git(_, *args):
        calls.append(args)
        return 'main' if args[0] == 'branch' else 'commit-123\trefs/heads/main'
    monkeypatch.setattr(batch, 'git', fake_git)
    evidence = batch.verify_remote(tmp_path, 'commit-123')
    assert evidence['commit'] == 'commit-123'
    assert evidence['verified_at']
    assert calls == [('branch', '--show-current'), ('ls-remote', '--exit-code', 'origin', 'refs/heads/main')]


def test_prepare_remote_guard_requires_main_branch(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, 'git', lambda *args: 'codex/unpublished')
    with pytest.raises(ValueError, match='main branch'):
        batch.verify_remote(tmp_path, 'commit-123')


def test_status_never_requires_remote_network(prepared, monkeypatch):
    path, manifest = prepared
    def forbidden(*args):
        raise AssertionError('Status must not query origin')
    monkeypatch.setattr(batch, 'verify_remote', forbidden)
    assert batch.status(path)['state'] == 'awaiting_runtime'


def test_unique_temporary_files_allow_concurrent_atomic_writes(tmp_path):
    path = tmp_path / 'progress.json'
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda n: batch.write_json(path, {'value': n, 'payload': 'x' * 2000}), range(30)))
    assert batch.read_json(path)['value'] in range(30)
    assert not list(tmp_path.glob('*.tmp'))


def test_status_lock_serializes_another_process(tmp_path):
    lock, marker = tmp_path / 'status.lock', tmp_path / 'entered'
    program = """import sys
from pathlib import Path
from batch_training import StatusLock
print('ready', flush=True)
with StatusLock(Path(sys.argv[1])):
    Path(sys.argv[2]).write_text('entered')
"""
    child = None
    try:
        with batch.StatusLock(lock):
            child = subprocess.Popen([sys.executable, '-c', program, str(lock), str(marker)],
                                     cwd=Path(batch.__file__).parent, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
            assert child.stdout.readline().strip() == 'ready'
            assert child.poll() is None and not marker.exists()
        _, error = child.communicate(timeout=15)
        assert child.returncode == 0, error
        assert marker.read_text() == 'entered'
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate()
