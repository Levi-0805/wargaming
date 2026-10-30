import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import cssim_campaign as module
from cssim_campaign import (Campaign, CampaignBlocked, RealBackend, Settings, WindowsProcesses,
                            descendant_candidates, exact_platform_command, ready_runtime, verified_ide_workdir)


class FakeBackend:
    def __init__(self, *, fail_attempts=(), cleanup_fails=False, commit_fails=False, free=100*1024**3):
        self.calls = []
        self.fail_attempts = set(fail_attempts)
        self.cleanup_fails, self.commit_fails, self.free = cleanup_fails, commit_fails, free
        self.attempt = 0

    def free_bytes(self, path):
        return self.free

    def preflight(self):
        self.calls.append('preflight')
        return 'a' * 40

    def launch_apps(self, context):
        self.attempt += 1
        self.calls.append('launch')

    def wait_client(self, context, guard):
        self.calls.append('window')
        guard()

    def flow(self, context, phase, guard):
        self.calls.append(phase)
        guard()
        if self.attempt in self.fail_attempts:
            raise RuntimeError('simulated client crash')

    def check_platform(self, context, guard):
        self.calls.append('pycharm_config')

    def resolve_auto_python(self, context):
        self.calls.append('conflict_check')

    def bind(self, context):
        self.calls.append('bind')
        return str(context['directory'] / 'manifest.json')

    def start_runtime(self, context):
        self.calls.append('runtime')

    def wait_ready(self, context, guard):
        self.calls.append('handshake')
        return {'pid': 123, 'directory': str(context['directory'])}

    def wait_result(self, context, guard, progress):
        self.calls.append('result')
        return dict(batch_id=context['batch_id'], result_path=str(context['directory'] / 'result.json'),
                    scores=[dict(episode=n, red=1000+n, blue=900) for n in range(1, 11)])

    def cleanup(self, context):
        self.calls.append('cleanup')
        if self.cleanup_fails:
            raise CampaignBlocked('owned dedicated server did not exit')

    def commit_report(self, result):
        self.calls.append('commit_report')
        if self.commit_fails:
            raise RuntimeError('push failed')
        return 'b' * 40

    def candidate(self, context, reports, guard):
        self.calls.append('candidate')
        return dict(state='trained_unpublished', promoted=False)


def campaign(tmp_path, backend=None, **settings):
    return Campaign(Settings('test', tmp_path / 'recipe.json', repo=tmp_path, max_batches=1,
                             retry_seconds=0, **settings), backend=backend or FakeBackend(), sleep=lambda _: None)


def test_success_requires_ready_python_before_play_then_cleanup_and_report_commit(tmp_path):
    runner = campaign(tmp_path)
    assert runner.run() == 0
    assert runner.backend.calls == ['preflight', 'launch', 'window', 'pre_python', 'pycharm_config',
                                    'conflict_check', 'bind', 'runtime', 'handshake', 'post_python',
                                    'result', 'cleanup', 'commit_report']
    record = runner.state['attempts'][0]
    assert len(record['scores']) == 10
    assert record['source_commit'] == 'a' * 40 and record['report_commit'] == 'b' * 40
    assert runner.state['state'] == 'completed'


def test_three_crashes_are_preserved_and_stop_retries(tmp_path):
    backend = FakeBackend(fail_attempts={1, 2, 3, 4})
    runner = campaign(tmp_path, backend)
    assert runner.run() == 2
    assert backend.calls.count('launch') == backend.calls.count('cleanup') == 3
    assert runner.state['consecutive_failures'] == 3
    assert len(runner.state['attempts']) == 3
    assert all((runner.directory / f'attempt-{n:04d}' / 'attempt.json').exists() for n in range(1, 4))
    assert 'commit_report' not in backend.calls


def test_success_after_crash_resets_failure_counter(tmp_path):
    runner = campaign(tmp_path, FakeBackend(fail_attempts={1}))
    assert runner.run() == 0
    assert len(runner.state['attempts']) == 2
    assert runner.state['consecutive_failures'] == 0


@pytest.mark.parametrize('cause', ['stop', 'disk'])
def test_stop_or_low_disk_never_launches_applications(tmp_path, cause):
    backend = FakeBackend(free=0 if cause == 'disk' else 100*1024**3)
    runner = campaign(tmp_path, backend)
    if cause == 'stop':
        runner.stopfile.touch()
    assert runner.run() == 2
    assert 'launch' not in backend.calls
    assert runner.state['state'] == 'stopped'


def test_cleanup_failure_stops_without_retry_or_committing_success(tmp_path):
    runner = campaign(tmp_path, FakeBackend(cleanup_fails=True))
    assert runner.run() == 2
    assert runner.backend.calls.count('launch') == 1
    assert 'commit_report' not in runner.backend.calls
    assert 'cleanup_error' in runner.state['attempts'][0]


def test_report_push_failure_preserves_score_and_stops(tmp_path):
    runner = campaign(tmp_path, FakeBackend(commit_fails=True))
    assert runner.run() == 2
    assert len(runner.state['attempts'][0]['scores']) == 10
    assert 'push failed' in runner.state['attempts'][0]['postprocessing_error']
    assert runner.backend.calls.count('launch') == 1


def test_candidate_is_unpublished_and_runs_after_score_commit(tmp_path):
    runner = campaign(tmp_path, train_candidate=True)
    assert runner.run() == 0
    assert runner.backend.calls.index('commit_report') < runner.backend.calls.index('candidate')
    assert runner.state['attempts'][0]['candidate']['promoted'] is False


def test_gate_rejects_stale_release_before_another_launch(tmp_path):
    runner = Campaign(Settings('test', tmp_path / 'recipe.json', repo=tmp_path, max_batches=2,
                               retry_seconds=0, gate_after_success=True), FakeBackend(), sleep=lambda _: None)
    (runner.directory / 'gate-release.json').write_text(json.dumps({'campaign_id': 'test', 'gate_token': 'test:0'}))
    assert runner.run() == 2
    assert runner.backend.calls.count('launch') == 1
    assert 'does not match' in runner.state['error']


def test_gate_revalidates_published_source_before_continuing(tmp_path):
    runner = Campaign(Settings('test', tmp_path / 'recipe.json', repo=tmp_path, max_batches=2,
                               retry_seconds=0, gate_after_success=True), FakeBackend(), sleep=lambda _: None)

    def release(_):
        (runner.directory / 'gate-release.json').write_text(json.dumps(
            {'campaign_id': 'test', 'gate_token': runner.state['gate_token']}))

    runner.sleep = release
    assert runner.run() == 0
    assert runner.backend.calls.count('launch') == 2
    assert runner.backend.calls.count('preflight') == 3
    assert (runner.directory / 'gate-release-0001.json').exists()


def test_ownership_excludes_wrong_path_preexisting_and_reused_parent():
    owner = dict(pid=10, created_ticks=100, exe_path=r'C:\CSSIM\Server.exe')
    child = dict(pid=20, ppid=10, created_ticks=101, exe_path=r'C:\CSSIM\python.exe')
    old = dict(child, pid=21, created_ticks=99)
    wrong = dict(child, pid=22, exe_path=r'C:\Other\python.exe')
    allowed = [r'C:\CSSIM\python.exe']
    assert descendant_candidates([dict(owner, ppid=1), child, old, wrong], [owner], allowed) == [child]
    assert descendant_candidates([dict(owner, ppid=1, created_ticks=500), child], [owner], allowed) == []


def test_only_exact_default_main_and_config_may_be_stopped():
    code = r'C:\CSSIM\Client\Data\AlgData\PythonCode'
    main, config = code + r'\main.py', code + r'\TaskConfig\cssim\cssim_25vs25.jsonc'
    assert exact_platform_command(['python.exe', 'main.py', '-c', r'.\TaskConfig\cssim\cssim_25vs25.jsonc'], main, config, code)
    assert not exact_platform_command(['python.exe', 'other.py', '-c', config], main, config, code)
    assert not exact_platform_command(['python.exe', main, '-c', config, '--extra'], main, config, code)
    assert not exact_platform_command(['python.exe', '-c', 'dangerous_python_code'], main, config, code)


def test_runtime_readiness_needs_identity_handshake_and_prepare(tmp_path):
    directory = tmp_path / 'logs' / 'run'
    directory.mkdir(parents=True)
    manifest = dict(batch_id='b', source_commit='a'*40, config_sha256='c', log_root=str(tmp_path / 'logs'),
                    runtime_exit_path=str(tmp_path / 'runtime-exit.json'))
    (tmp_path / 'runtime-process.json').write_text(json.dumps(dict(pid=123)))
    (directory / 'run.json').write_text(json.dumps({k: manifest[k] for k in ('batch_id', 'source_commit', 'config_sha256')} | {'process_id': 123}))
    protocol = directory / 'protocol.jsonl'
    handshake = dict(direction='ue_to_python', channel='handshake')
    prepare = dict(direction='python_to_ue', channel='prepare', type='reset_property')
    protocol.write_text(json.dumps(handshake) + '\n')
    assert ready_runtime(manifest) is None
    protocol.write_text(json.dumps(handshake) + '\n' + json.dumps(prepare))
    assert ready_runtime(manifest) is None  # A partial append is never accepted.
    protocol.write_text(json.dumps(handshake) + '\n' + json.dumps(prepare) + '\n')
    assert ready_runtime(manifest)['pid'] == 123


def test_first_report_commit_expands_untracked_directory_without_staging_other_files(tmp_path, monkeypatch):
    backend = RealBackend.__new__(RealBackend)
    backend.s = SimpleNamespace(repo=tmp_path)
    report = tmp_path / 'results/batches/first.json'
    calls = []

    def status(command, **kwargs):
        assert '--untracked-files=all' in command
        return b'?? results/batches/first.json\0'

    def git(repo, *args, **kwargs):
        calls.append(args)
        if args[0] == 'rev-parse':
            return 'a' * 40
        if args[0] == 'ls-remote':
            return 'a' * 40 + '\trefs/heads/main'
        if args[0] == 'log':
            return 'test-author'
        return ''

    monkeypatch.setattr(module.subprocess, 'check_output', status)
    monkeypatch.setattr(module, 'checked_git', git)
    assert backend.commit_report(dict(result_path=str(report), batch_id='first')) == 'a' * 40
    assert ('add', '--', 'results/batches/first.json') in calls
    assert ('push', 'origin', 'main') in calls


def test_relative_config_requires_exact_saved_project_run_configuration(tmp_path):
    project = tmp_path / 'PythonCode'
    idea = project / '.idea'
    idea.mkdir(parents=True)
    argv = ['python.exe', str(project / 'main.py'), '-c', r'.\TaskConfig\cssim\cssim_25vs25.jsonc']
    assert verified_ide_workdir(project, argv) is None
    content = '''<project><configuration type="PythonConfigurationType" name="main">
    <option name="SCRIPT_NAME" value="$PROJECT_DIR$/main.py" />
    <option name="WORKING_DIRECTORY" value="$PROJECT_DIR$" />
    <option name="PARAMETERS" value="-c .\\TaskConfig\\cssim\\cssim_25vs25.jsonc" />
    </configuration></project>'''
    path = idea / 'workspace.xml'
    path.write_text(content)
    assert verified_ide_workdir(project, argv)['source'] == 'saved_ide_run_configuration'
    path.write_text(content.replace('value="$PROJECT_DIR$"', 'value="C:/other"'))
    assert verified_ide_workdir(project, argv) is None
    path.write_text(content)
    assert verified_ide_workdir(project, ['python.exe', 'main.py', *argv[2:]]) is None


def test_reused_ide_binds_only_new_exact_default_python_and_never_owns_ide(tmp_path):
    backend = RealBackend.__new__(RealBackend)
    backend.paths = dict(pycharm=tmp_path / 'pycharm64.exe', server_pycharm=tmp_path / 'server_pycharm64.exe',
                         python=tmp_path / 'python.exe', dedicated=tmp_path / 'CSSIMServer-Win64-Shipping.exe')
    backend.code = tmp_path / 'PythonCode'
    backend.code.mkdir()
    backend.platform_config = backend.code / 'cssim_25vs25.jsonc'
    backend.platform_config.write_text('{}')
    backend.known_ides = []
    ide = dict(pid=10, ppid=1, created_ticks=50, exe_path=str(backend.paths['pycharm']), name='pycharm64.exe')
    child = dict(pid=20, ppid=10, created_ticks=120, exe_path=str(backend.paths['python']), name='python.exe')
    stopped = []
    backend.processes = SimpleNamespace(snapshot=lambda: [ide, child], identity=lambda pid: ide,
        command_line=lambda pid: [str(backend.paths['python']), str(backend.code / 'main.py'), '-c', str(backend.platform_config)],
        stop=lambda item: stopped.append(item['pid']))
    backend.desktop = SimpleNamespace(windows=lambda: [dict(hwnd=123, pid=10, exe_path=ide['exe_path'], title='PythonCode – main.py')])
    backend._wait = lambda context, guard, condition, seconds, description: condition()
    context = dict(directory=tmp_path, owned=[], baseline_pids={10}, started_ticks=100, config_before=None)
    backend.check_platform(context, lambda: None)
    backend.resolve_auto_python(context)
    assert stopped == [20]
    assert [p['identity']['pid'] for p in context['owned']] == [20]
    assert context['owned'][0]['role'] == 'platform_auto_python'
    assert backend.known_ides[0]['pid'] == 10

    # A child born before this batch is never claimed despite a matching argv.
    context['owned'] = []
    context['platform_python_candidates'] = []
    child['created_ticks'] = 90
    stopped.clear()
    with pytest.raises(CampaignBlocked, match='Unowned'):
        backend.resolve_auto_python(context)
    assert stopped == []


def test_unrelated_python_conflict_prevents_stopping_even_an_earlier_valid_child(tmp_path):
    backend = RealBackend.__new__(RealBackend)
    backend.paths = dict(python=tmp_path / 'python.exe')
    backend.code = tmp_path / 'PythonCode'
    backend.platform_config = backend.code / 'cssim_25vs25.jsonc'
    candidate = dict(pid=10, created_ticks=100, exe_path=str(backend.paths['python']))
    unrelated = dict(candidate, pid=20)
    stopped = []
    backend.refresh = lambda context: [candidate, unrelated]
    backend.processes = SimpleNamespace(
        command_line=lambda pid: [str(backend.paths['python']), str(backend.code / 'main.py'), '-c', str(backend.platform_config)],
        stop=lambda item: stopped.append(item['pid']))
    context = dict(owned=[], platform_python_candidates=[candidate])
    with pytest.raises(CampaignBlocked, match='Unowned'):
        backend.resolve_auto_python(context)
    assert stopped == []


def test_cleanup_accepts_an_already_exited_owned_child_with_retained_handle():
    processes = WindowsProcesses.__new__(WindowsProcesses)
    identity = dict(pid=123, created_ticks=100, exe_path=r'C:\CSSIM\python.exe')
    closed, killed = [], []
    processes.k = SimpleNamespace(OpenProcess=lambda *args: 42, WaitForSingleObject=lambda *args: 0,
                                   CloseHandle=lambda handle: closed.append(handle),
                                   TerminateProcess=lambda *args: killed.append(args))
    processes.identity = lambda pid, handle=None: identity
    processes.stop(identity)
    assert closed == [42] and killed == []


def test_source_change_during_ui_setup_prevents_manifest_and_runtime(tmp_path):
    backend = RealBackend.__new__(RealBackend)
    backend.preflight = lambda: 'new-commit'
    with pytest.raises(CampaignBlocked, match='Source commit changed'):
        backend.bind(dict(source_commit='original-commit', directory=tmp_path))
    assert not (tmp_path / 'effective.jsonc').exists()
