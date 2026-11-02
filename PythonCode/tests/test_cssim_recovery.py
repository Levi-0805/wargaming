"""Recovery must preserve results, process identity and the ten-game budget."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
import cssim_campaign as campaign
import cssim_cumulative as cumulative
import cssim_learning as learning


def interrupted(tmp_path, monkeypatch, count=0, commit='old'):
    settings = campaign.Settings('resume', tmp_path/'recipe.json', repo=tmp_path)
    folder = tmp_path/'logs/campaigns/resume'
    folder.mkdir(parents=True)
    games = [dict(source_commit='old', model_sha256='same', ordinal=i+1) for i in range(count)]
    state = dict(campaign_id='resume', state='needs_attention', target_games=10,
        completed_games=count, games=games, attempts=[dict(error='login timeout')]*3,
        no_progress_failures=3, source_commit='old', successful_reports=[])
    (folder/'state.json').write_text(json.dumps(state), encoding='utf-8')
    monkeypatch.setattr(cumulative, 'RealBackend', lambda _: SimpleNamespace(
        preflight=lambda:commit, free_bytes=lambda _:100*1024**3))
    return settings, folder, state


def test_resume_retains_games_failed_attempts_and_previous_state(tmp_path, monkeypatch):
    settings, folder, before = interrupted(tmp_path, monkeypatch, count=7)
    collector = cumulative.CumulativeCollection.resume(settings)
    assert collector.state['games'] == before['games']
    assert collector.state['attempts'] == before['attempts']
    assert collector.state['completed_games'] == 7
    assert collector.state['no_progress_failures'] == 0
    assert json.loads((folder/'recovery-0001-before.json').read_text()) == before
    budgets = []
    def attempt(number):
        budgets.append((number, 10-len(collector.state['games'])))
        collector.state['games'].extend([{}]*3)
    monkeypatch.setattr(collector, 'attempt', attempt)
    monkeypatch.setattr(collector, 'finish', lambda:None)
    assert collector.run() == 0
    assert budgets == [(4, 3)]


@pytest.mark.parametrize('count,allow,accepted', [(0,False,False),(0,True,True),(7,True,False)])
def test_code_rebind_is_explicit_and_only_for_zero_completed_games(tmp_path, monkeypatch, count, allow, accepted):
    settings, folder, before = interrupted(tmp_path, monkeypatch, count, 'new')
    if accepted:
        collector = cumulative.CumulativeCollection.resume(settings, rebind_empty=allow)
        assert collector.state['source_commit'] == 'new'
        assert collector.state['recoveries'][0]['source_commit_before'] == 'old'
    else:
        with pytest.raises(ValueError, match='code versions'):
            cumulative.CumulativeCollection.resume(settings, rebind_empty=allow)
        assert json.loads((folder/'state.json').read_text()) == before


def test_resume_cannot_ignore_stop(tmp_path, monkeypatch):
    settings, folder, before = interrupted(tmp_path, monkeypatch)
    (folder/'STOP').touch()
    with pytest.raises(ValueError, match='STOP'):
        cumulative.CumulativeCollection.resume(settings)
    assert json.loads((folder/'state.json').read_text()) == before


def client_backend(tmp_path):
    paths = dict(client=tmp_path/'CSSIMClient-Win64-Shipping.exe', manager=tmp_path/'Server.exe')
    client = dict(pid=10, created_ticks=100, exe_path=str(paths['client']), name=paths['client'].name)
    server = dict(pid=20, created_ticks=200, exe_path=str(paths['manager']), name=paths['manager'].name)
    stopped = []
    rows = [client, server]
    backend = campaign.RealBackend.__new__(campaign.RealBackend)
    backend.paths, backend.platform_config = paths, tmp_path/'platform.jsonc'
    backend.s = SimpleNamespace(server_pid=20, additional_server_pids=(), adopt_client=copy.deepcopy(client))
    backend.processes = SimpleNamespace(snapshot=lambda:rows, stop=lambda identity:stopped.append(identity))
    backend._save_processes = lambda context:None
    backend._start = lambda *args, **kwargs:pytest.fail('must not launch a second client or server')
    return backend, client, server, rows, stopped


def test_client_handoff_owns_only_the_explicit_creation_identity(tmp_path):
    backend, client, server, rows, stopped = client_backend(tmp_path)
    context = dict(owned=[])
    backend.launch_apps(context)
    assert [p['identity'] for p in context['owned']] == [client]
    assert context['borrowed_manager'] == server
    assert context['adopted_client']['pid'] == client['pid']
    assert backend.s.adopt_client is None
    process = context['owned'][0]['process']
    assert process.poll() is None
    process.terminate()
    assert stopped == [client]
    rows[0] = {**client, 'created_ticks':101}  # Reused PID is no longer our client.
    assert process.poll() == 0


@pytest.mark.parametrize('problem', ['reused_pid','another_client'])
def test_handoff_does_not_claim_unrelated_clients(tmp_path, problem):
    backend, client, server, rows, _ = client_backend(tmp_path)
    if problem == 'reused_pid': rows[0] = {**client, 'created_ticks':101}
    else: rows.append({**client, 'pid':11})
    context = dict(owned=[])
    with pytest.raises(campaign.CampaignBlocked):
        backend.launch_apps(context)
    assert not context['owned']


def test_resume_request_cannot_replace_a_live_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, 'STATE', tmp_path/'state.json')
    monkeypatch.setattr(learning, 'STOP', tmp_path/'STOP')
    state = dict(phase='needs_attention', worker={'pid':1})
    learning.STATE.write_text(json.dumps(state))
    monkeypatch.setattr(learning, 'live', lambda _:True)
    with pytest.raises(ValueError, match='stopped worker'):
        learning.resume_interrupted(SimpleNamespace(adopt_client_pid=None,rebind_empty=False))
    assert json.loads(learning.STATE.read_text()) == state
