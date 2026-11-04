import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import cssim_single_round as module


def ledger(count):
    return dict(state='needs_attention', completed_games=count, source_commit='source-a',
        error='report push failed', attempts=[dict(batch_id='one', supervisor_exit_code=1,
                                                  runtime_receipt=dict(exit_code=1))],
        games=[dict(protocol='protocol', episode=i, protocol_sha256='protocol-sha',
                    source_commit='source-a', model_sha256='model-a',
                    red=dict(kill_score=900, capture_score=1600, total=2500),
                    blue=dict(kill_score=800, capture_score=0, total=800)) for i in range(count)])


def launch():
    return dict(campaign_id='one', report_round=12, worker=dict(pid=12),
                campaign_state='campaign/state.json', collection_report='result.json')


def test_root_stop_is_forwarded_after_existing_campaign_guard(tmp_path, monkeypatch):
    checks = []
    monkeypatch.setattr(module.CumulativeCollection, 'guard', lambda _: checks.append('campaign'))
    collector = module.SingleRoundCollection.__new__(module.SingleRoundCollection)
    collector.s = SimpleNamespace(repo=tmp_path)
    stop = tmp_path / 'logs/learning/STOP'
    stop.parent.mkdir(parents=True)
    stop.touch()
    with pytest.raises(module.CampaignStop, match='Global learning STOP'):
        collector.guard()
    assert checks == ['campaign']


def test_reports_preserve_failed_exit_and_require_ten_valid_same_version_games(tmp_path):
    complete = ledger(10)
    control = tmp_path / 'complete'
    assert module.write_exit_report(complete, launch(), control, tmp_path / 'reports', 2)
    signal = json.loads((control / 'report-ready.json').read_text(encoding='utf-8'))
    assert signal['completed_games'] == 10 and not signal['report_published']
    assert signal['delivered_to_chat'] is False
    report = Path(signal['report_path'])
    assert report.read_bytes().startswith(b'\xef\xbb\xbf')
    text = report.read_text(encoding='utf-8-sig')
    assert '900 + 1600 = 2500' in text and 'runtime=1' in text
    assert 'GitHub备份：未核实' in text
    partial_control = tmp_path / 'partial'
    assert not module.write_exit_report(ledger(4), launch(), partial_control, tmp_path / 'partial-reports', 2)
    assert not (partial_control / 'report-ready.json').exists()
    partial = (tmp_path / 'partial-reports/第0012轮/复测状态.txt').read_text(encoding='utf-8-sig')
    assert '4/10' in partial and '尚未完成十局' in partial
    for field in ('arithmetic', 'source', 'model'):
        invalid = copy.deepcopy(complete)
        if field == 'arithmetic':
            invalid['games'][-1]['red']['total'] += 1
        else:
            invalid['games'][-1]['source_commit' if field == 'source' else 'model_sha256'] = 'other'
        with pytest.raises(ValueError):
            module.validated_games(invalid)


def test_borrowed_server_reuse_blocks_launch_without_cleanup_or_ownership():
    expected = dict(pid=5, created_ticks=100, exe_path='C:/CSSIM/Server/Server/Server.exe')
    calls = []
    backend = SimpleNamespace(processes=SimpleNamespace(identity=lambda _: {**expected, 'created_ticks': 101}),
                              launch_apps=lambda _: calls.append('launched'))
    with pytest.raises(module.CampaignBlocked, match='identity changed'):
        module.BoundBackend(backend, [expected]).launch_apps({})
    assert calls == []


def test_entry_runs_once_and_preserves_interrupted_learner(tmp_path, monkeypatch):
    old = tmp_path / 'logs/learning/state.json'
    old.parent.mkdir(parents=True)
    old.write_text('{"phase":"needs_attention","round_number":11,"completed_games":4,"worker":{"pid":77,"created_ticks":200}}')
    before = old.read_bytes()
    calls = []
    identity = dict(pid=1, created_ticks=100, exe_path='python.exe')
    rows = [dict(pid=77, created_ticks=200)]
    monkeypatch.setattr(module, 'WindowsProcesses', lambda: SimpleNamespace(identity=lambda _: identity, snapshot=lambda: rows))
    monkeypatch.setattr(module, 'desktop_reports', lambda: tmp_path / 'reports')
    class Collector:
        def __init__(self, settings, servers, launch):
            self.state = ledger(10)
        def run(self):
            calls.append('run')
            return 2
    monkeypatch.setattr(module, 'SingleRoundCollection', Collector)
    args = SimpleNamespace(campaign_id='one', report_round=12, cssim_root=tmp_path / 'CSSIM',
                           server_pid=None, additional_server_pid=[], adopt_client_pid=None)
    with pytest.raises(module.CampaignBlocked, match='worker is still live'):
        module.run_once(args, repo=tmp_path)
    assert calls == [] and old.read_bytes() == before
    rows.clear()
    assert module.run_once(args, repo=tmp_path) == 2
    assert calls == ['run'] and old.read_bytes() == before
    assert (tmp_path / 'logs/single-round/one/report-ready.json').exists()
    with pytest.raises(ValueError, match='fresh campaign'):
        module.run_once(args, repo=tmp_path)
