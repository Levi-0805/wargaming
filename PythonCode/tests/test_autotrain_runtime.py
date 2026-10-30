from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from autotrain_runtime import run_metadata, stop_reason
import autotrain_runtime as runtime


def test_metadata_binds_actual_owned_pid_and_batch(tmp_path):
    for name, batch, pid in [('old', 'other', 7), ('run', 'batch', 8), ('unrelated', 'batch', 9)]:
        p = tmp_path / name
        p.mkdir()
        (p / 'run.json').write_text(json.dumps(dict(batch_id=batch, process_id=pid)))
    assert run_metadata(tmp_path, 'batch', 8) == tmp_path / 'run'
    assert run_metadata(tmp_path, 'batch', 10) is None
    extra = tmp_path / 'duplicate'; extra.mkdir()
    (extra / 'run.json').write_text(json.dumps(dict(batch_id='batch', process_id=8)))
    with pytest.raises(ValueError, match='Multiple'):
        run_metadata(tmp_path, 'batch', 8)


@pytest.mark.parametrize('overrides,expected', [
    ({}, None), ({'stop_requested': True}, 'requested_stop'),
    ({'free_bytes': 9}, 'low_disk_space'), ({'elapsed': 120}, 'batch_timeout'),
    ({'idle': 10}, 'progress_timeout'),
])
def test_watchdog_limits(overrides, expected):
    args = dict(stop_requested=False, free_bytes=20, minimum_free_bytes=10,
                elapsed=30, total_timeout=120, idle=2, idle_timeout=10)
    args.update(overrides)
    assert stop_reason(**args) == expected


@pytest.mark.parametrize('failure', ['process_record', 'metadata'])
def test_owned_process_is_reaped_and_failure_receipt_written(tmp_path, monkeypatch, failure):
    config = tmp_path / 'config.json'; config.write_text('{}')
    manifest = dict(batch_id='batch', source_commit='abc', config_sha256=runtime.sha(config),
                    config_path=str(config), cssim_root=str(tmp_path / 'cssim'),
                    repo=str(tmp_path), log_root=str(tmp_path / 'logs'),
                    runtime_exit_path=str(tmp_path / 'runtime-exit.json'))
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps(manifest))
    monkeypatch.setattr(runtime, 'status', lambda *_: {'state': 'running'})
    class Child:
        pid = 123
        returncode = None
        def poll(self): return self.returncode
        def terminate(self): self.returncode = 1
        def wait(self, timeout): return self.returncode
        def kill(self): self.returncode = 2
    child = Child()
    monkeypatch.setattr(runtime.subprocess, 'Popen', lambda *a, **k: child)
    original_write = runtime.write_json
    def write(file, data):
        if failure == 'process_record' and file.name == 'runtime-process.json':
            raise OSError('record write failed')
        return original_write(file, data)
    monkeypatch.setattr(runtime, 'write_json', write)
    if failure == 'metadata':
        def broken_metadata(*_): raise ValueError('ambiguous metadata')
        monkeypatch.setattr(runtime, 'run_metadata', broken_metadata)
    assert runtime.run(path, minimum_free_gb=.000001) == 2
    assert child.returncode is not None
    receipt = json.loads((tmp_path / 'runtime-exit.json').read_text())
    assert receipt['exit_code'] == 1
    assert receipt['stop_reason'] == 'supervisor_error'
    assert receipt['error']
