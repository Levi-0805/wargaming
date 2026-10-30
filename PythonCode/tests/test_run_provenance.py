import hashlib
import json
import sys
from pathlib import Path

import pytest

from cssim.config import RuntimeConfig
from cssim.logging_setup import SimulationLog
from main import CssimApplication


def test_same_second_runs_never_append_to_existing_session(tmp_path, monkeypatch):
    monkeypatch.setattr("cssim.logging_setup.strftime", lambda _: "20261031_120000")
    first = SimulationLog(tmp_path, detailed=True)
    second = SimulationLog(tmp_path, detailed=True)
    try:
        first.write_simulation_record({"batch": "first"})
        second.write_simulation_record({"batch": "second"})
        first.write_protocol("ue_to_python", "frame", "step", '{"first":1}')
        second.write_protocol("ue_to_python", "frame", "step", '{"second":2}')
    finally:
        first.close()
        second.close()
    assert first.directory.name == "20261031_120000"
    assert second.directory.name == "20261031_120000-001"
    assert json.loads(first.stats_path.read_text()) == {"batch": "first"}
    assert json.loads(second.stats_path.read_text()) == {"batch": "second"}
    assert json.loads(first.protocol_path.read_text())["payload"] == {"first": 1}
    assert json.loads(second.protocol_path.read_text())["payload"] == {"second": 2}


def test_metadata_is_complete_before_publication_and_cannot_be_overwritten(tmp_path, monkeypatch):
    import cssim.logging_setup as logs

    run_log = SimulationLog(tmp_path)
    metadata = run_log.directory / "run.json"
    record = {"batch_id": "first", "source_commit": "a" * 40}
    original_dump, original_link = logs.json.dump, logs.os.link

    def slow_dump(value, stream, **kwargs):
        stream.write('{"unfinished":')
        stream.flush()
        assert not metadata.exists()
        stream.seek(0)
        stream.truncate()
        original_dump(value, stream, **kwargs)

    def checked_link(source, destination):
        assert not metadata.exists()
        assert json.loads(Path(source).read_text(encoding="utf-8")) == record
        return original_link(source, destination)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(logs.json, "dump", slow_dump)
            patch.setattr(logs.os, "link", checked_link)
            run_log.write_run_metadata(record)
        assert json.loads(metadata.read_text(encoding="utf-8")) == record
        with pytest.raises(FileExistsError):
            run_log.write_run_metadata({"batch_id": "replacement"})
        assert json.loads(metadata.read_text(encoding="utf-8")) == record
        assert not list(run_log.directory.glob(".run-*.tmp"))
    finally:
        run_log.close()


@pytest.mark.parametrize("failure_stage", ["serialization", "publication"])
def test_failed_metadata_write_leaves_no_partial_file(tmp_path, monkeypatch, failure_stage):
    import cssim.logging_setup as logs

    run_log = SimulationLog(tmp_path)

    def fail_dump(value, stream, **kwargs):
        stream.write('{"unfinished":')
        stream.flush()
        raise OSError("serialization interrupted")

    def fail_link(*args):
        raise OSError("publication interrupted")

    if failure_stage == "serialization":
        monkeypatch.setattr(logs.json, "dump", fail_dump)
    else:
        monkeypatch.setattr(logs.os, "link", fail_link)
    try:
        with pytest.raises(OSError, match="interrupted"):
            run_log.write_run_metadata({"batch_id": "attempt"})
        assert not (run_log.directory / "run.json").exists()
        assert not list(run_log.directory.glob(".run-*.tmp"))
    finally:
        run_log.close()


def _application(tmp_path, monkeypatch, *, fail=False, explicit_provenance=True):
    import cssim.runtime

    config_file = tmp_path / "config.jsonc"
    config_file.write_text(json.dumps({
        "config.py->GlobalConfig": {"max_episode": 10, "private_token": "do-not-record"},
        "Common.cssim_env_wrapper.py->ScenarioConfig": {
            "MaxStepsPerEpisode": 500, "Algorithm_Names": ["red->Policy", "blue->Policy"],
        },
    }), encoding="utf-8")
    log_root = tmp_path / "batch" / "attempt" / "logs"
    monkeypatch.setattr(sys, "argv", ["main.py", "-c", str(config_file), "--log-root", str(log_root)])
    if explicit_provenance:
        monkeypatch.setenv("CSSIM_SOURCE_COMMIT", "a" * 40)
        monkeypatch.setenv("CSSIM_BATCH_ID", "batch-001-attempt-002")
    else:
        monkeypatch.delenv("CSSIM_SOURCE_COMMIT", raising=False)
        monkeypatch.delenv("CSSIM_BATCH_ID", raising=False)
    monkeypatch.setattr(CssimApplication, "seed_random_generators", lambda _: None)
    monkeypatch.setattr(SimulationLog, "configure", classmethod(lambda cls, root, level: cls(root, detailed=True)))

    class StubRunner:
        def __init__(self, config, run_log):
            self.log = run_log

        async def run(self):
            if fail:
                raise ConnectionError("simulated disconnect")
            self.log.write_simulation_record({"kind": "reset", "episode": 1})

    monkeypatch.setattr(cssim.runtime, "Runner", StubRunner)
    return config_file, log_root


@pytest.mark.parametrize("explicit_provenance", [False, True])
def test_application_uses_requested_log_root_and_only_public_metadata(tmp_path, monkeypatch, explicit_provenance):
    config, root = _application(tmp_path, monkeypatch, explicit_provenance=explicit_provenance)
    CssimApplication.run()
    sessions = list(root.iterdir())
    assert len(sessions) == 1
    metadata_text = (sessions[0] / "run.json").read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)
    assert "do-not-record" not in metadata_text and "private_token" not in metadata_text
    assert metadata["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert metadata["config_path"] == str(config.resolve())
    assert metadata["algorithm_names"] == ["red->Policy", "blue->Policy"]
    assert metadata["configured_max_episodes"] == 10
    assert metadata["source_commit"] == ("a" * 40 if explicit_provenance else None)
    assert metadata["source_commit_evidence"] == ("provided_by_launcher" if explicit_provenance else "unknown")
    assert metadata["batch_id"] == ("batch-001-attempt-002" if explicit_provenance else None)
    records = [json.loads(line) for line in (sessions[0] / "simulation.jsonl").read_text().splitlines()]
    assert records[-1]["kind"] == "run_end"
    assert records[-1]["status"] == "returned"
    assert records[-1]["error_type"] is None


def test_application_records_failure_and_preserves_exception(tmp_path, monkeypatch):
    _, root = _application(tmp_path, monkeypatch, fail=True)
    with pytest.raises(ConnectionError, match="simulated disconnect"):
        CssimApplication.run()
    session = next(root.iterdir())
    end = json.loads((session / "simulation.jsonl").read_text().splitlines()[-1])
    assert end["status"] == "failed"
    assert end["error_type"] == "ConnectionError"
    assert "simulated disconnect" not in json.dumps(end)


def test_configuration_change_during_load_is_rejected(tmp_path, monkeypatch):
    config, root = _application(tmp_path, monkeypatch)
    original = RuntimeConfig.load

    def changing_load(path):
        value = original(path)
        config.write_text(config.read_text() + "\n")
        return value

    monkeypatch.setattr(RuntimeConfig, "load", changing_load)
    with pytest.raises(RuntimeError, match="配置在读取期间发生变化"):
        CssimApplication.run()
    assert not root.exists()


@pytest.mark.parametrize('fail', [False, True])
def test_managed_exit_follows_closed_success_audit_only(tmp_path, monkeypatch, fail):
    _, root = _application(tmp_path, monkeypatch, fail=fail)
    monkeypatch.setattr(sys, 'argv', [*sys.argv, '--exit-after-run'])
    monkeypatch.setattr('main.logging.shutdown', lambda: None)
    exited = []
    def exit_process(code):
        session = next(root.iterdir())
        end = json.loads((session/'simulation.jsonl').read_text().splitlines()[-1])
        assert end['status'] == 'returned'
        exited.append(code)
    monkeypatch.setattr('main.os._exit', exit_process)
    if fail:
        with pytest.raises(ConnectionError):
            CssimApplication.run()
        assert exited == []
    else:
        CssimApplication.run()
        assert exited == [0]


def test_launch_cli_keeps_defaults_and_overrides_environment(monkeypatch):
    monkeypatch.setenv("CSSIM_SOURCE_COMMIT", "environment-commit")
    monkeypatch.setenv("CSSIM_BATCH_ID", "environment-batch")
    monkeypatch.setattr(sys, "argv", ["main.py", "-c", "config.jsonc", "-s",
                                    "--source-commit", "explicit-commit", "--batch-id", "explicit-batch"])
    args = CssimApplication.parse_arguments()
    assert args.log_root == "logs"
    assert args.source_commit == "explicit-commit"
    assert args.batch_id == "explicit-batch"
    assert args.s
