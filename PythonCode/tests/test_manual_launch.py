import argparse
import hashlib
import json
from pathlib import Path
import sys

import pytest

from cssim import manual_launch as manual
from main import CssimApplication


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "PythonCode"
    team = root / f"TeamAlg/{manual.TEAM}"
    folder = team / "Algorithm/Champion6cb5a79"
    folder.mkdir(parents=True)
    (folder / "route.py").write_bytes(b"immutable historical route")
    snapshot = json.dumps({"files": {"route.py": hashlib.sha256((folder / "route.py").read_bytes()).hexdigest()}}).encode()
    (folder / "snapshot.json").write_bytes(snapshot)
    monkeypatch.setattr(manual, "SNAPSHOT_SHA", hashlib.sha256(snapshot).hexdigest())
    model = team / "Models/target_iql_v1.json"
    model.parent.mkdir()
    model.write_bytes(b"original numeric model")
    monkeypatch.setattr(manual, "MODEL_SHA", hashlib.sha256(model.read_bytes()).hexdigest())
    source = root / "TaskConfig/cssim/cssim_25vs25.jsonc"
    source.parent.mkdir(parents=True)
    raw = {
        "config.py->GlobalConfig": {"max_episode": 10, "num_worker_threads": 1},
        "Common.cssim_env_wrapper.py->ScenarioConfig": {
            "Algorithm_Names": [manual.RED + "->ReinforceAgentAlgorithm", "blue->Policy"],
            "CSSIMPort": 29999, "TcpAddr": "127.0.0.7", "MaxStepsPerEpisode": 500,
        },
        manual.RED + ".py->AlgorithmConfig": {"mode": "rule", "enable_continuous_column": True},
        "blue override": {"unchanged": True},
    }
    source.write_text(json.dumps(raw), encoding="utf-8")
    args = argparse.Namespace(config=None, strategy=None, non_interactive=False,
        batch_id=None, source_commit=None, s=False, episode_limit=None,
        exit_after_run=False, log_root=str(tmp_path / "logs"))
    return root, source, args, raw, folder, model


def resolve(project, answers):
    root, _, args, *_ = project
    replies = iter(answers)
    return manual.resolve_manual_launch(args, project_root=root,
        input_fn=lambda _: next(replies), output_fn=lambda _: None)


def test_direct_main_can_choose_guarded_highscore_and_preserves_live_platform(project):
    root, source, args, raw, *_ = project
    original = source.read_bytes()
    result = resolve(project, ["1", "10"])
    effective = json.loads(Path(args.config).read_bytes())
    policy = effective[manual.RED + ".py->AlgorithmConfig"]
    assert policy["policy_baseline"] == "6cb5a79_hq_guard_entry"
    assert policy["mode"] == "hybrid_iql" and policy["iql_shadow"] is False
    assert "enable_continuous_column" not in policy
    assert Path(policy["iql_model"]).is_absolute()
    assert effective["Common.cssim_env_wrapper.py->ScenarioConfig"] == raw["Common.cssim_env_wrapper.py->ScenarioConfig"]
    assert effective["blue override"] == raw["blue override"]
    assert args.episode_limit == 10 and args.exit_after_run
    assert source.read_bytes() == original
    assert result["source_config_sha256"] == hashlib.sha256(original).hexdigest()


def test_existing_pycharm_three_game_arguments_use_current_generated_settings(project):
    root, _, args, *_ = project
    args.config = str(root / "TaskConfig/cssim/iql_eval_red.jsonc")
    args.episode_limit = 3
    result = resolve(project, ["2", "10"])
    assert result["policy_baseline"] == "6cb5a79"
    assert args.episode_limit == 10
    assert json.loads(Path(args.config).read_bytes())["Common.cssim_env_wrapper.py->ScenarioConfig"]["CSSIMPort"] == 29999


def test_platform_choice_keeps_original_config_bytes_and_policy(project):
    _, source, args, *_ = project
    original = source.read_bytes()
    result = resolve(project, ["3", "2"])
    assert args.config == str(source)
    assert result["strategy"] == "platform" and args.episode_limit == 2
    assert source.read_bytes() == original


@pytest.mark.parametrize("flag", ["batch_id", "source_commit", "s"])
def test_managed_launch_never_prompts_or_overrides_bound_config(project, flag):
    root, source, args, *_ = project
    args.config = str(source)
    setattr(args, flag, True if flag == "s" else "bound-value")
    assert manual.resolve_manual_launch(args, project_root=root,
        input_fn=lambda _: pytest.fail("managed prompt")) is None
    args.strategy = "historical"
    with pytest.raises(ValueError, match="已绑定"):
        manual.resolve_manual_launch(args, project_root=root)


@pytest.mark.parametrize("mode", ["custom", "non_interactive"])
def test_existing_script_config_can_opt_out_of_manual_choice(project, mode):
    root, source, args, *_ = project
    args.config = str(source.with_name("custom.jsonc") if mode == "custom" else source)
    args.non_interactive = mode == "non_interactive"
    assert manual.resolve_manual_launch(args, project_root=root,
        input_fn=lambda _: pytest.fail("unexpected prompt")) is None


@pytest.mark.parametrize("tamper", ["model", "snapshot", "route"])
def test_highscore_identity_mismatch_prevents_launch(project, tamper):
    _, _, args, _, folder, model = project
    file = model if tamper == "model" else folder / ("snapshot.json" if tamper == "snapshot" else "route.py")
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="不匹配"):
        resolve(project, ["1", "1"])
    assert not Path(args.log_root).exists()


def test_episode_choice_cannot_exceed_platform_budget(project):
    _, source, args, raw, *_ = project
    raw["config.py->GlobalConfig"]["max_episode"] = 3
    source.write_text(json.dumps(raw), encoding="utf-8")
    resolve(project, ["1", "10", "3"])
    assert args.episode_limit == 3


def test_changed_platform_settings_while_choosing_are_not_silently_used(project):
    root, source, args, *_ = project
    answers = iter(["1", "1"])
    def changing_input(_):
        source.write_bytes(source.read_bytes() + b"\n")
        return next(answers)
    with pytest.raises(RuntimeError, match="发生变化"):
        manual.resolve_manual_launch(args, project_root=root, input_fn=changing_input)


def test_console_eof_and_cancel_never_choose_a_strategy(project):
    root, _, args, *_ = project
    def no_console(_):
        raise EOFError
    with pytest.raises(manual.StartupCancelled):
        manual.resolve_manual_launch(args, project_root=root, input_fn=no_console)
    with pytest.raises(manual.StartupCancelled):
        resolve(project, ["q"])


def test_runtime_lease_rejects_duplicate_main_and_releases_after_error(tmp_path):
    lock = tmp_path / "main-runtime.lock"
    with manual.RuntimeLease(lock):
        with pytest.raises(RuntimeError, match="已有main"):
            with manual.RuntimeLease(lock):
                pytest.fail("duplicate main entered")
    with pytest.raises(ValueError):
        with manual.RuntimeLease(lock):
            raise ValueError("failed run")
    with manual.RuntimeLease(lock):
        pass


def test_main_without_arguments_is_a_supported_manual_entry(monkeypatch):
    monkeypatch.delenv("CSSIM_SOURCE_COMMIT", raising=False)
    monkeypatch.delenv("CSSIM_BATCH_ID", raising=False)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    args = CssimApplication.parse_arguments()
    assert args.config is None and args.strategy is None


@pytest.mark.parametrize("choice,baseline", [("1", "6cb5a79_hq_guard_entry"), ("2", "6cb5a79")])
def test_manual_menu_loads_the_selected_real_policy_without_connecting_to_ue(tmp_path, choice, baseline):
    from dataclasses import replace
    from test_champion_baseline import context
    from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RLAgentAlgorithm import ReinforceAgentAlgorithm
    from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.ChampionGuardHold import HQGuardEntryRoute
    root = Path(__file__).resolve().parents[1]
    args = argparse.Namespace(config=None, strategy=None, non_interactive=False,
        batch_id=None, source_commit=None, s=False, episode_limit=None,
        exit_after_run=False, log_root=str(tmp_path / "logs"))
    answers = iter([choice, "1"])
    manual.resolve_manual_launch(args, project_root=root,
        input_fn=lambda _: next(answers), output_fn=lambda _: None)
    effective = json.loads(Path(args.config).read_bytes())
    policy = ReinforceAgentAlgorithm(replace(context(), config=effective[manual.RED + ".py->AlgorithmConfig"]))
    metadata = policy.policy_metadata()
    assert metadata["policy_baseline"] == baseline
    assert metadata["model_sha256"] == manual.MODEL_SHA and metadata["shadow"] is False
    assert isinstance(policy._route, HQGuardEntryRoute) == (choice == "1")
