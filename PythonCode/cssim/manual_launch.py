"""Explicit manual policy selection; managed batches keep their bound config."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from .config import RuntimeConfig

TEAM = "FsLutk7wD2RkNgbs"
RED = f"TeamAlg.{TEAM}.Algorithm.RLAgentAlgorithm"
MODEL_SHA = "3e36fb18875258f58065cfa748b57c8947675cded629654ce3b79696a8863f4f"
SNAPSHOT_SHA = "7d7e619118e5c7c22235deca80d49d0ab1d959d8172f5e8fd62f03f72ecdcac5"
STRATEGIES = {"hq-guard": "6cb5a79_hq_guard_entry", "historical": "6cb5a79"}


class StartupCancelled(RuntimeError):
    pass


class RuntimeLease:
    """One main.py per project; OS locks are released even after a crash."""
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        try:
            if self.path.stat().st_size == 0:
                self.stream.write(b" ")
                self.stream.flush()
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise RuntimeError("已有main正在运行，请等待当前对局结束，不要重复启动。") from exc
        return self

    def __exit__(self, *_):
        self.stream.close()


def _ask(prompt, input_fn):
    try:
        value = input_fn(prompt).strip()
    except EOFError as exc:
        raise StartupCancelled("控制台无法读取选择；请使用PyCharm运行控制台，或传入--strategy。") from exc
    if value.lower() in {"q", "quit", "取消"}:
        raise StartupCancelled("已取消启动。")
    return value


def prepare_highscore(source, strategy, output, project_root):
    """Keep live platform connection/settings; never edit its generated input."""
    config = RuntimeConfig.load(source)
    if config.algorithm_names[0].replace(".py->", "->") != RED + "->ReinforceAgentAlgorithm":
        raise ValueError("平台红方未选择FsLutk7wD2RkNgbs算法，请先在CSSIM中选择该算法。")
    folder = Path(project_root) / f"TeamAlg/{TEAM}/Algorithm/Champion6cb5a79"
    snapshot = (folder / "snapshot.json").read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != SNAPSHOT_SHA:
        raise ValueError("历史高分战术清单不匹配，停止启动。")
    for name, expected in json.loads(snapshot)["files"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"历史高分战术文件不匹配：{name}")
    model = Path(project_root) / f"TeamAlg/{TEAM}/Models/target_iql_v1.json"
    if hashlib.sha256(model.read_bytes()).hexdigest() != MODEL_SHA:
        raise ValueError("原高分模型不匹配，停止启动。")
    raw = config.raw
    keys = [key for key in raw if key.replace(".py->", "->") == RED + "->AlgorithmConfig"]
    if len(keys) > 1:
        raise ValueError("红方策略配置重复，停止启动。")
    key = keys[0] if keys else RED + ".py->AlgorithmConfig"
    raw[key] = dict(mode="hybrid_iql", policy_baseline=STRATEGIES[strategy],
                    iql_shadow=False, iql_model=str(model.resolve()))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def resolve_manual_launch(args, *, project_root, input_fn=None, output_fn=print):
    """Prompt for normal main/PyCharm launches; never prompt inside a batch."""
    managed = bool(args.batch_id or args.source_commit or args.s)
    if managed:
        if args.strategy:
            raise ValueError("托管批次必须使用已绑定配置，不能在main中另选策略。")
        if not args.config:
            raise ValueError("托管启动必须提供-c配置路径。")
        return None
    known = args.config is None or Path(args.config).name in {"cssim_25vs25.jsonc", "iql_eval_red.jsonc"}
    interactive = not args.non_interactive and known and args.strategy is None
    if not interactive and args.strategy is None:
        if not args.config:
            raise ValueError("请提供-c配置，或直接运行main选择策略。")
        return None
    root = Path(project_root).resolve()
    # The old PyCharm main points to a static three-game evaluation template.
    # A manual policy choice must use the current UE-generated port and budget.
    source = root / "TaskConfig/cssim/cssim_25vs25.jsonc" if known else Path(args.config).resolve()
    if not source.is_file():
        raise ValueError("未找到平台运行配置。请先在CSSIM选择场景、算法和局数，再运行main。")
    source_bytes = source.read_bytes()
    platform = RuntimeConfig.load(source)
    upper = min(10, platform.max_episodes)
    if upper < 1:
        raise ValueError("平台局数必须大于0。")
    strategy = args.strategy
    if interactive:
        input_fn = input_fn or input
        output_fn("\n选择本次战略（q取消）：\n"
                  "  1 高分基线＋HQ守备修复（默认；实战收益待验证）\n"
                  "  2 原始高分基线（历史对照）\n"
                  "  3 平台原配置")
        while True:
            choice = _ask("请输入1/2/3 [1]：", input_fn) or "1"
            if choice in {"1", "2", "3"}:
                strategy = {"1": "hq-guard", "2": "historical", "3": "platform"}[choice]
                break
            output_fn("请输入1、2、3，或q取消。")
        default = min(args.episode_limit or upper, upper)
        while True:
            value = _ask(f"本次运行局数1—{upper} [{default}]：", input_fn)
            if not value:
                args.episode_limit = default
                break
            if value.isdecimal() and 1 <= int(value) <= upper:
                args.episode_limit = int(value)
                break
            output_fn(f"局数不能超过平台配置的{upper}局；请先在CSSIM修改局数。")
        args.exit_after_run = True
    if args.episode_limit is not None and args.episode_limit > upper:
        raise ValueError(f"本次局数超过平台配置的{upper}局。")
    if source.read_bytes() != source_bytes:
        raise RuntimeError("平台配置在选择期间发生变化，请重新运行main。")
    selected = source
    if strategy != "platform":
        selected = prepare_highscore(source, strategy,
            Path(args.log_root).resolve() / "manual-configs" / uuid4().hex / "effective.jsonc", root)
    if source.read_bytes() != source_bytes:
        raise RuntimeError("平台配置在准备期间发生变化，请重新运行main。")
    args.config = str(selected)
    result = dict(strategy=strategy, source_config=str(source),
                  source_config_sha256=hashlib.sha256(source_bytes).hexdigest(),
                  effective_config=str(selected), policy_baseline=STRATEGIES.get(strategy),
                  model_sha256=MODEL_SHA if strategy != "platform" else None)
    output_fn(f"已选择：{strategy}；平台连接 {platform.host}:{platform.port}；"
              f"最多运行{args.episode_limit or upper}局。记录将写入独立日志。")
    return result
