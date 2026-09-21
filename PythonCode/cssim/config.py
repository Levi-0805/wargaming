from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import commentjson

@dataclass(frozen=True)
class RuntimeConfig:
    """从 UE 生成的 TaskConfig 提取出的 Python 运行参数。"""

    path: Path
    raw: dict[str, Any]
    host: str
    port: int
    algorithm_names: tuple[str, str]
    max_steps: int
    max_episodes: int
    time_dilation: float
    frame_rate: float
    step_game_time: float
    seed: int
    device: str
    log_level: str
    connect_timeout: float = 60.0
    reply_timeout: float = 60.0

    @staticmethod
    def _section(data: dict[str, Any], class_name: str) -> dict[str, Any]:
        return next(
            (
                value for key, value in data.items()
                if key.split("->")[-1] == class_name and isinstance(value, dict)
            ),
            {},
        )

    @classmethod
    def load(cls, path: str | Path) -> "RuntimeConfig":
        """读取 JSONC，并校验重构版必须具备的双队伍、单环境约束。"""

        source = Path(path).resolve()
        raw = commentjson.loads(source.read_text(encoding="utf-8-sig"))
        global_cfg = cls._section(raw, "GlobalConfig")
        scenario = cls._section(raw, "ScenarioConfig")
        names = tuple(scenario.get("Algorithm_Names", ()))
        if len(names) != 2:
            raise ValueError("ScenarioConfig.Algorithm_Names 必须包含红、蓝两个算法")
        if int(global_cfg.get("num_worker_threads", 1)) != 1:
            raise ValueError("重构版只支持单环境，请将 num_worker_threads 设为 1")
        return cls(
            path=source,
            raw=raw,
            host=str(scenario.get("GrpcAddr", scenario.get("TcpAddr", "127.0.0.1"))),
            port=int(scenario.get("CSSIMPort", 21051)),
            algorithm_names=(str(names[0]), str(names[1])),
            max_steps=int(scenario.get("MaxStepsPerEpisode", 200)),
            max_episodes=int(global_cfg.get("max_episode", 1)),
            time_dilation=float(scenario.get("TimeDilation", 1.0)),
            frame_rate=float(scenario.get("FrameRate", 60.0)),
            step_game_time=float(scenario.get("StepGameTime", 0.5)),
            seed=int(global_cfg.get("seed", 0)),
            device=str(global_cfg.get("device", "cpu")),
            log_level=str(os.environ.get(
                "CSSIM_LOG_LEVEL", global_cfg.get("log_level", "DEBUG")
            )).upper(),
            connect_timeout=float(scenario.get("GrpcConnectTimeout", 60.0)),
            reply_timeout=float(scenario.get("GrpcReplyTimeout", 60.0)),
        )
