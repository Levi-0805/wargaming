from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import strftime
from typing import Any


class SimulationLog:
    """运行日志、逐帧统计和 DEBUG 原始协议的统一入口。"""

    _DEBUG_FIELDS = frozenset({
        "action_indices", "action_strings", "action_echo", "entities",
        "roster", "commanders", "key_objects",
    })

    def __init__(self, root: str | Path = "logs", detailed: bool = False):
        self.detailed = detailed
        self.directory = Path(root) / strftime("%Y%m%d_%H%M%S")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stats_path = self.directory / "simulation.jsonl"
        self._stats = self.stats_path.open("a", encoding="utf-8")
        self.protocol_path = self.directory / "protocol.jsonl"
        self._protocol = (
            self.protocol_path.open("a", encoding="utf-8") if detailed else None
        )

    @classmethod
    def configure(
        cls, root: str | Path = "logs", level: str | int = "DEBUG",
    ) -> "SimulationLog":
        """创建本轮日志对象并配置 Python 日志系统。"""

        if isinstance(level, str):
            normalized = level.upper()
            if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
                raise ValueError(f"不支持的日志级别: {level}")
            level = getattr(logging, normalized)
        run_log = cls(root, detailed=level <= logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        file_handler = RotatingFileHandler(
            run_log.directory / "runtime.log",
            maxBytes=20 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logging.basicConfig(level=level, handlers=[console, file_handler], force=True)
        logging.getLogger("grpc").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        return run_log

    def write_simulation_record(self, record: dict[str, Any]) -> None:
        """写入一条reset或step记录；INFO级别自动移除大体积诊断字段。"""

        if not self.detailed:
            record = {
                key: value for key, value in record.items()
                if key not in self._DEBUG_FIELDS
            }
        self._stats.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._stats.flush()

    def write_protocol(
        self, direction: str, channel: str, type_name: str, payload: str
    ) -> None:
        """DEBUG 时逐条保存未经字段裁剪的 gRPC JSON，供协议语义取证。"""

        if self._protocol is None:
            return
        try:
            data: Any = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            data = payload
        self._protocol.write(json.dumps({
            "direction": direction,
            "channel": channel,
            "type": type_name,
            "payload": data,
        }, ensure_ascii=False, default=str) + "\n")
        self._protocol.flush()

    def close(self) -> None:
        """刷新并关闭统计日志和可选的原始协议日志。"""
        self._stats.close()
        if self._protocol is not None:
            self._protocol.close()
