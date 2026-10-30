from __future__ import annotations

import json
import logging
from itertools import count
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
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
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        stamp = strftime("%Y%m%d_%H%M%S")
        # Exclusive directory creation also separates simultaneous processes.
        # Reusing an existing second-resolution name would append two runs.
        for suffix in count():
            directory = root / (stamp if suffix == 0 else f"{stamp}-{suffix:03d}")
            try:
                directory.mkdir()
            except FileExistsError:
                continue
            self.directory = directory
            break
        self.stats_path = self.directory / "simulation.jsonl"
        self._stats = self.stats_path.open("x", encoding="utf-8")
        self.protocol_path = self.directory / "protocol.jsonl"
        self._protocol = (
            self.protocol_path.open("x", encoding="utf-8") if detailed else None
        )

    def write_run_metadata(self, record: dict[str, Any]) -> None:
        """Publish a complete provenance file atomically and never replace it."""

        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                    prefix=".run-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(record, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            # A hard link publishes the fully written same-volume file in one
            # operation. Unlike replace(), it rejects an existing run.json on
            # Windows and POSIX, preserving the original launch identity.
            os.link(temporary, self.directory / "run.json")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

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
