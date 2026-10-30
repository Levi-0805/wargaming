# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import random


# 限制数值库内部线程；仿真调度和 gRPC 均由主事件循环串行执行。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


class CssimApplication:
    """Python 端命令行入口及运行生命周期。"""

    @staticmethod
    def parse_arguments():
        """解析UE固定启动参数以及可选日志级别。"""

        parser = argparse.ArgumentParser(description="CSSIM 单线程 Python 客户端")
        parser.add_argument("-c", "--config", required=True, help="UE 生成的 JSONC 配置")
        parser.add_argument("-s", action="store_true", help="兼容 UE 启动参数，不启用守护进程")
        parser.add_argument("--log-root", default="logs", help="日志根目录；每次启动创建独立会话子目录")
        parser.add_argument("--source-commit", default=os.environ.get("CSSIM_SOURCE_COMMIT"),
                            help="启动器已核验的代码提交；不从目录或时间推断")
        parser.add_argument("--batch-id", default=os.environ.get("CSSIM_BATCH_ID"),
                            help="启动器提供的采集批次标识")
        parser.add_argument("--exit-after-run", action="store_true",
                            help="托管批次正常结束并刷新日志后退出，避免 gRPC 后台线程阻止进程关闭")
        parser.add_argument(
            "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
            help="日志级别；DEBUG 会记录完整的 gRPC JSON 收发报文",
        )
        return parser.parse_args()

    @staticmethod
    def seed_random_generators(seed: int) -> None:
        """同步Python、NumPy和PyTorch随机种子，便于复现实验。"""

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    @classmethod
    def run(cls) -> None:
        """装载配置和日志，并在唯一事件循环中运行仿真。"""

        from cssim.config import RuntimeConfig
        from cssim.logging_setup import SimulationLog
        from cssim.runtime import Runner

        args = cls.parse_arguments()
        config_path = Path(args.config).resolve()
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        config = RuntimeConfig.load(args.config)
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != config_sha256:
            raise RuntimeError("配置在读取期间发生变化，请重新启动")
        cls.seed_random_generators(config.seed)
        run_log = SimulationLog.configure(args.log_root, args.log_level or config.log_level)
        status, error_type = "failed", None
        try:
            run_log.write_run_metadata({
                "schema_version": 1,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "process_id": os.getpid(),
                "batch_id": args.batch_id,
                "source_commit": args.source_commit,
                "source_commit_evidence": "provided_by_launcher" if args.source_commit else "unknown",
                "config_path": str(config_path),
                "config_sha256": config_sha256,
                "algorithm_names": list(config.algorithm_names),
                "configured_max_episodes": config.max_episodes,
                "configured_max_steps": config.max_steps,
                "detailed_protocol": run_log.detailed,
            })
            asyncio.run(Runner(config, run_log).run())
            status = "returned"
        except BaseException as exc:
            error_type = type(exc).__name__
            logging.getLogger(__name__).exception("CSSIM 运行失败")
            raise
        finally:
            try:
                # A normal return does not establish the requested number of
                # complete score packets; the batch controller verifies that.
                run_log.write_simulation_record({
                    "kind": "run_end", "status": status, "error_type": error_type,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                })
            finally:
                run_log.close()
        if args.exit_after_run:
            # Reached only after a successful Runner return and closed audit
            # files. gRPC's native polling thread can otherwise hang shutdown.
            logging.shutdown()
            os._exit(0)


if __name__ == "__main__":
    CssimApplication.run()
