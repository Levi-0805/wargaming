# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import asyncio
import logging
import os
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
        config = RuntimeConfig.load(args.config)
        cls.seed_random_generators(config.seed)
        run_log = SimulationLog.configure("logs", args.log_level or config.log_level)
        try:
            asyncio.run(Runner(config, run_log).run())
        except BaseException:
            logging.getLogger(__name__).exception("CSSIM 运行失败")
            raise
        finally:
            run_log.close()


if __name__ == "__main__":
    CssimApplication.run()
