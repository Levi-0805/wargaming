from __future__ import annotations

from dataclasses import dataclass

from cssim.algorithms import AlgorithmContext


@dataclass(frozen=True)
class AlgorithmConfig:
    """No.1 的运行参数；入口名称由 CSSIM 固定读取。"""

    mode: str = "rule"
    guard_radius: float = 900.0
    enable_parent_assignment: bool = False

    @classmethod
    def from_context(cls, context: AlgorithmContext) -> "AlgorithmConfig":
        try:
            return cls(**context.config)
        except TypeError as exc:
            raise ValueError(f"算法配置包含未知参数: {context.config}") from exc


class ReinforceAgentAlgorithm:
    """CSSIM 固定入口；No.1 当前提供稳定的规则模式。"""

    def __new__(cls, context: AlgorithmContext):
        config = AlgorithmConfig.from_context(context)
        if config.mode != "rule":
            raise ValueError("No.1 目前只支持 mode=rule")
        from .RuleAgentAlgorithm import ReinforceAgentAlgorithm as RuleAgent

        return RuleAgent(
            context,
            guard_radius=config.guard_radius,
            enable_parent_assignment=config.enable_parent_assignment,
        )
