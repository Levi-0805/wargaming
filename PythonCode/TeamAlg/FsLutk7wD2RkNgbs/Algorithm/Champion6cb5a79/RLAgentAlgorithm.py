from __future__ import annotations

from dataclasses import dataclass

from cssim.algorithms import AlgorithmContext


@dataclass(frozen=True)
class AlgorithmConfig:
    """UE固定读取的红方配置；规则、DQN及离线 IQL 目标选择。"""

    mode: str = "rule"
    hidden_size: int = 256
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 128
    replay_size: int = 10_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 20_000
    target_update: int = 500
    load_model: bool = False
    enable_parent_assignment: bool = True
    iql_model: str = 'Models/target_iql_v1.json'
    iql_shadow: bool = True
    # Retired ground interventions. Enable one at a time only for comparisons.
    enable_preservation_pause: bool = False
    enable_shared_contact_maneuver: bool = False
    enable_forced_entry_under_contact: bool = False
    enable_continuous_column: bool = False
    enable_entry_waves: bool = False

    def __post_init__(self):
        for name in ('iql_shadow', 'enable_preservation_pause', 'enable_shared_contact_maneuver',
                     'enable_forced_entry_under_contact', 'enable_continuous_column',
                     'enable_entry_waves'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a JSON boolean")

    @classmethod
    def from_context(cls, context: AlgorithmContext) -> "AlgorithmConfig":
        """把当前队伍的TaskConfig覆盖项解析为独立配置对象。"""
        try:
            return cls(**context.config)
        except TypeError as exc:
            raise ValueError(f"算法配置包含未知参数: {context.config}") from exc


class ReinforceAgentAlgorithm:
    """UE固定入口；选择规则、DQN 示例或训练过的 IQL 目标子策略。"""

    def __new__(cls, context: AlgorithmContext):
        config = AlgorithmConfig.from_context(context)
        if config.mode == 'hybrid_iql':
            from .LearnedTargeting import HybridIQLAgent
            return HybridIQLAgent(context, config)
        if config.mode == "rule":
            from .RuleAgentAlgorithm import ReinforceAgentAlgorithm as RuleAgent
            return RuleAgent(context, config.enable_parent_assignment,
                             enable_preservation_pause=config.enable_preservation_pause,
                             enable_shared_contact_maneuver=config.enable_shared_contact_maneuver,
                             enable_forced_entry_under_contact=config.enable_forced_entry_under_contact,
                             enable_continuous_column=config.enable_continuous_column,
                             enable_entry_waves=config.enable_entry_waves)
        if config.mode == "reinforcement_learning":
            from .DQN import DQNAgentAlgorithm

            return DQNAgentAlgorithm(context, config)
        raise ValueError(f"未知算法模式: {config.mode}")
