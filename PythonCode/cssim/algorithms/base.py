from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from cssim.environment.models import Entity, TeamState
from cssim.protocol import (
    Action,
    ChangeParentAction,
    DiscreteAction,
    RLActionSet,
)


@dataclass(frozen=True)
class AlgorithmContext:
    """算法初始化及回合刷新时由框架提供的队伍上下文。"""

    team: int  # 当前算法阵营：0=红方，1=蓝方；不是实体行号。
    agents: tuple[Entity, ...]
    # 算法创建或回合刷新时的队伍级可见敌方快照。
    visible_opponents: tuple[Entity, ...]
    action_set: RLActionSet
    device: str
    config: dict[str, Any]
    commanders: tuple[Entity, ...] = ()


@dataclass(frozen=True)
class ActionSetDefinition(ABC):
    """选手自定义强化学习动作集合时需要实现的最小接口。"""

    name: str
    version: int = 1
    include_target_attacks: bool = True

    @abstractmethod
    def define(self, context: AlgorithmContext) -> Sequence[DiscreteAction]:
        """按固定顺序返回离散动作；返回顺序就是模型动作编号。"""

        raise NotImplementedError

    def build(self, context: AlgorithmContext) -> RLActionSet:
        """校验选手定义并构造框架运行时动作集合。"""

        choices = tuple(self.define(context))
        if not all(isinstance(item, DiscreteAction) for item in choices):
            raise TypeError("ActionSetDefinition.define 必须只返回 DiscreteAction")
        return RLActionSet(
            name=self.name,
            version=int(self.version),
            choices=choices,
            include_target_attacks=bool(self.include_target_attacks),
            target_count=context.action_set.target_count,
        )


@dataclass(frozen=True)
class EnvironmentTransition:
    """运行器与算法基类之间使用的内部环境转移。"""

    state: TeamState
    actions: np.ndarray | None
    reward: np.ndarray
    next_state: TeamState
    done: bool


@dataclass(frozen=True)
class LearningTransition:
    """已经过选手态势接口转换的强化学习训练数据。"""

    state: TeamState
    observation: Any
    actions: np.ndarray | None
    reward: np.ndarray
    next_state: TeamState
    next_observation: Any
    next_action_mask: np.ndarray
    done: bool


class Algorithm(ABC):
    """规则和强化学习算法的共同生命周期。"""

    mode = "base"

    def __init__(self, context: AlgorithmContext):
        self.context = context
        self.last_action_indices: np.ndarray | None = None
        self._selected_special_commands: tuple[ChangeParentAction, ...] = ()

    @abstractmethod
    def act(self, state: TeamState) -> Sequence[Action]:
        """为本队每个战斗实体生成一条动作。"""

        raise NotImplementedError

    def calculate_reward(
        self, previous: TeamState, current: TeamState, events: Sequence[dict[str, str]]
    ) -> np.ndarray:
        """返回与本方实体行一一对应的奖励；默认奖励全部为零。"""

        return np.zeros(len(current.agents), dtype=np.float32)

    def _process_transition(self, transition: EnvironmentTransition) -> None:
        """框架内部转移入口；规则算法默认不处理训练数据。"""

    def take_selected_special_commands(self) -> tuple[ChangeParentAction, ...]:
        """取出本帧作为唯一动作选中的特殊指令。"""

        commands = self._selected_special_commands
        self._selected_special_commands = ()
        return commands

    def on_episode_end(self, episode: int) -> None:
        """回合结束钩子，默认不执行保存或训练。"""

    def on_simulation_end(self) -> None:
        """全部回合结束后的算法钩子，可用于最终保存或释放算法资源。"""

    def training_metrics(self) -> dict[str, int | float]:
        """返回可写入仿真日志的轻量训练指标。"""

        return {}


class RuleAlgorithm(Algorithm):
    """规则模式选手只需要实现 ``decide``。"""

    mode = "rule"

    @abstractmethod
    def decide(self, state: TeamState) -> Sequence[Action | ChangeParentAction]:
        """按 ``state.agents`` 顺序为每个实体返回唯一动作。"""

        raise NotImplementedError

    def act(self, state: TeamState) -> Sequence[Action]:
        """调用选手规则并校验每个实体都返回一个动作结果。"""
        choices = list(self.decide(state))
        if len(choices) != len(state.agents):
            raise ValueError(
                f"规则算法必须为本队每个实体返回一个动作，共{len(state.agents)}个"
            )
        actions = []
        special_commands = []
        for agent, choice in zip(state.agents, choices):
            if isinstance(choice, Action):
                actions.append(choice)
                continue
            if isinstance(choice, ChangeParentAction):
                if choice.child_uid != agent.uid:
                    raise ValueError(
                        f"特殊动作的child必须是当前动作行实体: "
                        f"agent={agent.uid} child={choice.child_uid}"
                    )
                actions.append(Action.idle())
                special_commands.append(choice)
                continue
            raise TypeError(
                f"规则动作必须是Action或ChangeParentAction: {type(choice).__name__}"
            )
        self._selected_special_commands = tuple(special_commands)
        return actions


class ReinforcementAlgorithm(Algorithm):
    """强化学习公共基类，不限定 DQN、QMIX、PPO 等具体算法。"""

    mode = "reinforcement_learning"
    action_definition: ActionSetDefinition | None = None

    def __init__(self, context: AlgorithmContext):
        definition = type(self).action_definition
        # 未指定自定义接口时沿用框架传入的标准动作集合。
        if definition is not None and not isinstance(definition, ActionSetDefinition):
            raise TypeError("action_definition 必须是 ActionSetDefinition 对象")
        action_set = context.action_set if definition is None else definition.build(context)
        super().__init__(replace(context, action_set=action_set))
        self._last_observation: Any = None

    def build_observation(self, state: TeamState) -> Any:
        """默认返回安全的队伍态势；选手负责构造自己的模型输入。"""

        return state

    @abstractmethod
    def select_actions(
        self, observation: Any, action_mask: np.ndarray, training: bool
    ) -> Sequence[int]:
        """根据自定义态势返回每个实体的离散动作编号。"""

        raise NotImplementedError

    def build_action_mask(self, state: TeamState) -> np.ndarray:
        """默认使用动作集合的可用性规则；选手可以在自己的算法中覆盖。"""

        return self.context.action_set.action_mask(state)

    def act(self, state: TeamState) -> Sequence[Action]:
        """校验模型输出，并把离散编号统一解码为普通动作和特殊指令。"""

        self._selected_special_commands = ()
        observation = self.build_observation(state)
        self._last_observation = observation
        action_mask = np.asarray(state.action_mask, dtype=bool)
        expected_mask_shape = (len(state.agents), self.context.action_set.n)
        if action_mask.shape != expected_mask_shape:
            raise ValueError(
                f"强化学习动作掩码应为 {expected_mask_shape}，实际为 {action_mask.shape}"
            )
        indices = np.asarray(
            self.select_actions(observation, action_mask, state.training),
            dtype=np.int64,
        )
        if indices.shape != (len(state.agents),):
            raise ValueError(f"强化学习动作形状应为 {(len(state.agents),)}，实际为 {indices.shape}")
        for row, index in enumerate(indices):
            if not 0 <= index < self.context.action_set.n:
                raise ValueError(f"agent={row} 动作索引越界: {index}")
            if not action_mask[row, index]:
                raise ValueError(
                    f"agent={row} 选择了当前不可用的动作: "
                    f"index={index} key={self.context.action_set.action_keys[index]}"
                )
        self.last_action_indices = indices.copy()
        actions = []
        special_commands = []
        for row, index in enumerate(indices):
            plan = self.context.action_set.decode(index, state, row)
            if plan.special_commands and any(
                command.child_uid != state.agents[row].uid
                for command in plan.special_commands
            ):
                raise ValueError(
                    f"特殊动作的child必须是当前动作行实体: agent={state.agents[row].uid}"
                )
            actions.append(plan.output)
            special_commands.extend(plan.special_commands)
        self._selected_special_commands = tuple(special_commands)
        return actions

    def train_on_transition(self, transition: LearningTransition) -> None:
        """训练扩展点；默认不保存经验也不更新模型。"""

    def _process_transition(self, transition: EnvironmentTransition) -> None:
        """把环境转移转换为选手定义的态势后，调用训练扩展点。"""

        observation = (
            self._last_observation
            if self._last_observation is not None
            else self.build_observation(transition.state)
        )
        learning_transition = LearningTransition(
            state=transition.state,
            observation=observation,
            actions=transition.actions,
            reward=transition.reward,
            next_state=transition.next_state,
            next_observation=self.build_observation(transition.next_state),
            next_action_mask=transition.next_state.action_mask.copy(),
            done=transition.done,
        )
        self._last_observation = None
        self.train_on_transition(learning_transition)
