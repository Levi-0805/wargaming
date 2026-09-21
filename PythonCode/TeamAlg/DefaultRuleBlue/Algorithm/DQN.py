"""可直接运行的蓝方DQN案例。

建议按以下顺序阅读：DQNObservation定义25维必要态势，ReplayExperience定义训练样本，
DQNActionSet定义8个离散动作，_QNetwork定义网络，DQNAlgorithm实现训练流程，
DQNAgentAlgorithm定义蓝方任务奖励。
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from cssim.algorithms import (
    ActionSetDefinition,
    AlgorithmContext,
    LearningTransition,
    ReinforcementAlgorithm,
)
from cssim.environment import TeamState
from cssim.protocol import (
    Action,
    ChangeParentAction,
    DiscreteAction,
)


logger = logging.getLogger(__name__)


# DQN示例只保留完成当前任务所需的固定宽度信息，字段使用UE原始工程量。
# 不按UE蓝图类名枚举兵种；新增兵种直接用自身通用能力字段进入同一个网络。
_SELF_DIM = 10       # 存活、生命值、速度xyz、最大速度、冷却、攻击/警戒/感知范围
_OBJECTIVE_DIM = 5  # 是否有目标、相对xyz、相对距离
_ENEMY_DIM = 6      # 是否有敌人、生命值、相对xyz、相对距离
_PARENT_DIM = 3     # 是否有候选上级、是否需要切换、当前通信状态
_GLOBAL_DIM = 1     # 当前步数
_DQN_INPUT_DIM = _SELF_DIM + _OBJECTIVE_DIM + _ENEMY_DIM + _PARENT_DIM + _GLOBAL_DIM
_OBSERVATION_VERSION = 1


class MultiObjectivePlan:
    """用Python的team_index动作行号稳定分配目标，确保多目标均有守军。"""

    defending_team = 1

    @classmethod
    def target(cls, state: TeamState, agent):
        objectives = tuple(sorted(
            state.key_objects_for_team(cls.defending_team), key=lambda item: item.uid
        ))
        return objectives[agent.team_index % len(objectives)] if objectives else None


@dataclass(frozen=True)
class DQNObservation:
    """每个Agent一行的精简固定态势，不包含可变长实体集合。"""

    self_values: np.ndarray
    objective: np.ndarray
    enemy: np.ndarray
    parent: np.ndarray
    global_values: np.ndarray

    @staticmethod
    def _number(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _nearest_enemy(state: TeamState, agent):
        return min(
            state.perceived_opponents(agent),
            key=lambda item: float(np.linalg.norm(agent.position - item.position)),
            default=None,
        )

    @classmethod
    def from_state(cls, state: TeamState) -> "DQNObservation":
        """提取动作决策真正使用的自身、目标、敌人和上级信息。"""

        agent_count = len(state.agents)
        self_values = np.zeros((agent_count, _SELF_DIM), dtype=np.float32)
        objectives = np.zeros((agent_count, _OBJECTIVE_DIM), dtype=np.float32)
        enemies = np.zeros((agent_count, _ENEMY_DIM), dtype=np.float32)
        parents = np.zeros((agent_count, _PARENT_DIM), dtype=np.float32)
        global_values = np.zeros((agent_count, _GLOBAL_DIM), dtype=np.float32)

        for row, agent in enumerate(state.agents):
            properties = agent.properties
            raw = agent.raw
            self_values[row] = np.asarray([
                float(agent.alive),
                agent.hp,
                *agent.velocity,
                agent.max_speed,
                cls._number(raw.get("weaponCD", properties.get("weaponCD"))),
                cls._number(properties.get("fireRange")),
                cls._number(properties.get("guardRange")),
                cls._number(properties.get("perceptionRange")),
            ], dtype=np.float32)

            target = MultiObjectivePlan.target(state, agent)
            if target is not None:
                relative = target.position - agent.position
                objectives[row] = np.asarray([
                    1.0, *relative, float(np.linalg.norm(relative)),
                ], dtype=np.float32)

            enemy = cls._nearest_enemy(state, agent)
            if enemy is not None:
                relative = enemy.position - agent.position
                enemies[row] = np.asarray([
                    1.0,
                    enemy.hp,
                    *relative,
                    float(np.linalg.norm(relative)),
                ], dtype=np.float32)

            candidate = state.nearest_available_parent(agent)
            can_change = (
                agent.alive
                and agent.entity_type != "BP_BaseSoldier_C"
                and candidate is not None
            )
            parents[row] = np.asarray([
                float(can_change),
                float(can_change and agent.commander_uid() != candidate.uid),
                float(agent.communication_ok() is True),
            ], dtype=np.float32)

            global_values[row, 0] = state.step

        return cls(self_values, objectives, enemies, parents, global_values)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "self": self.self_values,
            "objective": self.objective,
            "enemy": self.enemy,
            "parent": self.parent,
            "global": self.global_values,
        }

    def agent_row(self, index: int) -> dict[str, np.ndarray]:
        """复制一个Agent的固定态势，供经验回放安全保存。"""

        return {
            name: values[index].copy()
            for name, values in self.as_arrays().items()
        }


@dataclass(frozen=True)
class ReplayExperience:
    """DQN的一条单Agent训练经验。"""

    observation: dict[str, np.ndarray]
    action: int
    reward: float
    next_observation: dict[str, np.ndarray]
    done: bool
    next_action_mask: np.ndarray


class _QNetwork(nn.Module):
    """把每个Agent的必要固定态势直接编码为八个动作评分。"""

    def __init__(self, hidden_size: int, action_count: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(_DQN_INPUT_DIM, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        values = torch.cat((
            features["self"],
            features["objective"],
            features["enemy"],
            features["parent"],
            features["global"],
        ), dim=-1)
        return self.model(values)


class DQNAlgorithm(ReinforcementAlgorithm):
    """共享参数 DQN；每个Agent使用固定宽度的精简任务态势。"""

    def __init__(
        self,
        context: AlgorithmContext,
        *,
        hidden_size: int = 256,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 128,
        replay_size: int = 10_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 20_000,
        target_update: int = 500,
        model_path: str | Path | None = None,
        load_model: bool = False,
    ):
        super().__init__(context)
        selected = self.context.action_set
        requested = context.device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，DQN 自动使用 CPU")
            requested = "cpu"
        self.device = torch.device(requested)
        self.online = _QNetwork(hidden_size, selected.n).to(self.device)
        self.target = _QNetwork(hidden_size, selected.n).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.batch_size = max(int(batch_size), 1)
        self.replay = deque(maxlen=max(int(replay_size), 1))
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = max(epsilon_decay, 1)
        self.target_update = max(target_update, 1)
        self.decision_count = 0
        self.update_count = 0
        self.last_loss: float | None = None
        self.training_enabled = False
        self.model_path = Path(model_path) if model_path else None
        if load_model:
            if self.model_path is None:
                raise ValueError("load_model=True时必须提供model_path")
            if not self.model_path.exists():
                raise FileNotFoundError(f"找不到DQN模型: {self.model_path}")
            self.load_checkpoint()

    def _convert_arrays_to_tensors(
        self, features: dict[str, np.ndarray]
    ) -> dict[str, torch.Tensor]:
        return {
            name: torch.as_tensor(
                value,
                dtype=torch.bool if name.endswith("_mask") else torch.float32,
                device=self.device,
            )
            for name, value in features.items()
        }

    def build_observation(self, state: TeamState) -> DQNObservation:
        """自定义态势入口；返回值会同时用于决策和训练。"""

        return DQNObservation.from_state(state)

    @staticmethod
    def _collate_replay_feature_rows(
        rows: list[dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """固定宽度态势可以直接堆叠，不需要集合补齐和额外mask。"""

        return {
            name: np.stack([row[name] for row in rows]).astype(np.float32)
            for name in ("self", "objective", "enemy", "parent", "global")
        }

    def select_actions(self, observation: DQNObservation, action_mask, training):
        """使用掩码后的Q值执行epsilon-greedy，并返回每个实体的动作编号。"""

        if not isinstance(observation, DQNObservation):
            raise TypeError("DQN案例必须使用DQNObservation作为模型输入")
        self.training_enabled = bool(training)
        self.online.train(self.training_enabled)
        with torch.no_grad():
            q_values = self.online(
                self._convert_arrays_to_tensors(observation.as_arrays())
            ).cpu().numpy()
        q_values[~action_mask] = -np.inf
        greedy = q_values.argmax(axis=1)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * max(
            0.0, 1.0 - self.decision_count / self.epsilon_decay
        )
        # 当前框架不按单位状态过滤动作；探索率每个环境决策帧只推进一次。
        active_rows = np.flatnonzero(action_mask.sum(axis=1) > 1)
        if training:
            for row in active_rows:
                if random.random() < epsilon:
                    greedy[row] = random.choice(np.flatnonzero(action_mask[row]).tolist())
            # 探索衰减按环境决策帧计数，不随一方实体数量变化。
            self.decision_count += int(bool(len(active_rows)))
        return greedy

    def train_on_transition(self, transition: LearningTransition) -> None:
        """训练帧到达后依次存储经验，并执行一次DQN训练步骤。"""

        self.training_enabled = bool(transition.state.training)
        if not self.training_enabled or transition.actions is None:
            return
        self.store_transition(transition)
        self.train_step()

    def store_transition(self, transition: LearningTransition) -> None:
        """把一条队伍级转移拆成存活Agent的单体经验并写入回放池。"""

        for index, action in enumerate(transition.actions):
            # 已经阵亡的实体不再产生无意义经验；本步阵亡则保留死亡转移并终止 bootstrap。
            if not transition.state.agents[index].alive:
                continue
            agent_done = transition.done or not transition.next_state.agents[index].alive
            self.replay.append(ReplayExperience(
                observation=transition.observation.agent_row(index),
                action=int(action),
                reward=float(transition.reward[index]),
                next_observation=transition.next_observation.agent_row(index),
                done=bool(agent_done),
                next_action_mask=transition.next_action_mask[index].copy(),
            ))

    @staticmethod
    def _pad_action_masks(masks: list[np.ndarray]) -> np.ndarray:
        width = max(len(item) for item in masks)
        result = np.zeros((len(masks), width), dtype=bool)
        for index, item in enumerate(masks):
            result[index, :len(item)] = item
        return result

    def train_step(self) -> None:
        """经验量充足时，从回放池采样并完成一次梯度更新。"""

        if len(self.replay) < self.batch_size:
            return
        batch = random.sample(self.replay, self.batch_size)
        state_t = self._convert_arrays_to_tensors(
            self._collate_replay_feature_rows([item.observation for item in batch])
        )
        next_t = self._convert_arrays_to_tensors(
            self._collate_replay_feature_rows([item.next_observation for item in batch])
        )
        action_t = torch.as_tensor(
            [item.action for item in batch], dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        reward_t = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        done_t = torch.as_tensor(
            [item.done for item in batch], dtype=torch.float32, device=self.device
        )
        mask_t = torch.as_tensor(
            self._pad_action_masks([item.next_action_mask for item in batch]),
            dtype=torch.bool,
            device=self.device,
        )

        predicted = self.online(state_t).gather(1, action_t).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_t).masked_fill(~mask_t, -torch.inf).max(dim=1).values
            expected = reward_t + self.gamma * (1.0 - done_t) * next_q
        loss = nn.functional.smooth_l1_loss(predicted, expected)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()
        self.last_loss = float(loss.detach().cpu())
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

    def training_metrics(self) -> dict[str, int | float]:
        """返回探索、训练和经验池指标，供运行日志记录。"""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * max(
            0.0, 1.0 - self.decision_count / self.epsilon_decay
        )
        metrics: dict[str, int | float] = {
            "decision_count": self.decision_count,
            "update_count": self.update_count,
            "replay_size": len(self.replay),
            "epsilon": float(epsilon),
            "training_enabled": int(self.training_enabled),
        }
        if self.last_loss is not None:
            metrics["loss"] = self.last_loss
        return metrics

    def save_checkpoint(self) -> None:
        """保存网络、优化器以及特征和动作集合兼容元数据。"""

        if self.model_path is None:
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            **self.context.action_set.checkpoint_metadata(),
            "observation_version": _OBSERVATION_VERSION,
            "observation_width": _DQN_INPUT_DIM,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "decision_count": self.decision_count,
            "update_count": self.update_count,
        }, self.model_path)
        logger.info("DQN 模型已保存: %s", self.model_path)

    def load_checkpoint(self) -> None:
        """仅加载与当前动作语义一致的检查点。"""

        checkpoint = torch.load(self.model_path, map_location=self.device)
        expected = self.context.action_set.checkpoint_metadata()
        stored_signature = checkpoint.get("action_signature")
        if stored_signature is None:
            raise ValueError("模型缺少动作集合元数据，不能加载旧版 checkpoint")
        if stored_signature != expected["action_signature"]:
            raise ValueError(
                "模型动作集合不兼容: "
                f"{checkpoint.get('action_set_name')} v{checkpoint.get('action_set_version')} "
                f"!= {expected['action_set_name']} v{expected['action_set_version']}"
            )
        if (
            checkpoint.get("observation_version") != _OBSERVATION_VERSION
            or checkpoint.get("observation_width") != _DQN_INPUT_DIM
        ):
            raise ValueError("模型态势结构与当前25维精简态势不兼容")
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["online"]))
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.decision_count = int(checkpoint.get("decision_count", 0))
        self.update_count = int(checkpoint.get("update_count", 0))
        logger.info("DQN 模型已加载: %s", self.model_path)

    def on_episode_end(self, episode: int) -> None:
        """每10回合保存一次中间检查点。"""
        if self.training_enabled and self.model_path and episode % 10 == 0:
            self.save_checkpoint()

    def on_simulation_end(self) -> None:
        """全部回合结束时保存本次训练的最终检查点。"""

        if self.training_enabled and self.model_path and self.update_count:
            self.save_checkpoint()


class DQNActionSet(ActionSetDefinition):
    """蓝方精简动作集合：移动、局部攻击和切换上级。"""

    defending_team = 1

    def __init__(self):
        # 攻击目标在当前帧动态解析，不再为每个敌人扩展动作槽。
        super().__init__("blue_minimal_control", version=1, include_target_attacks=False)

    def define(self, context: AlgorithmContext):
        """动作顺序就是DQN输出语义，修改顺序或含义后必须升级版本。"""

        alive = lambda _state, agent: agent.alive
        return (
            DiscreteAction.fixed("idle", Action.idle()),
            DiscreteAction.dynamic("move:+X", lambda _s, _a: Action.move("+X"), alive),
            DiscreteAction.dynamic("move:+Y", lambda _s, _a: Action.move("+Y"), alive),
            DiscreteAction.dynamic("move:-X", lambda _s, _a: Action.move("-X"), alive),
            DiscreteAction.dynamic("move:-Y", lambda _s, _a: Action.move("-Y"), alive),
            DiscreteAction.dynamic(
                "move:assigned_objective", self.move_to_objective, self.has_objective
            ),
            DiscreteAction.dynamic(
                "attack:nearest_local", self.attack_nearest_enemy, self.has_enemy
            ),
            DiscreteAction.dynamic(
                "special:change_parent:nearest_alive_parent",
                self.create_parent_action,
                self.has_parent,
            ),
        )

    @classmethod
    def has_objective(cls, state: TeamState, agent) -> bool:
        return agent.alive and MultiObjectivePlan.target(state, agent) is not None

    @classmethod
    def move_to_objective(cls, state: TeamState, agent) -> Action:
        return Action.move_to(MultiObjectivePlan.target(state, agent))

    @staticmethod
    def nearest_enemy(state: TeamState, agent):
        """只从当前Agent的局部感知中选择最近敌方。"""

        return min(
            state.perceived_opponents(agent),
            key=lambda item: float(np.linalg.norm(agent.position - item.position)),
            default=None,
        )

    @classmethod
    def has_enemy(cls, state: TeamState, agent) -> bool:
        return agent.alive and cls.nearest_enemy(state, agent) is not None

    @classmethod
    def attack_nearest_enemy(cls, state: TeamState, agent) -> Action:
        return Action.attack(cls.nearest_enemy(state, agent))

    @staticmethod
    def select_parent(state: TeamState, child):
        """为非士兵无人装备选择最近的存活士兵或初始Commander。"""

        if child.entity_type == "BP_BaseSoldier_C":
            return None
        return state.nearest_available_parent(child)

    @classmethod
    def has_parent(cls, state: TeamState, child) -> bool:
        return child.alive and cls.select_parent(state, child) is not None

    @classmethod
    def create_parent_action(cls, state: TeamState, child) -> ChangeParentAction:
        return ChangeParentAction(cls.select_parent(state, child), child)


class DQNAgentAlgorithm(DQNAlgorithm):
    """完整蓝方 DQN 案例：返回并守住己方指挥所、打击可见敌方。"""

    defending_team = 1
    defense_radius = 1800.0
    action_definition = DQNActionSet()

    def __init__(self, context: AlgorithmContext, config):
        team_root = Path(__file__).resolve().parents[1]
        super().__init__(
            context,
            hidden_size=config.hidden_size,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            batch_size=config.batch_size,
            replay_size=config.replay_size,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay=config.epsilon_decay,
            target_update=config.target_update,
            model_path=team_root / "TrainLogs" / "dqn_model.pt",
            load_model=config.load_model,
        )

    def calculate_reward(
        self, previous: TeamState, current: TeamState, events: Sequence[dict[str, str]]
    ) -> np.ndarray:
        reward = np.zeros(len(current.agents), dtype=np.float32)
        for index, (old_agent, new_agent) in enumerate(
            zip(previous.agents, current.agents)
        ):
            old_target = MultiObjectivePlan.target(previous, old_agent)
            new_target = MultiObjectivePlan.target(current, new_agent)
            # 目标变更造成的距离跳变不是本帧动作效果。
            if (
                not new_agent.alive
                or old_target is None
                or new_target is None
                or old_target.uid != new_target.uid
            ):
                continue
            old_distance = float(np.linalg.norm(
                old_agent.position[:2] - old_target.position[:2]
            ))
            new_distance = float(np.linalg.norm(
                new_agent.position[:2] - new_target.position[:2]
            ))
            if new_distance > self.defense_radius:
                reward[index] += float(np.clip(
                    (old_distance - new_distance) / 10_000.0, -0.02, 0.02
                ))
            else:
                reward[index] += 0.003

        old_alive = np.asarray([agent.alive for agent in previous.agents])
        new_alive = np.asarray([agent.alive for agent in current.agents])
        reward[old_alive & ~new_alive] -= 0.10
        uid_to_index = {agent.uid: index for index, agent in enumerate(current.agents)}
        for event in events:
            try:
                causer = int(event.get("DamageCauser", -1))
            except (TypeError, ValueError):
                continue
            if event.get("Event") == "Destroyed" and causer in uid_to_index:
                reward[uid_to_index[causer]] += 0.10
        return np.nan_to_num(reward)
