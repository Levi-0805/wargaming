from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace
from typing import Sequence

import numpy as np

from cssim.algorithms import AlgorithmContext
from cssim.algorithms.base import EnvironmentTransition
from cssim.config import RuntimeConfig
from cssim.environment import CssimEnvironment, SimulationState, TeamState
from cssim.logging_setup import SimulationLog
from cssim.protocol import ActionSetFactory

from .loader import AlgorithmLoader
from .special_commands import SpecialCommandExecutor


logger = logging.getLogger(__name__)


class Runner:
    """单环境、单事件循环的红蓝双方运行器。"""

    def __init__(self, config: RuntimeConfig, run_log: SimulationLog):
        self.config = config
        self.run_log = run_log
        self.environment = CssimEnvironment(config, run_log.write_protocol)
        self.special_command_executor = SpecialCommandExecutor(self.environment.client)
        self.algorithm_loader = AlgorithmLoader(config.raw)
        self.algorithms = []

    def _create_algorithms(self, state: SimulationState) -> None:
        for team, path in enumerate(self.config.algorithm_names):
            view = self.environment.build_team_state(state, team)
            context = AlgorithmContext(
                team=team,
                agents=view.agents,
                visible_opponents=view.visible_opponents,
                action_set=ActionSetFactory.standard(self._target_count(view)),
                device=self.config.device,
                config={},
                commanders=view.commanders,
            )
            self.algorithms.append(self.algorithm_loader.load(path, context))
            logger.info(
                "算法加载: team=%d mode=%s path=%s action_set=%s actions=%d",
                team,
                self.algorithms[-1].mode,
                path,
                self.algorithms[-1].context.action_set.name,
                self.algorithms[-1].context.action_set.n,
            )

    async def run(self) -> None:
        """串行执行连接、reset、双方决策、step、训练和多回合生命周期。"""

        try:
            await self.environment.connect()
            state = await self.environment.reset()
            self._create_algorithms(state)
            self._write_reset_log(state)
            target_episodes = self.environment.max_episodes
            while True:
                previous_views = self._prepare_team_views(state)
                step = state.step
                while True:
                    team_actions = [
                        list(algorithm.act(previous_views[team]))
                        for team, algorithm in enumerate(self.algorithms)
                    ]
                    special_results = []
                    for team, algorithm in enumerate(self.algorithms):
                        commands = algorithm.take_selected_special_commands()
                        results = await self.special_command_executor.execute(
                            team, state, commands
                        )
                        special_results.extend(item.to_dict() for item in results)
                    # team选择红/蓝动作表，team_index选择该队动作行；二者都不是UE UID。
                    combined = [
                        team_actions[entity.team][entity.team_index]
                        for entity in state.entities
                    ]
                    result = await self.environment.step(combined, step)
                    current_views = self._prepare_team_views(result.state)
                    rewards = []
                    for team, algorithm in enumerate(self.algorithms):
                        visible_events = self._filter_visible_events(
                            result.events, previous_views[team], current_views[team]
                        )
                        reward = np.asarray(
                            algorithm.calculate_reward(
                                previous_views[team], current_views[team], visible_events
                            ),
                            dtype=np.float32,
                        )
                        if reward.shape != (len(current_views[team].agents),):
                            raise ValueError(f"team={team} 奖励形状错误: {reward.shape}")
                        if result.done and result.winner in (0, 1):
                            reward += 1.0 if result.winner == team else -1.0
                        transition = EnvironmentTransition(
                            previous_views[team],
                            algorithm.last_action_indices,
                            reward,
                            current_views[team],
                            result.done,
                        )
                        # 具体算法自行决定是否以及怎样消费该队伍级转移。
                        algorithm._process_transition(transition)
                        rewards.append(reward)

                    action_counts = Counter(action.command for action in combined)
                    sent_by_uid: dict[int, str] = {}
                    for sent in result.action_strings:
                        try:
                            uid = int(sent.rsplit(";", 1)[-1])
                        except (TypeError, ValueError):
                            continue
                        sent_by_uid[uid] = sent
                    echo_pairs = [
                        (
                            entity.uid,
                            sent_by_uid.get(entity.uid, ""),
                            str(entity.raw.get("previousAction", "")),
                        )
                        for entity in result.state.entities
                        if entity.raw.get("previousAction") not in (None, "", -1)
                    ]
                    echo_mismatches = [
                        {"uid": uid, "sent": sent, "reported": reported}
                        for uid, sent, reported in echo_pairs
                        if reported != sent
                    ]
                    self.run_log.write_simulation_record(
                        {
                            "episode": state.episode,
                            "step": result.state.step,
                            "done": result.done,
                            "winner": result.winner,
                            "end_reason": result.end_reason,
                            "alive": {
                                str(team): sum(item.alive and item.team == team for item in result.state.entities)
                                for team in (0, 1)
                            },
                            "actions": dict(action_counts),
                            "action_indices": [
                                algorithm.last_action_indices.tolist()
                                if algorithm.last_action_indices is not None else None
                                for algorithm in self.algorithms
                            ],
                            "action_strings": result.action_strings,
                            "special_commands": special_results,
                            "action_echo": {
                                "reported": len(echo_pairs),
                                "matched": len(echo_pairs) - len(echo_mismatches),
                                "mismatch_count": len(echo_mismatches),
                                "mismatches": echo_mismatches[:5],
                            },
                            "reward_sum": [float(item.sum()) for item in rewards],
                            "training_metrics": [
                                algorithm.training_metrics()
                                for algorithm in self.algorithms
                            ],
                            "events": result.events,
                            "entities": [
                                {
                                    "uid": item.uid,
                                    "team": item.team,
                                    "team_index": item.team_index,
                                    "type": item.entity_type,
                                    "alive": item.alive,
                                    "hp": item.hp,
                                    "position": item.position.tolist(),
                                    "velocity": item.velocity.tolist(),
                                    "yaw": item.yaw,
                                    "previous_action": item.raw.get("previousAction", ""),
                                }
                                for item in result.state.entities
                            ],
                            "key_objects": [
                                {
                                    "uid": item.uid,
                                    "class_name": item.class_name,
                                    "team": item.team,
                                    "position": item.position.tolist(),
                                    "hp": item.hp,
                                    "valid": item.valid,
                                    "rSVD1": item.raw.get("rSVD1", ""),
                                }
                                for item in result.state.key_objects
                            ],
                        }
                    )
                    state = result.state
                    previous_views = current_views
                    step = state.step
                    if result.done:
                        logger.info(
                            "回合结束: episode=%d winner=%s reason=%s step=%d",
                            state.episode,
                            result.winner,
                            result.end_reason,
                            state.step,
                        )
                        for algorithm in self.algorithms:
                            algorithm.on_episode_end(state.episode)
                        break

                if state.episode >= target_episodes:
                    break
                state = await self.environment.reset()
                self._synchronize_algorithm_contexts_for_new_episode(state)
                self._write_reset_log(state)
        finally:
            for algorithm in self.algorithms:
                try:
                    algorithm.on_simulation_end()
                except Exception:
                    # 清理钩子不能阻止另一方释放资源，也不能阻止gRPC关闭。
                    logger.exception(
                        "算法结束钩子失败: team=%s mode=%s",
                        getattr(getattr(algorithm, "context", None), "team", "unknown"),
                        getattr(algorithm, "mode", "unknown"),
                    )
            await self.environment.close()

    def _synchronize_algorithm_contexts_for_new_episode(
        self, state: SimulationState
    ) -> None:
        """新回合开始后刷新动态UID、公开敌方、指挥员和动作集合。"""

        self._prepare_team_views(state)

    def _prepare_team_views(self, state: SimulationState) -> list[TeamState]:
        """构造双方隔离视图，同步算法上下文，并生成本帧动作掩码。"""

        views = [self.environment.build_team_state(state, team) for team in (0, 1)]
        for team, algorithm in enumerate(self.algorithms):
            target_count = self._target_count(views[team])
            action_set = algorithm.context.action_set.with_target_count(target_count)
            algorithm.context = replace(
                algorithm.context,
                agents=views[team].agents,
                visible_opponents=views[team].visible_opponents,
                action_set=action_set,
                commanders=views[team].commanders,
            )
            if hasattr(algorithm, "build_action_mask"):
                views[team] = replace(
                    views[team], action_mask=algorithm.build_action_mask(views[team])
                )
        return views

    @staticmethod
    def _target_count(view: TeamState) -> int:
        """标准动作集只需保留单个 Agent 最大局部敌方槽数。"""

        return max(
            (len(view.perceived_opponents(agent)) for agent in view.agents),
            default=0,
        )

    @staticmethod
    def _filter_visible_events(
        events: Sequence[dict[str, str]],
        previous: TeamState,
        current: TeamState,
    ) -> tuple[dict[str, str], ...]:
        """只向算法公开与己方或本队可见实体有关的 UE 事件。"""

        allowed_uids = {
            item.uid
            for view in (previous, current)
            for item in (*view.agents, *view.commanders, *view.visible_opponents)
        }
        visible = []
        for event in events:
            if event.get("Event") == "EndEpisode":
                visible.append({
                    field: event[field]
                    for field in ("Event", "EndReason", "WinTeam")
                    if field in event
                })
                continue
            referenced = []
            for field in ("DamageCauser", "Target", "uId", "Uid"):
                if field not in event:
                    continue
                try:
                    referenced.append(int(event[field]))
                except (TypeError, ValueError):
                    pass
            if referenced and all(uid in allowed_uids for uid in referenced):
                visible.append({
                    field: event[field]
                    for field in ("Event", "DamageCauser", "Target")
                    if field in event
                })
        return tuple(visible)

    def _write_reset_log(self, state: SimulationState) -> None:
        team_views = {
            str(team): self.environment.build_team_state(state, team)
            for team in (0, 1)
        }
        self.run_log.write_simulation_record(
            {
                "kind": "reset",
                "episode": state.episode,
                "training": state.training,
                "team_state_counts": {
                    team: {
                        "agents": len(view.agents),
                        "visible_opponents": len(view.visible_opponents),
                        "commanders": len(view.commanders),
                        "key_objects": len(view.key_objects),
                    }
                    for team, view in team_views.items()
                },
                "action_sets": [
                    {
                        **algorithm.context.action_set.checkpoint_metadata(),
                        "fixed_count": algorithm.context.action_set.fixed_count,
                        "total_count": algorithm.context.action_set.n,
                    }
                    for algorithm in self.algorithms
                ],
                "roster": [
                    {
                        "uid": item.uid,
                        "team": item.team,
                        "team_index": item.team_index,
                        "type": item.entity_type,
                        "initial_hp": item.initial_hp,
                    }
                    for item in state.entities
                ],
                "commanders": [
                    {
                        "uid": item.uid,
                        "team": item.team,
                        "team_index": item.team_index,
                        "type": item.entity_type,
                        "class_name": item.class_name,
                        "position": item.position.tolist(),
                    }
                    for item in state.commanders
                ],
                "key_objects": [
                    {
                        "uid": item.uid,
                        "class_name": item.class_name,
                        "team": item.team,
                        "position": item.position.tolist(),
                        "hp": item.hp,
                        "valid": item.valid,
                        "rSVD1": item.raw.get("rSVD1", ""),
                    }
                    for item in state.key_objects
                ],
            }
        )
