from __future__ import annotations

import asyncio
import copy
import json
import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from cssim.config import RuntimeConfig
from cssim.protocol import Action, EventParser
from cssim.transport import GrpcClient

from .models import Entity, KeyObject, SimulationState, TeamState
from .observation import PerceptionModel


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepResult:
    """一次 UE frame 回包转换后的状态、终局信息和已发送动作。"""

    state: SimulationState
    done: bool
    winner: int | None
    end_reason: str
    events: tuple[dict[str, str], ...]
    action_strings: tuple[str, ...]


class CssimEnvironment:
    """gRPC 单环境封装，所有 UE 交互均在主事件循环内串行执行。"""

    def __init__(self, config: RuntimeConfig, protocol_writer=None):
        self.config = config
        self.client = GrpcClient(
            f"{config.host}:{config.port}", config.connect_timeout, protocol_writer
        )
        self.entities: tuple[Entity, ...] = ()
        self.commanders: tuple[Entity, ...] = ()
        self.episode = 0
        self.max_episodes = config.max_episodes
        self.training = False

    async def connect(self) -> None:
        """建立到 UE gRPC 服务的连接。"""
        await self.client.connect()

    def _build_prepare_request(self, command: str, step: int = 0) -> str:
        """构造reset_property或reset_obs使用的准备阶段JSON。"""

        return json.dumps(
            {
                "valid": True,
                "DataCmd": command,
                "NumAgents": 0,
                "TimeStepMax": self.config.max_steps,
                "TimeDilation": self.config.time_dilation,
                "FrameRate": self.config.frame_rate,
                "TimeStep": step,
                "Actions": None,
            },
            ensure_ascii=False,
        )

    async def reset(self) -> SimulationState:
        """等待 UE 开始仿真，重建动态编组并读取本回合首帧。"""

        properties = await self._wait_for_prepare(
            "reset_property",
            self._build_prepare_request("reset_property"),
            required_keys=("propertyArr",),
        )
        self._build_roster(properties)
        response = await self._wait_for_prepare(
            "reset_obs",
            self._build_prepare_request("reset_obs"),
            required_keys=("dataArr", "dataGlobal"),
        )
        game_config = response.get("dataGlobal", {}).get("rSVD1", "")
        if ";" in game_config:
            episodes, training = game_config.split(";", 1)
            self.max_episodes = int(episodes)
            self.training = bool(int(training))
        self.episode += 1
        state = self._parse_state(response)
        logger.info(
            "回合初始化: episode=%d agents=%d commanders=%d key_objects=%d training=%s level=%s",
            self.episode,
            len(self.entities),
            len(state.commanders),
            len(state.key_objects),
            self.training,
            response.get("dataGlobal", {}).get("levelName"),
        )
        return state

    async def _wait_for_prepare(
        self,
        type_name: str,
        payload: str,
        required_keys: tuple[str, ...],
    ) -> dict:
        """准备阶段持续重发，直到用户在 UE 中点击开始并返回有效数据。"""

        attempt = 0
        while True:
            attempt += 1
            try:
                # UE 在回合切换时需要销毁并重新生成实体，准备回包可能超过 2 秒。
                raw = await self.client.prepare(type_name, payload, timeout=10.0)
                response = json.loads(raw)
                missing = [key for key in required_keys if key not in response]
                if not missing:
                    return response
                logger.warning("UE %s 回包缺少字段 %s，继续等待", type_name, missing)
            except (TimeoutError, json.JSONDecodeError):
                if attempt == 1 or attempt % 15 == 0:
                    logger.info(
                        "等待 UE 开始仿真: prepare=%s attempts=%d",
                        type_name,
                        attempt,
                    )
            await asyncio.sleep(0.1)

    def _build_roster(self, response: dict) -> None:
        """按team分阵营，并为战斗实体分配各自队内的team_index。"""

        if "propertyArr" not in response:
            raise ValueError("reset_property 未返回 propertyArr，请检查 UE 的 IsUsePython")
        counts = {0: 0, 1: 0}
        commander_counts = {0: 0, 1: 0}
        roster: list[Entity] = []
        commanders: list[Entity] = []
        for item in response["propertyArr"]:
            team = int(item.get("agentTeam", -1))
            if team not in counts:
                continue
            entity_type = str(item.get("type", ""))
            class_name = str(item.get("className", ""))
            if entity_type == "BaseCommander_C" or class_name.startswith("BaseCommander_C"):
                commanders.append(Entity.from_property(item, commander_counts[team]))
                commander_counts[team] += 1
                continue
            roster.append(Entity.from_property(item, counts[team]))
            counts[team] += 1
        roster.sort(key=lambda entity: (entity.team, entity.team_index))
        commanders.sort(key=lambda entity: (entity.team, entity.team_index))
        if not roster or 0 in (counts[0], counts[1]):
            raise ValueError(f"UE 实体编组不完整: red={counts[0]} blue={counts[1]}")
        self.entities = tuple(roster)
        self.commanders = tuple(commanders)

    def _parse_state(self, response: dict) -> SimulationState:
        if not response.get("valid", False):
            raise ValueError("UE 返回 valid=false")
        by_uid = {int(item["uId"]): item for item in response.get("dataArr", [])}
        missing = [entity.uid for entity in self.entities if entity.uid not in by_uid]
        if missing:
            raise ValueError(f"UE 回包缺少实体: {missing}")
        ordered = [by_uid[entity.uid] for entity in self.entities]
        response["dataArr"] = ordered
        for entity, item in zip(self.entities, ordered):
            entity.update(item)
        # 当前 UE 通常不在逐帧 dataArr 中返回指挥员；若将来返回则同步其动态状态。
        for commander in self.commanders:
            if commander.uid in by_uid:
                commander.update(by_uid[commander.uid])
        step = int(response.get("dataGlobal", {}).get("timeCnt", 0))
        snapshot = tuple(copy.deepcopy(item) for item in self.entities)
        commander_snapshot = tuple(copy.deepcopy(item) for item in self.commanders)
        key_objects = tuple(
            KeyObject.from_raw(item)
            for item in response.get("dataGlobal", {}).get("keyObjArr", [])
            if isinstance(item, dict) and "uId" in item
        )
        return SimulationState(
            entities=snapshot,
            raw=response,
            episode=self.episode,
            step=step,
            training=self.training,
            commanders=commander_snapshot,
            key_objects=key_objects,
        )

    def build_team_state(self, state: SimulationState, team: int) -> TeamState:
        """生成指定阵营视图；team只接受0=红方或1=蓝方。"""

        # 必须从传入帧的快照构造视图，避免 UE 更新下一帧时污染上一帧经验。
        indices = [index for index, item in enumerate(state.entities) if item.team == team]
        # 算法获得独立副本，防止选手修改领域对象后污染运行器或另一方视图。
        internal_agents = tuple(state.entities[index] for index in indices)
        agents = tuple(item.own_team_view() for item in internal_agents)
        internal_opponents = tuple(item for item in state.entities if item.team != team)
        local_visible_uids = {
            opponent.uid
            for agent in agents
            for opponent in internal_opponents
            if PerceptionModel.can_observe(agent, opponent)
        }
        # availActions 是 UE 返回的队伍级敌方候选池；与逐实体感知取并集，避免快速变化帧
        # 的更新时序差异让任一已公开目标在同一帧中丢失。
        team_visible_uids = set(local_visible_uids)
        for agent in internal_agents:
            team_visible_uids.update(agent.team_available_uids())
        opponents = tuple(
            item.public_enemy_view(index)
            for index, item in enumerate(
                opponent
                for opponent in internal_opponents
                if opponent.alive and opponent.uid in team_visible_uids
            )
        )
        mask = np.ones((len(agents), 0), dtype=bool)
        data_global = state.raw.get("dataGlobal", {})
        commanders = tuple(
            item.own_team_view() for item in state.commanders if item.team == team
        )
        key_objects = copy.deepcopy(state.key_objects)
        public_raw = {
            "dataGlobal": {
                name: copy.deepcopy(data_global[name])
                for name in (
                    "timeCnt", "maxEpisodeStep", "levelName", "episodeDone", "rSVD1"
                )
                if name in data_global
            }
        }
        return TeamState(
            team=team,
            action_mask=mask,
            agents=agents,
            visible_opponents=opponents,
            raw=public_raw,
            episode=state.episode,
            step=state.step,
            training=state.training,
            commanders=commanders,
            key_objects=key_objects,
        )

    async def step(
        self, actions: Sequence[Action], step: int,
    ) -> StepResult:
        """编码并发送一个完整决策帧；超时后不重发可能已执行的动作。"""

        if len(actions) != len(self.entities):
            raise ValueError(f"动作数量应为 {len(self.entities)}，实际为 {len(actions)}")
        strings = []
        for action, entity in zip(actions, self.entities):
            if isinstance(action, Action):
                strings.append(action.to_ue_string(entity.uid, self.entities))
            else:
                raise TypeError(f"实体动作必须是Action: uid={entity.uid}")
        payload = json.dumps(
            {
                "valid": True,
                "DataCmd": "step",
                "TimeStep": step,
                "Actions": None,
                "StringActions": strings,
                "RSVD1": "None",
            },
            ensure_ascii=False,
        )
        try:
            raw_response = await self.client.frame(
                payload,
                sleep_time=self.config.step_game_time,
                timeout=self.config.reply_timeout,
            )
        except TimeoutError:
            # step 不能盲目重发：UE 可能已经执行动作，只是回包丢失。
            logger.error(
                "等待 UE 正式帧超时: episode=%d step=%d timeout=%.1fs",
                self.episode,
                step,
                self.config.reply_timeout,
            )
            raise
        response = json.loads(raw_response)
        state = self._parse_state(response)
        raw_events = response.get("dataGlobal", {}).get("events", [])
        events = tuple(EventParser.parse(item) for item in raw_events)
        done = bool(response.get("dataGlobal", {}).get("episodeDone", False))
        # timeCnt 从0开始；timeCnt=199 已经表示第200次动作执行完毕。
        if state.step + 1 >= self.config.max_steps:
            done = True
        winner, reason = self._terminal_result(events, done, response.get("dataGlobal", {}))
        return StepResult(state, done, winner, reason, events, tuple(strings))

    def _terminal_result(
        self, events: tuple[dict[str, str], ...], done: bool, data_global: dict
    ) -> tuple[int | None, str]:
        if not done:
            return None, ""
        end = next((event for event in events if event.get("Event") == "EndEpisode"), {})
        reason = end.get("EndReason", str(data_global.get("episodeEndReason", "")))
        for value in (end.get("WinTeam"), data_global.get("teamWin")):
            try:
                winner = int(value)
            except (TypeError, ValueError):
                continue
            if winner in (0, 1):
                return winner, reason

        # 两队初始数量可能不同，因此按存活率、剩余血量率裁决，避免数量较多的一方天然获胜。
        scores: list[tuple[float, float]] = []
        for team in (0, 1):
            members = [item for item in self.entities if item.team == team]
            survival = sum(item.alive for item in members) / len(members)
            total_hp = sum(max(item.hp, 0.0) for item in members)
            initial_hp = sum(max(item.initial_hp, 0.0) for item in members)
            scores.append((survival, total_hp / initial_hp if initial_hp else 0.0))
        if scores[0] == scores[1]:
            return -1, reason
        return (0 if scores[0] > scores[1] else 1), reason

    async def close(self) -> None:
        """尽力通知 UE 结束当前会话，并始终释放 gRPC 资源。"""

        try:
            try:
                await self.client.custom(
                    "end_unreal_engine",
                    json.dumps({"valid": True, "DataCmd": "end_unreal_engine"}),
                )
            except Exception as exc:
                logger.warning("gRPC 会话已经结束，跳过结束通知: %s", exc)
        finally:
            await self.client.close()
