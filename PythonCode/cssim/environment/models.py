from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np


_OWN_RAW_FIELDS = frozenset({
    "valid", "agentAlive", "agentTeam", "indexInTeam", "uId", "type",
    "maxMoveSpeed", "agentLocation", "agentLocationArr", "agentRotation",
    "agentRotationArr", "agentScale", "agentVelocity", "agentVelocityArr",
    "agentHp", "weaponCD", "previousAction", "agentPerception", "reward",
    "isTeamReward", "interaction", "rSVD1",
})


class VectorParser:
    """把 UE 坐标字段转换为统一的三维向量。"""

    @staticmethod
    def parse(value: Any, default: tuple[float, float, float]) -> np.ndarray:
        """兼容 UE 的 xyz 字典和数组表示。"""
        if isinstance(value, dict):
            return np.asarray(
                [value.get("x", 0), value.get("y", 0), value.get("z", 0)],
                dtype=np.float32,
            )
        if value is None:
            return np.asarray(default, dtype=np.float32)
        return np.asarray(value, dtype=np.float32)


@dataclass
class Entity:
    """UE 实体的稳定身份和最新动态状态。"""

    uid: int
    team: int  # 阵营编号：0=红方，1=蓝方。
    # 本方agents中是队内动作行号；公开敌方中只是当前可见列表序号。
    # 它不是UE UID，也不是UE的indexInTeam。
    team_index: int
    entity_type: str
    class_name: str
    max_speed: float
    initial_hp: float
    alive: bool = True
    hp: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    yaw: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_property(cls, item: dict[str, Any], team_index: int) -> "Entity":
        """从静态条目创建实体；team_index是Python分配的队内动作行号。"""
        rotation = item.get("initRotator", {})
        return cls(
            uid=int(item["uId"]),
            team=int(item["agentTeam"]),
            team_index=team_index,
            entity_type=str(item.get("type", "")),
            class_name=str(item.get("className", item.get("type", ""))),
            max_speed=float(item.get("maxMoveSpeed", 0.0)),
            initial_hp=float(item.get("agentHp", 0.0)),
            hp=float(item.get("agentHp", 0.0)),
            position=VectorParser.parse(item.get("initLocation"), (0.0, 0.0, 0.0)),
            velocity=VectorParser.parse(item.get("initVelocity"), (0.0, 0.0, 0.0)),
            yaw=float(rotation.get("yaw", 0.0)),
            properties=dict(item),
            raw=dict(item),
        )

    def update(self, item: dict[str, Any]) -> None:
        """用当前帧 dataArr 条目刷新实体动态状态。"""
        self.raw = dict(item)
        self.alive = bool(item.get("agentAlive", False))
        self.hp = float(item.get("agentHp", self.hp if self.alive else 0.0))
        if not self.alive:
            return
        self.max_speed = float(item.get("maxMoveSpeed", self.max_speed))
        self.position = VectorParser.parse(
            item.get("agentLocation", item.get("agentLocationArr")), tuple(self.position)
        )
        self.velocity = VectorParser.parse(
            item.get("agentVelocity", item.get("agentVelocityArr")), tuple(self.velocity)
        )
        rotation = item.get("agentRotation", item.get("agentRotationArr"))
        if isinstance(rotation, dict):
            self.yaw = float(rotation.get("yaw", self.yaw))
        elif rotation is not None:
            self.yaw = float(rotation[0])

    def perceived_uids(self) -> set[int]:
        """解析本实体当前帧的 agentPerception UID。"""

        result: set[int] = set()
        for item in self.raw.get("agentPerception", []) or []:
            value = item.get("uId") if isinstance(item, dict) else item
            try:
                result.add(int(value))
            except (TypeError, ValueError):
                continue
        return result

    def team_available_uids(self) -> set[int]:
        """解析 UE 为本队返回的 availActions 敌方候选 UID。"""

        result: set[int] = set()
        values = self.raw.get("availActions", self.properties.get("availActions", [])) or []
        for item in values:
            value = item.get("uId") if isinstance(item, dict) else item
            try:
                result.add(int(value))
            except (TypeError, ValueError):
                continue
        return result

    def commander_uid(self) -> int | None:
        """读取UE的indexInTeam，返回本实体当前上级UID。"""

        value = self.raw.get("indexInTeam", self.properties.get("indexInTeam"))
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def communication_ok(self) -> bool | None:
        """解析 rSVD1 末尾的通信状态；字段缺失时返回 None。"""

        text = str(self.raw.get("rSVD1", ""))
        _, separator, value = text.rpartition(";")
        if not separator:
            return None
        try:
            return int(value) == 1
        except ValueError:
            return None

    def own_team_view(self) -> "Entity":
        """生成算法侧己方副本，只保留审查过的逐帧原始字段。"""

        return Entity(
            uid=self.uid,
            team=self.team,
            team_index=self.team_index,
            entity_type=self.entity_type,
            class_name=self.class_name,
            max_speed=self.max_speed,
            initial_hp=self.initial_hp,
            alive=self.alive,
            hp=self.hp,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            yaw=self.yaw,
            properties={
                name: copy.deepcopy(value)
                for name, value in self.properties.items() if name != "availActions"
            },
            raw={
                name: copy.deepcopy(self.raw[name])
                for name in _OWN_RAW_FIELDS if name in self.raw
            },
        )

    def public_enemy_view(self, visible_index: int) -> "Entity":
        """生成敌方公开视图；team_index仅表示当前可见列表序号。"""

        return Entity(
            uid=self.uid,
            team=self.team,
            team_index=int(visible_index),
            entity_type=self.entity_type,
            class_name=self.entity_type,
            max_speed=self.max_speed,
            initial_hp=self.initial_hp,
            alive=self.alive,
            hp=self.hp,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            yaw=self.yaw,
        )


@dataclass(frozen=True)
class KeyObject:
    """UE 在 dataGlobal.keyObjArr 中返回的指挥所或其他战略目标。"""

    uid: int
    class_name: str
    team: int | None  # 目标归属阵营：0=红、1=蓝、None=UE未提供。
    position: np.ndarray
    velocity: np.ndarray
    hp: float | None
    valid: bool
    raw: dict[str, Any]

    @classmethod
    def from_raw(cls, item: dict[str, Any]) -> "KeyObject":
        """从 dataGlobal.keyObjArr 条目创建战略目标。"""
        hp = item.get("hp")
        team_value = next(
            (item.get(name) for name in ("team", "agentTeam", "ownerTeam") if item.get(name) is not None),
            None,
        )
        try:
            team = None if team_value is None else int(team_value)
        except (TypeError, ValueError):
            team = None
        return cls(
            uid=int(item["uId"]),
            class_name=str(item.get("className", item.get("type", ""))),
            team=team,
            position=VectorParser.parse(item.get("location"), (0.0, 0.0, 0.0)),
            velocity=VectorParser.parse(item.get("velocity"), (0.0, 0.0, 0.0)),
            hp=None if hp is None else float(hp),
            valid=bool(item.get("valid", True)),
            raw=dict(item),
        )

@dataclass(frozen=True)
class SimulationState:
    """框架内部的完整单帧快照，不直接交给任一方选手。"""

    entities: tuple[Entity, ...]
    raw: dict[str, Any]
    episode: int
    step: int
    training: bool
    commanders: tuple[Entity, ...] = ()
    key_objects: tuple[KeyObject, ...] = ()


@dataclass(frozen=True)
class TeamState:
    """提供给选手算法的本队视图。"""

    team: int  # 当前算法阵营：0=红方，1=蓝方。
    action_mask: np.ndarray
    agents: tuple[Entity, ...]
    # 队伍级唯一敌方接口：UE availActions 与全队单体感知并集的脱敏对象。
    visible_opponents: tuple[Entity, ...]
    raw: dict[str, Any]
    episode: int
    step: int
    training: bool
    commanders: tuple[Entity, ...] = ()
    key_objects: tuple[KeyObject, ...] = ()

    def entity_by_uid(self, uid: int) -> Entity | None:
        """按 UE 本轮动态 UID 查询战斗实体或指挥员。"""

        target = int(uid)
        return next(
            (
                item for item in (
                    *self.agents,
                    *self.visible_opponents,
                    *self.commanders,
                )
                if item.uid == target
            ),
            None,
        )

    def key_object_by_uid(self, uid: int) -> KeyObject | None:
        """按本轮动态 UID 查询指挥所或其他战略目标。"""

        target = int(uid)
        return next((item for item in self.key_objects if item.uid == target), None)

    def key_objects_for_team(self, team: int) -> tuple[KeyObject, ...]:
        """按归属方筛选目标；UE 未提供归属字段时回退为全部有效目标。"""

        valid = tuple(item for item in self.key_objects if item.valid)
        assigned = tuple(item for item in valid if item.team == int(team))
        if assigned:
            return assigned
        return valid if all(item.team is None for item in valid) else ()

    def perceived_opponents(self, agent: Entity) -> tuple[Entity, ...]:
        """返回该 Agent 当前可见的敌方公开视图。"""

        # 局部导入避免领域模型与特征模块在加载时形成循环依赖。
        from .observation import PerceptionModel

        return tuple(
            item for item in self.visible_opponents
            if PerceptionModel.can_observe(agent, item)
        )

    def current_commander(self, agent: Entity) -> Entity | None:
        """按动态指挥员 UID 查询本实体当前指挥员。"""

        uid = agent.commander_uid()
        commander = self.entity_by_uid(uid) if uid is not None else None
        return commander if commander is not None and commander.team == agent.team else None

    def nearest_friendly(
        self, origin: Entity, entity_type: str | None = None,
    ) -> Entity | None:
        """选择距离最近的本方存活实体；同距离时按 UID 保证结果稳定。"""

        candidates = (
            item for item in (*self.agents, *self.commanders)
            if item.alive
            and item.team == origin.team
            and item.uid != origin.uid
            and (entity_type is None or item.entity_type == entity_type)
        )
        return min(
            candidates,
            key=lambda item: (
                float(np.linalg.norm(origin.position - item.position)), item.uid
            ),
            default=None,
        )

    def nearest_available_parent(self, child: Entity) -> Entity | None:
        """返回离child最近的本方存活士兵或初始Commander。"""

        commander_uids = {item.uid for item in self.commanders}
        candidates = (
            item for item in (*self.agents, *self.commanders)
            if item.alive
            and item.team == child.team
            and item.uid != child.uid
            and (
                item.entity_type == "BP_BaseSoldier_C"
                or item.uid in commander_uids
            )
        )
        return min(
            candidates,
            key=lambda item: (
                float(np.linalg.norm(child.position - item.position)), item.uid
            ),
            default=None,
        )

    def nearest_key_object(self, origin: object, team: int | None = None) -> KeyObject | None:
        """返回离给定实体或坐标最近的有效战略目标，不假设目标数量或 UID。"""

        point = getattr(origin, "position", origin)
        try:
            location = np.asarray(point, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        candidates = (
            self.key_objects_for_team(team)
            if team is not None
            else tuple(item for item in self.key_objects if item.valid)
        )
        return min(
            candidates,
            key=lambda item: float(np.linalg.norm(location - item.position)),
            default=None,
        )
