from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Iterable


# 用户手册表 4 中规则模式可直接使用的完整移动方向。
ALL_MOVE_DIRECTIONS = (
    "+X", "+Y", "-X", "-Y",
    "+X+Y", "-X+Y", "-X-Y", "+X-Y",
    "+X+Z", "+Y+Z", "-X+Z", "-Y+Z",
    "+X+Y+Z", "-X+Y+Z", "-X-Y+Z", "+X-Y+Z",
    "+X-Z", "+Y-Z", "-X-Z", "-Y-Z",
    "+X+Y-Z", "-X+Y-Z", "-X-Y-Z", "+X-Y-Z",
    "-Z", "+Z",
)
GUARD_DIRECTIONS = ("+X", "+Y", "-X", "-Y", "+X+Y", "-X+Y", "-X-Y", "+X-Y")
GRENADE_DIRECTIONS = GUARD_DIRECTIONS
SUPPORTED_ACTION_COMMANDS = (
    "Idle", "Moving", "Guard", "NormalAttacking", "GrenadeAttacking",
    "PlantMine", "DetonateMine", "SelfDestruct", "Board",
)


@dataclass(frozen=True)
class Action:
    """选手使用的结构化动作；发送前统一生成 UE 的 ActionSet2 字符串。"""

    command: str
    direction: str = "N/A"
    points: tuple[float, ...] | None = None
    target_uid: int | None = None

    def __post_init__(self) -> None:
        if self.command not in SUPPORTED_ACTION_COMMANDS:
            raise ValueError(f"当前 UE C++ 不支持动作: {self.command}")

    @staticmethod
    def _points(value: object) -> tuple[float, ...]:
        """接受坐标、路径点列表或领域对象，并统一展平成 xyz 序列。"""

        if hasattr(value, "position"):
            value = getattr(value, "position")
        if isinstance(value, (str, bytes)):
            raise ValueError("坐标不能是字符串")
        try:
            items = list(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("坐标必须是可迭代的 xyz 或路径点序列") from exc

        values: list[float] = []
        for item in items:
            if isinstance(item, Real):
                values.append(float(item))
            else:
                try:
                    values.extend(float(number) for number in item)
                except TypeError as exc:
                    raise ValueError("路径点必须由 xyz 三元组组成") from exc
        if not values or len(values) % 3:
            raise ValueError("坐标必须是非空的 xyz 三元组序列")
        return tuple(values)

    @staticmethod
    def _target_uid(target: object) -> int:
        """攻击目标既可传 UE 动态 UID，也可直接传 Entity/KeyObject。"""

        value = getattr(target, "uid", target)
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("攻击目标必须包含有效的动态 UID") from exc

    @classmethod
    def idle(cls) -> "Action":
        """保持当前实体空闲。"""
        return cls("Idle")

    @classmethod
    def move(cls, direction: str) -> "Action":
        """按 UE 支持的方向持续移动。"""
        if direction not in ALL_MOVE_DIRECTIONS:
            raise ValueError(f"未知移动方向: {direction}")
        return cls("Moving", direction=direction)

    @classmethod
    def move_to(cls, points: object) -> "Action":
        """沿一个或多个 xyz 路径点移动。"""
        return cls("Moving", points=cls._points(points))

    @classmethod
    def move_at(cls, point: object) -> "Action":
        """移动至单个 xyz 坐标。"""
        values = cls._points(point)
        if len(values) != 3:
            raise ValueError("坐标移动只能包含一个 xyz 点")
        return cls("Moving", points=values)

    @classmethod
    def guard(cls, direction: str) -> "Action":
        """警戒 UE 支持的水平方向。"""
        if direction not in GUARD_DIRECTIONS:
            raise ValueError(f"未知警戒方向: {direction}")
        return cls("Guard", direction=direction)

    @classmethod
    def guard_at(cls, points: object) -> "Action":
        """沿一个或多个 xyz 路径点执行警戒。"""
        return cls("Guard", points=cls._points(points))

    @classmethod
    def guard_position(cls, point: object) -> "Action":
        """在单个 xyz 坐标执行警戒。"""
        values = cls._points(point)
        if len(values) != 3:
            raise ValueError("坐标警戒只能包含一个 xyz 点")
        return cls("Guard", points=values)

    @classmethod
    def attack(cls, target: object) -> "Action":
        """普通攻击一个 Entity、KeyObject 或动态 UID。"""
        return cls("NormalAttacking", target_uid=cls._target_uid(target))

    @classmethod
    def attack_uid(cls, uid: int) -> "Action":
        """普通攻击指定的本轮动态 UID。"""
        return cls.attack(uid)

    @classmethod
    def grenade(cls, direction: str) -> "Action":
        """向 UE 支持的水平方向投掷手雷。"""
        if direction not in GRENADE_DIRECTIONS:
            raise ValueError(f"手册未定义该手雷攻击方向: {direction}")
        return cls("GrenadeAttacking", direction=direction)

    @classmethod
    def plant_mine(cls) -> "Action":
        """在当前位置布雷。"""
        return cls("PlantMine")

    @classmethod
    def detonate_mine(cls) -> "Action":
        """引爆当前实体已布设的地雷。"""
        return cls("DetonateMine")

    @classmethod
    def self_destruct(cls, point: object | None = None) -> "Action":
        """在指定 xyz 坐标执行自爆。"""
        if point is None:
            raise ValueError("SelfDestruct 必须提供目标坐标")
        values = cls._points(point)
        if len(values) != 3:
            raise ValueError("SelfDestruct 目标只能包含一个 xyz 点")
        return cls("SelfDestruct", points=values)

    @classmethod
    def board(cls, vehicle: object) -> "Action":
        """登乘指定载具；执行者 UID 由运行器编码时补充。"""
        return cls("Board", target_uid=cls._target_uid(vehicle))

    def to_ue_string(self, executor_uid: int, roster: Iterable[object]) -> str:
        """转换为 UE 当前识别的最终字符串，并补上执行者的动态 UID。"""

        roster = tuple(roster)
        executor = next(
            (item for item in roster if int(getattr(item, "uid")) == int(executor_uid)),
            None,
        )
        if executor is None:
            raise ValueError(f"执行动作的实体不存在: uid={executor_uid}")
        if self.command == "Moving":
            if self.points is not None:
                body = f"ActionSet2::Moving;;Points={list(self.points)}"
            else:
                if self.direction not in ALL_MOVE_DIRECTIONS:
                    raise ValueError(f"当前 UE C++ 不支持移动方向: {self.direction}")
                body = f"ActionSet2::Moving;{self.direction}"
        elif self.command == "NormalAttacking":
            if self.target_uid is None:
                raise ValueError("普通攻击缺少目标 UID")
            body = f"ActionSet2::NormalAttacking;{self.target_uid}"
        elif self.command == "Guard" and self.points is not None:
            body = f"ActionSet2::Guard;;Points={list(self.points)}"
        elif self.command == "Guard":
            if self.direction not in GUARD_DIRECTIONS:
                raise ValueError(f"当前 UE C++ 不支持警戒方向: {self.direction}")
            body = f"ActionSet2::Guard;{self.direction}"
        elif self.command == "GrenadeAttacking":
            if self.points is not None or self.direction not in GRENADE_DIRECTIONS:
                raise ValueError("当前 UE C++ 的手雷动作只支持八个水平方向")
            body = f"ActionSet2::GrenadeAttacking;{self.direction}"
        elif self.command == "SelfDestruct" and self.points is not None:
            if len(self.points) != 3:
                raise ValueError("SelfDestruct 目标只能包含一个 xyz 点")
            body = f"ActionSet2::SelfDestruct;;Points={list(self.points)}"
        elif self.command == "Idle":
            body = "ActionSet2::Idle;N/A"
        elif self.command in {"PlantMine", "DetonateMine"}:
            body = f"ActionSet2::{self.command};N/A"
        elif self.command == "Board":
            if self.target_uid is None:
                raise ValueError("Board 缺少目标载具 UID")
            body = f"ActionSet2::Board;;Uid={int(self.target_uid)}"
        else:
            raise ValueError(f"暂不支持的 UE 动作: {self.command}")
        return f"{body};{int(executor_uid)}"
