from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable

import numpy as np

from .actions import (
    ALL_MOVE_DIRECTIONS,
    GRENADE_DIRECTIONS,
    GUARD_DIRECTIONS,
    Action,
)
from .special_commands import ChangeParentAction


@dataclass(frozen=True)
class _ResolvedAction:
    """框架内部的离散动作解析结果，不属于选手接口。"""

    frame_action: Action
    special_command: ChangeParentAction | None = None

    @property
    def output(self) -> Action:
        return self.frame_action

    @property
    def special_commands(self) -> tuple[ChangeParentAction, ...]:
        return () if self.special_command is None else (self.special_command,)


@dataclass(frozen=True)
class DiscreteAction:
    """一个强化学习离散槽；槽位解析器直接返回动作对象。"""

    key: str
    _resolver: Callable[
        [object | None, object | None],
        Action | ChangeParentAction | _ResolvedAction,
    ]
    _available: Callable[[object, object], bool] | None = None

    def __post_init__(self) -> None:
        if not self.key or ":target:" in self.key:
            raise ValueError(f"无效的离散动作键: {self.key!r}")

    @classmethod
    def fixed(cls, key: str, action: Action) -> "DiscreteAction":
        """定义不需要当前态势参数的普通动作。"""

        return cls(key, lambda _state, _agent, value=action: value)

    @classmethod
    def dynamic(
        cls,
        key: str,
        resolver: Callable[
            [object, object], Action | ChangeParentAction
        ],
        available: Callable[[object, object], bool] | None = None,
    ) -> "DiscreteAction":
        """定义动态槽；resolver根据当前态势直接创建动作对象。"""

        return cls(key, resolver, available)

    def is_available(self, state: object, agent_index: int) -> bool:
        agent = tuple(getattr(state, "agents"))[agent_index]
        return self._available is None or bool(self._available(state, agent))

    def resolve(
        self, state: object | None, agent_index: int | None,
    ) -> _ResolvedAction:
        agent = None
        if state is not None and agent_index is not None:
            agent = tuple(getattr(state, "agents"))[agent_index]
        resolved = self._resolver(state, agent)
        if isinstance(resolved, _ResolvedAction):
            return resolved
        if isinstance(resolved, Action):
            return _ResolvedAction(resolved)
        if isinstance(resolved, ChangeParentAction):
            return _ResolvedAction(Action.idle(), resolved)
        raise TypeError(
            f"离散动作 {self.key!r} 必须生成Action或ChangeParentAction"
        )


@dataclass(frozen=True)
class RLActionSet:
    """框架运行时使用的可版本化离散动作集合。"""

    name: str
    version: int
    choices: tuple[DiscreteAction, ...]
    include_target_attacks: bool = True
    target_count: int = 0

    def __post_init__(self) -> None:
        keys = self.fixed_keys
        if not self.name or self.version < 1:
            raise ValueError("动作集合必须提供名称和正整数版本")
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("固定动作键必须非空且唯一")
        if self.target_count < 0:
            raise ValueError("攻击目标槽数量不能为负数")

    @property
    def fixed_keys(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.choices)

    @property
    def fixed_count(self) -> int:
        return len(self.choices)

    @property
    def n(self) -> int:
        return self.fixed_count + (self.target_count if self.include_target_attacks else 0)

    @property
    def action_keys(self) -> tuple[str, ...]:
        attacks = tuple(
            f"attack:target:{index}" for index in range(self.target_count)
        ) if self.include_target_attacks else ()
        return (*self.fixed_keys, *attacks)

    @property
    def signature(self) -> str:
        payload = json.dumps(
            {
                "name": self.name,
                "version": self.version,
                "fixed_keys": self.fixed_keys,
                "include_target_attacks": self.include_target_attacks,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "action_set_name": self.name,
            "action_set_version": self.version,
            "action_keys": list(self.fixed_keys),
            "action_signature": self.signature,
        }

    def with_target_count(self, target_count: int) -> "RLActionSet":
        return RLActionSet(
            self.name,
            self.version,
            self.choices,
            self.include_target_attacks,
            int(target_count),
        )

    def action_mask(self, state: object) -> np.ndarray:
        agents = tuple(getattr(state, "agents"))
        mask = np.ones((len(agents), self.n), dtype=bool)
        for row in range(len(agents)):
            for column, choice in enumerate(self.choices):
                mask[row, column] = choice.is_available(state, row)
        if self.include_target_attacks:
            mask[:, self.fixed_count:] = False
            for row, agent in enumerate(agents):
                count = min(
                    len(getattr(state, "perceived_opponents")(agent)),
                    self.target_count,
                )
                mask[row, self.fixed_count:self.fixed_count + count] = True
        return mask

    def decode(
        self, index: int, state: object | None = None, agent_index: int | None = None,
    ) -> _ResolvedAction:
        index = int(index)
        if not 0 <= index < self.n:
            raise ValueError(f"动作索引越界: {index}, 合法范围为 [0, {self.n})")
        if index < self.fixed_count:
            return self.choices[index].resolve(state, agent_index)
        if state is None or agent_index is None:
            raise ValueError("逐目标攻击动作必须结合当前Agent状态解码")
        agent = tuple(getattr(state, "agents"))[agent_index]
        targets = tuple(getattr(state, "perceived_opponents")(agent))
        target_index = index - self.fixed_count
        if target_index >= len(targets):
            raise ValueError(f"当前Agent的可见攻击目标槽不存在: {target_index}")
        return _ResolvedAction(Action.attack(targets[target_index]))


class ActionSetFactory:
    """创建框架标准强化学习动作集合。"""

    @classmethod
    def standard(cls, target_count: int = 0) -> RLActionSet:
        choices = [DiscreteAction.fixed("idle:N/A", Action.idle())]
        choices.extend(
            DiscreteAction.fixed(f"move:{direction}", Action.move(direction))
            for direction in ALL_MOVE_DIRECTIONS
        )
        choices.extend(
            DiscreteAction.fixed(f"guard:{direction}", Action.guard(direction))
            for direction in GUARD_DIRECTIONS
        )
        choices.extend(
            DiscreteAction.fixed(f"grenade:{direction}", Action.grenade(direction))
            for direction in GRENADE_DIRECTIONS
        )
        choices.extend((
            DiscreteAction.fixed("mine:plant", Action.plant_mine()),
            DiscreteAction.fixed("mine:detonate", Action.detonate_mine()),
        ))
        return RLActionSet(
            "cssim_standard_v1",
            1,
            tuple(choices),
            include_target_attacks=True,
            target_count=int(target_count),
        )
