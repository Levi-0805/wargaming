from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, init=False)
class ChangeParentAction:
    """指派上级动作；创建对象时可直接传实体或本轮 UID。"""

    command_name: ClassVar[str] = "ChangeActorParent"
    parent_uid: int
    child_uid: int

    def __init__(self, parent: object, child: object) -> None:
        self._require_alive(parent, "parent")
        self._require_alive(child, "child")
        object.__setattr__(self, "parent_uid", self._uid(parent, "parent"))
        object.__setattr__(self, "child_uid", self._uid(child, "child"))
        if self.parent_uid == self.child_uid:
            raise ValueError("实体不能成为自己的上级")

    @staticmethod
    def _require_alive(value: object, name: str) -> None:
        """传入实体对象时立即拒绝阵亡对象；仅传UID时由执行层复核。"""

        if hasattr(value, "alive") and not bool(getattr(value, "alive")):
            raise ValueError(f"{name} 必须是存活实体")

    @staticmethod
    def _uid(value: object, name: str) -> int:
        value = getattr(value, "uid", value)
        try:
            uid = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 必须包含有效的本轮实体 UID") from exc
        if uid < 0:
            raise ValueError(f"{name} UID 不能为负数")
        return uid

    def parameters(self) -> dict[str, object]:
        """返回日志使用的父子动态 UID。"""
        return {"parent_uid": self.parent_uid, "child_uid": self.child_uid}
