from __future__ import annotations

from typing import Any

import numpy as np

from .models import Entity


class PerceptionModel:
    """集中解释 UE 单实体感知字段，不替选手构造模型特征。"""

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if np.isfinite(result) else default

    @staticmethod
    def in_range(observer: Entity, target: Entity) -> bool:
        """按 observer 的感知半径判断 target 是否位于几何范围内。"""
        merged = observer.properties | observer.raw
        # UE没有提供感知半径时不猜测默认值，避免框架制造额外可见信息。
        radius = PerceptionModel._number(merged.get("perceptionRange"), 0.0)
        return float(np.linalg.norm(observer.position - target.position)) <= max(radius, 0.0)

    @classmethod
    def can_observe(cls, observer: Entity, target: Entity) -> bool:
        """逐帧感知列表优先；旧回包没有该字段时才按观测半径回退。"""

        if not target.alive:
            return False
        if "agentPerception" in observer.raw:
            return target.uid in observer.perceived_uids()
        return cls.in_range(observer, target)
