from __future__ import annotations

import re

class EventParser:
    """解析 UE 的 ``<字段>值`` 事件格式。"""

    _field = re.compile(r"<([^<>]*)>([^<>]*)")

    @classmethod
    def parse(cls, raw: str) -> dict[str, str]:
        return dict(cls._field.findall(raw or ""))
