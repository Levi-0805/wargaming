from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Any

from cssim.algorithms import Algorithm, AlgorithmContext


class AlgorithmLoader:
    """依据同一份 TaskConfig 加载并校验红蓝算法。"""

    def __init__(self, raw_config: dict[str, Any]):
        self.raw_config = raw_config
        self.global_config = next(
            (
                value for key, value in raw_config.items()
                if key.split("->")[-1] == "GlobalConfig" and isinstance(value, dict)
            ),
            {},
        )

    @staticmethod
    def _split_path(path: str) -> tuple[str, str]:
        if "->" not in path:
            raise ValueError(f"算法路径应采用 模块->类名 格式: {path}")
        module, class_name = path.split("->", 1)
        return module.removesuffix(".py"), class_name

    def _class_values(self, object_path: str) -> dict[str, Any]:
        normalized = object_path.replace(".py->", "->")
        for key, value in self.raw_config.items():
            if key.replace(".py->", "->") == normalized and isinstance(value, dict):
                return dict(value)
        return {}

    def load(self, path: str, context: AlgorithmContext) -> Algorithm:
        """加载 TaskConfig 指向的类，应用覆盖项并校验公共算法基类。"""

        module_name, class_name = self._split_path(path)
        module = importlib.import_module(module_name)

        pointer_name = "AlgorithmConfig_Red" if context.team == 0 else "AlgorithmConfig_Blue"
        config_pointer = self.global_config.get(pointer_name)
        config_class_name = "AlgorithmConfig"
        if isinstance(config_pointer, str) and "->" in config_pointer:
            config_module, config_class_name = self._split_path(config_pointer)
            if config_module.casefold() != module_name.casefold():
                raise ValueError(
                    f"{pointer_name} 与 Algorithm_Names 不一致: "
                    f"{config_module!r} != {module_name!r}"
                )

        # UE 可在 TaskConfig 中提供配置覆盖段；每个算法实例获得自己的配置字典。
        config_class = getattr(module, config_class_name, None)
        config_values: dict[str, Any] = {}
        if config_class is not None:
            config_path = f"{module_name}->{config_class_name}"
            config_values = self._class_values(config_path)

        algorithm_class = getattr(module, class_name)
        algorithm = algorithm_class(replace(context, config=config_values))
        if not isinstance(algorithm, Algorithm):
            raise TypeError(f"{path} 必须继承 cssim.algorithms.Algorithm")
        return algorithm
