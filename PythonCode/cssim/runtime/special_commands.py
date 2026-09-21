from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Sequence

from cssim.environment import SimulationState
from cssim.protocol import ChangeParentAction
from cssim.transport import GrpcClient


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpecialCommandResult:
    """特殊指令执行结果，用于仿真日志。"""

    team: int  # 发起动作的阵营：0=红方，1=蓝方。
    command: str
    success: bool
    parameters: dict[str, object]
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SpecialCommandExecutor:
    """校验并执行框架当前正式开放的指派上级动作。"""

    def __init__(self, client: GrpcClient):
        self.client = client

    @staticmethod
    def _validate(
        team: int, state: SimulationState, command: ChangeParentAction,
    ) -> None:
        children = {item.uid: item for item in state.entities}
        parents = {item.uid: item for item in (*state.entities, *state.commanders)}
        child = children.get(command.child_uid)
        parent = parents.get(command.parent_uid)
        if child is None or child.team != team:
            raise ValueError(f"child 必须是本方战斗实体: uid={command.child_uid}")
        if parent is None or parent.team != team:
            raise ValueError(f"parent 必须是本方实体或指挥员: uid={command.parent_uid}")
        if not child.alive or not parent.alive:
            raise ValueError("不能为阵亡实体设置上级")

    async def execute(
        self,
        team: int,
        state: SimulationState,
        commands: Sequence[ChangeParentAction],
    ) -> tuple[SpecialCommandResult, ...]:
        """逐条执行本帧被 Agent 选中的特殊动作，不跨帧去重。"""

        results = []
        for command in commands:
            if not isinstance(command, ChangeParentAction):
                raise TypeError(f"未开放的特殊指令: {type(command).__name__}")
            self._validate(team, state, command)
            parameters = command.parameters()
            try:
                response = await self.client.change_actor_parent(
                    command.parent_uid, command.child_uid
                )
                success = bool(response.get("success", False))
                message = str(response.get("text", ""))
            except Exception as exc:  # 特殊RPC失败不影响普通帧动作。
                success = False
                message = str(exc)
                logger.exception(
                    "特殊指令调用失败: team=%d command=%s parameters=%s",
                    team,
                    command.command_name,
                    parameters,
                )
            logger.info(
                "特殊指令结果: team=%d command=%s parameters=%s success=%s message=%s",
                team,
                command.command_name,
                parameters,
                success,
                message,
            )
            results.append(SpecialCommandResult(
                team, command.command_name, success, parameters, message=message
            ))
        return tuple(results)
