from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable

import grpc


# 生成代码沿用 protoc 的顶层导入形式，仅在本模块内补充搜索路径。
_GENERATED = Path(__file__).resolve().parent / "_generated_grpc"
if str(_GENERATED) not in sys.path:
    sys.path.insert(0, str(_GENERATED))

import custom_network_pb2 as pb2  # type: ignore  # noqa: E402
import custom_network_pb2_grpc as pb2_grpc  # type: ignore  # noqa: E402
import custom_interface_pb2 as interface_pb2  # type: ignore  # noqa: E402
import custom_interface_pb2_grpc as interface_pb2_grpc  # type: ignore  # noqa: E402
import util_pb2  # type: ignore  # noqa: E402


logger = logging.getLogger(__name__)


class GrpcClient:
    """单事件循环 gRPC 客户端，不创建工作线程或守护进程。"""

    def __init__(
        self,
        endpoint: str,
        timeout: float = 60.0,
        protocol_writer: Callable[[str, str, str, str], None] | None = None,
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self.client_id = f"py-{uuid.uuid4().hex[:12]}"
        self.client_name = f"cssim-{uuid.uuid4().hex[:6]}"
        self._channel: grpc.aio.Channel | None = None
        self._interface_stub: Any = None
        self._call: Any = None
        self._reader_task: asyncio.Task | None = None
        self._waiters: dict[str, deque[asyncio.Future]] = defaultdict(deque)
        self._protocol_writer = protocol_writer

    def _record_protocol(
        self, direction: str, channel: str, type_name: str, payload: str
    ) -> None:
        if self._protocol_writer is not None:
            self._protocol_writer(direction, channel, type_name, payload)

    def _connect_info(self) -> pb2.ConnectInfo:
        """构造每条gRPC消息携带的当前Python客户端身份。"""

        return pb2.ConnectInfo(id=self.client_id, name=self.client_name)

    async def connect(self) -> None:
        """建立双向流并完成握手；连接失败时在超时范围内重试。"""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        last_error: BaseException | None = None
        while loop.time() < deadline:
            try:
                self._channel = grpc.aio.insecure_channel(
                    self.endpoint,
                    options=[
                        ("grpc.max_send_message_length", 100 * 1024 * 1024),
                        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                    ],
                )
                await asyncio.wait_for(self._channel.channel_ready(), timeout=5.0)
                stub = pb2_grpc.CustomNetworkServiceStub(self._channel)
                self._interface_stub = interface_pb2_grpc.CustomInterfaceServiceStub(
                    self._channel
                )
                self._call = stub.DataChannel()
                self._reader_task = asyncio.create_task(
                    self._reader_loop(), name=f"{self.client_name}-grpc-reader"
                )
                self._record_protocol(
                    "python_to_ue", "handshake", "handshake",
                    json.dumps({
                        "info": {"id": self.client_id, "name": self.client_name},
                        "protocol_version": 0,
                    }),
                )
                reply = await self._write_and_wait(
                    "handshake",
                    pb2.StreamMessage(
                        info=self._connect_info(),
                        handshake=pb2.Handshake(protocol_version=0),
                    ),
                    timeout=5.0,
                )
                self._record_protocol(
                    "ue_to_python", "handshake", "handshake",
                    json.dumps({
                        "protocol_version": reply.handshake.protocol_version,
                    }),
                )
                logger.info(
                    "gRPC 握手完成: endpoint=%s client_id=%s version=%s",
                    self.endpoint,
                    self.client_id,
                    reply.handshake.protocol_version,
                )
                return
            except (grpc.RpcError, TimeoutError, asyncio.TimeoutError, ConnectionError) as exc:
                last_error = exc
                await self.close()
                await asyncio.sleep(1.0)
        raise ConnectionError(f"连接 UE gRPC 超时: {self.endpoint}") from last_error

    def _new_waiter(self, body: str) -> asyncio.Future:
        future = asyncio.get_running_loop().create_future()
        self._waiters[body].append(future)
        return future

    async def _write_and_wait(self, body: str, message, timeout: float | None = None):
        if self._call is None:
            raise RuntimeError("gRPC 尚未连接")
        # 必须先登记等待者再写请求，避免 UE 的快速回包先于 Python 开始读取。
        waiter = self._new_waiter(body)
        try:
            await self._call.write(message)
            try:
                if timeout is None:
                    return await waiter
                return await asyncio.wait_for(waiter, timeout)
            except asyncio.TimeoutError as exc:
                raise TimeoutError(f"等待 UE 的 {body} 回包超时") from exc
        except BaseException:
            if not waiter.done():
                waiter.cancel()
            raise

    async def _reader_loop(self) -> None:
        """持续接收并按 oneof 类型分发；它是同一事件循环中的任务，不创建线程。"""

        try:
            while self._call is not None:
                message = await self._call.read()
                if message is grpc.aio.EOF:
                    raise ConnectionError("UE 已关闭 gRPC 数据流")
                body = message.WhichOneof("body")
                if body is None:
                    logger.warning("收到没有消息体的 gRPC 消息")
                    continue
                waiters = self._waiters[body]
                while waiters and waiters[0].done():
                    waiters.popleft()
                if waiters:
                    waiters.popleft().set_result(message)
                    continue
                payload = getattr(message, body, None)
                preview = ""
                if payload is not None and hasattr(payload, "data"):
                    raw = bytes(payload.data).decode("utf-8", errors="replace")
                    preview = raw[:300]
                    self._record_protocol("ue_to_python", body, "unsolicited", raw)
                logger.warning("收到无等待者的 gRPC 消息: body=%s data=%s", body, preview)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            for waiters in self._waiters.values():
                while waiters:
                    waiter = waiters.popleft()
                    if not waiter.done():
                        waiter.set_exception(exc)
            logger.error("gRPC 接收循环结束: %s", exc)

    async def prepare(
        self, type_name: str, payload: str, timeout: float | None = None
    ) -> str:
        """发送reset_property/reset_obs等准备请求并等待同类型回包。"""

        if self._call is None:
            raise RuntimeError("gRPC 尚未连接")
        self._record_protocol("python_to_ue", "prepare", type_name, payload)
        logger.debug("gRPC JSON 发送: channel=prepare type=%s payload=%s", type_name, payload)
        reply = await self._write_and_wait(
            "prepare",
            pb2.StreamMessage(
                info=self._connect_info(),
                prepare=pb2.Prepare(type=type_name, data=payload.encode("utf-8")),
            ),
            timeout=timeout,
        )
        response = reply.prepare.data.decode("utf-8")
        self._record_protocol("ue_to_python", "prepare", type_name, response)
        logger.debug("gRPC JSON 接收: channel=prepare type=%s payload=%s", type_name, response)
        return response

    async def frame(
        self,
        payload: str,
        sleep_time: float = 2.0,
        timeout: float | None = None,
    ) -> str:
        """发送一步动作JSON，并等待UE推进指定游戏时间后的状态JSON。"""

        if self._call is None:
            raise RuntimeError("gRPC 尚未连接")
        self._record_protocol("python_to_ue", "frame", "step", payload)
        logger.debug("gRPC JSON 发送: channel=frame payload=%s", payload)
        reply = await self._write_and_wait(
            "frame",
            pb2.StreamMessage(
                info=self._connect_info(),
                frame=pb2.Payload(data=payload.encode("utf-8"), sleepTime=sleep_time),
            ),
            timeout=timeout,
        )
        response = reply.frame.data.decode("utf-8")
        self._record_protocol("ue_to_python", "frame", "step", response)
        logger.debug("gRPC JSON 接收: channel=frame payload=%s", response)
        return response

    async def custom(self, type_name: str, payload: str) -> None:
        """发送无需等待响应的双向流控制消息。"""

        if self._call is None:
            return
        self._record_protocol("python_to_ue", "custom", type_name, payload)
        logger.debug("gRPC JSON 发送: channel=custom type=%s payload=%s", type_name, payload)
        await self._call.write(
            pb2.StreamMessage(
                info=self._connect_info(),
                custom=util_pb2.Custom(type=type_name, data=payload.encode("utf-8")),
            )
        )

    async def change_actor_parent(
        self, parent_uid: int, child_uid: int, timeout: float = 10.0,
    ) -> dict[str, object]:
        """通过独立 CustomInterfaceService 指派实体上级。"""

        if self._interface_stub is None:
            raise RuntimeError("gRPC 尚未连接")
        request = interface_pb2.InterfaceMessage()
        request.info.user_id = self.client_id
        request.info.user_name = self.client_name
        request.changeactorparent.CopyFrom(interface_pb2.ChangeActorParent(
            parent=str(int(parent_uid)), child=str(int(child_uid)),
        ))
        payload = json.dumps({
            "parent": str(int(parent_uid)), "child": str(int(child_uid)),
        })
        self._record_protocol(
            "python_to_ue", "special", "ChangeActorParent", payload
        )
        response = await self._interface_stub.InterfaceCall(request, timeout=timeout)
        if response.HasField("interfaceresult"):
            raw_text = bytes(response.interfaceresult.text)
            result = {
                "success": bool(response.interfaceresult.success),
                "text": raw_text.decode("utf-8", errors="replace"),
            }
        else:
            result = {"success": False, "text": "UE 未返回 InterfaceResult"}
        self._record_protocol(
            "ue_to_python", "special", "ChangeActorParent",
            json.dumps(result, ensure_ascii=False),
        )
        return result

    async def close(self) -> None:
        """幂等关闭双向流、接收任务和底层channel。"""

        call, channel, reader = self._call, self._channel, self._reader_task
        self._call = None
        self._channel = None
        self._interface_stub = None
        self._reader_task = None
        if call is not None:
            with contextlib.suppress(Exception):
                await call.done_writing()
            call.cancel()
        if reader is not None:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        if channel is not None:
            await channel.close()
