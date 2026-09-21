from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import grpc
import pytest


generated = Path(__file__).resolve().parents[1] / "cssim" / "transport" / "_generated_grpc"
sys.path.insert(0, str(generated))
import custom_network_pb2_grpc as pb2_grpc  # noqa: E402
import custom_network_pb2 as pb2  # noqa: E402

from cssim.transport import GrpcClient  # noqa: E402


class _EchoService(pb2_grpc.CustomNetworkServiceServicer):
    def __init__(self):
        self.prepare_count = 0
        self.frame_sleep_time = None

    async def DataChannel(self, request_iterator, context):
        async for message in request_iterator:
            if message.WhichOneof("body") == "handshake":
                # 复现实际 UE：握手确认前可能先到达一条无请求者的 frame。
                yield pb2.StreamMessage(
                    info=message.info,
                    frame=pb2.Payload(data=b"early-frame", sleepTime=0),
                )
            if message.WhichOneof("body") == "prepare":
                self.prepare_count += 1
                if self.prepare_count == 1:
                    # 复现 UE 尚未点击开始时忽略第一条准备请求。
                    continue
            if message.WhichOneof("body") == "frame":
                self.frame_sleep_time = message.frame.sleepTime
            yield message


class _NoFrameReplyService(_EchoService):
    async def DataChannel(self, request_iterator, context):
        async for message in request_iterator:
            if message.WhichOneof("body") == "frame":
                continue
            yield message


def test_single_loop_grpc_request_reply():
    async def scenario():
        service = _EchoService()
        server = grpc.aio.server()
        pb2_grpc.add_CustomNetworkServiceServicer_to_server(service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        client = GrpcClient(f"127.0.0.1:{port}", timeout=5)
        try:
            await client.connect()
            with pytest.raises(TimeoutError):
                await client.prepare(
                    "reset_property", '{"DataCmd":"reset_property"}', timeout=0.05
                )
            assert await client.prepare(
                "reset_property", '{"DataCmd":"reset_property"}', timeout=1
            ) == '{"DataCmd":"reset_property"}'
            assert await client.frame(
                '{"DataCmd":"step"}', sleep_time=0.25, timeout=1
            ) == '{"DataCmd":"step"}'
            assert service.frame_sleep_time == 0.25
        finally:
            await client.close()
            await server.stop(0)

    asyncio.run(scenario())


def test_frame_reply_timeout():
    async def scenario():
        server = grpc.aio.server()
        pb2_grpc.add_CustomNetworkServiceServicer_to_server(_NoFrameReplyService(), server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        client = GrpcClient(f"127.0.0.1:{port}", timeout=5)
        try:
            await client.connect()
            with pytest.raises(TimeoutError):
                await client.frame('{"DataCmd":"step"}', timeout=0.05)
        finally:
            await client.close()
            await server.stop(0)

    asyncio.run(scenario())
