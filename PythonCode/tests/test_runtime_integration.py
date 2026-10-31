from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import grpc
import pytest


generated = Path(__file__).resolve().parents[1] / "cssim" / "transport" / "_generated_grpc"
sys.path.insert(0, str(generated))
import custom_network_pb2 as pb2  # noqa: E402
import custom_network_pb2_grpc as pb2_grpc  # noqa: E402

from cssim.config import RuntimeConfig  # noqa: E402
from cssim.logging_setup import SimulationLog  # noqa: E402
from cssim.runtime import Runner  # noqa: E402


class _TwoEpisodeUeService(pb2_grpc.CustomNetworkServiceServicer):
    """最小假UE：完整执行两个单步回合，并在第二回合更换实体UID。"""

    def __init__(self):
        self.episode = 0
        self.red_uid = 0
        self.blue_uid = 0
        self.prepare_types: list[str] = []
        self.frame_requests: list[dict] = []
        self.custom_types: list[str] = []
        self.final_finish_count = 2

    async def _finishes(self, info, count=2):
        for side in ('red', 'blue')[:count]:
            # A real delayed response, after the triggering request returns to
            # the event loop, catches the former immediate reader cancellation.
            await asyncio.sleep(.03)
            packet = {'Score': {'unit': {'Name': 'unit', 'SumWorth': 100}},
                      'Reward': 800, 'Events': [{'摧毁方': side, '被摧毁方': 'blue' if side == 'red' else 'red',
                                                '得分': '红方100' if side == 'red' else '蓝方100'}]}
            yield pb2.StreamMessage(info=info, finish=pb2.Finish(data=json.dumps(packet).encode()))

    def _properties(self) -> dict:
        self.episode += 1
        self.red_uid = self.episode * 100 + 10
        self.blue_uid = self.episode * 100 + 20
        return {
            "valid": True,
            "propertyArr": [
                {
                    "uId": self.red_uid,
                    "agentTeam": 0,
                    "type": "BP_BaseSoldier_C",
                    "className": "red",
                    "agentHp": 100,
                    "perceptionRange": 1000,
                    "initLocation": {"x": 0, "y": 0, "z": 0},
                },
                {
                    "uId": self.blue_uid,
                    "agentTeam": 1,
                    "type": "BP_BaseSoldier_C",
                    "className": "blue",
                    "agentHp": 100,
                    "perceptionRange": 1000,
                    "initLocation": {"x": 100, "y": 0, "z": 0},
                },
            ],
        }

    def _observation(self, *, done: bool) -> dict:
        entities = []
        for uid, team, x in (
            (self.red_uid, 0, 0),
            (self.blue_uid, 1, 100),
        ):
            entities.append({
                "valid": True,
                "uId": uid,
                "agentTeam": team,
                "agentAlive": True,
                "agentHp": 100,
                "agentLocation": {"x": x, "y": 0, "z": 0},
                "agentVelocity": {"x": 0, "y": 0, "z": 0},
                "agentRotation": {"pitch": 0, "yaw": 0, "roll": 0},
                "agentPerception": [],
                "availActions": [],
                "rSVD1": f"agent-{uid};1",
            })
        events = (
            ["<Event>EndEpisode<EndReason>FakeRound<WinTeam>0"]
            if done else []
        )
        return {
            "valid": True,
            "dataArr": entities,
            "dataGlobal": {
                "valid": True,
                "timeCnt": 0 if done else -1,
                "maxEpisodeStep": 5,
                "episodeDone": done,
                "episodeEndReason": "FakeRound" if done else "",
                "teamWin": 0 if done else -1,
                "events": events,
                "keyObjArr": [],
                "levelName": "FakeLevel",
                "rSVD1": "2;0",
            },
        }

    async def DataChannel(self, request_iterator, context):
        async for message in request_iterator:
            body = message.WhichOneof("body")
            if body == "handshake":
                yield pb2.StreamMessage(
                    info=message.info,
                    handshake=pb2.Handshake(protocol_version=0),
                )
            elif body == "prepare":
                type_name = message.prepare.type
                self.prepare_types.append(type_name)
                if type_name == "reset_property" and self.episode:
                    async for finish in self._finishes(message.info):
                        yield finish
                payload = (
                    self._properties()
                    if type_name == "reset_property"
                    else self._observation(done=False)
                )
                yield pb2.StreamMessage(
                    info=message.info,
                    prepare=pb2.Prepare(
                        type=type_name,
                        data=json.dumps(payload).encode("utf-8"),
                    ),
                )
            elif body == "frame":
                self.frame_requests.append(json.loads(message.frame.data.decode("utf-8")))
                yield pb2.StreamMessage(
                    info=message.info,
                    frame=pb2.Payload(
                        data=json.dumps(self._observation(done=True)).encode("utf-8"),
                        sleepTime=message.frame.sleepTime,
                    ),
                )
            elif body == "custom":
                self.custom_types.append(message.custom.type)
                if message.custom.type == 'end_unreal_engine':
                    async for finish in self._finishes(message.info, self.final_finish_count):
                        yield finish


@pytest.mark.parametrize('episode_limit', [None, 1])
def test_runner_completes_two_episode_grpc_lifecycle(tmp_path, episode_limit):
    async def scenario():
        service = _TwoEpisodeUeService()
        server = grpc.aio.server()
        pb2_grpc.add_CustomNetworkServiceServicer_to_server(service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        run_log = SimulationLog(tmp_path / "logs", detailed=True)
        config = RuntimeConfig(
            path=tmp_path / "task.jsonc",
            raw={},
            host="127.0.0.1",
            port=port,
            algorithm_names=(
                "TeamAlg.RedModel.Algorithm.RLAgentAlgorithm->ReinforceAgentAlgorithm",
                "TeamAlg.BlueModel.Algorithm.RLAgentAlgorithm->ReinforceAgentAlgorithm",
            ),
            max_steps=5,
            max_episodes=1,
            time_dilation=1.0,
            frame_rate=60.0,
            step_game_time=0.0,
            seed=1,
            device="cpu",
            log_level="DEBUG",
            connect_timeout=5.0,
            reply_timeout=2.0,
        )
        try:
            runner = Runner(config, run_log)
            runner.episode_limit = episode_limit
            await runner.run()
        finally:
            run_log.close()
            await server.stop(0)
        return service, run_log.stats_path

    service, stats_path = asyncio.run(scenario())

    expected = episode_limit or 2
    assert service.prepare_types == ["reset_property", "reset_obs"]*expected
    assert len(service.frame_requests) == expected
    assert service.frame_requests[0]["StringActions"] == [
        "ActionSet2::Idle;N/A;110", "ActionSet2::Idle;N/A;120",
    ]
    if expected == 2:
        assert service.frame_requests[1]["StringActions"] == [
            "ActionSet2::Idle;N/A;210", "ActionSet2::Idle;N/A;220",
        ]
    assert service.custom_types == ["end_unreal_engine"]

    records = [
        json.loads(line)
        for line in stats_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["episode"] for item in records if item.get("kind") == "reset"] == list(range(1,expected+1))
    assert [item["episode"] for item in records if item.get("done")] == list(range(1,expected+1))
    protocol = [json.loads(line) for line in stats_path.with_name('protocol.jsonl').read_text(encoding='utf-8').splitlines()]
    finishes = [row for row in protocol if row.get('channel') == 'finish']
    assert len(finishes) == 2*expected
    end = next(i for i, row in enumerate(protocol) if row.get('type') == 'end_unreal_engine')
    assert sum(row.get('channel') == 'finish' for row in protocol[end+1:]) == 2


def test_missing_last_score_fails_even_with_prior_round_pair(tmp_path, monkeypatch):
    from cssim.transport import GrpcClient
    from cssim.environment.env import CssimEnvironment

    original = GrpcClient.wait_for_finishes
    async def bounded(self, expected, timeout=10):
        return await original(self, expected, timeout=.2)
    monkeypatch.setattr(GrpcClient, 'wait_for_finishes', bounded)

    async def scenario():
        service = _TwoEpisodeUeService()
        service.final_finish_count = 1
        server = grpc.aio.server()
        pb2_grpc.add_CustomNetworkServiceServicer_to_server(service, server)
        port = server.add_insecure_port('127.0.0.1:0')
        await server.start()
        config = RuntimeConfig(tmp_path / 'task', {}, '127.0.0.1', port, (),
                               5, 2, 1., 60., .5, 1, 'cpu', 'DEBUG')
        environment = CssimEnvironment(config)
        try:
            await environment.connect()
            await environment.reset()
            await environment.reset()  # Drains the first episode's pair.
            with pytest.raises(TimeoutError, match='3/4'):
                await environment.close(completed=True)
            assert environment.client._call is None
        finally:
            await environment.client.close()
            await server.stop(0)
    asyncio.run(scenario())
