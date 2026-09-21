from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cssim.algorithms import (
    ActionSetDefinition,
    AlgorithmContext,
    ReinforcementAlgorithm,
    RuleAlgorithm,
)
from cssim.algorithms.base import EnvironmentTransition
from AlgTemplate.red.Algorithm.DQN import (
    DQNActionSet as RedDQNActionSet,
    DQNAgentAlgorithm as RedDQNAgentAlgorithm,
    DQNAlgorithm,
    DQNObservation as RedDQNObservation,
    MultiObjectivePlan as RedDQNObjectivePlan,
)
from AlgTemplate.blue.Algorithm.DQN import (
    DQNActionSet as BlueDQNActionSet,
    DQNAgentAlgorithm as BlueDQNAgentAlgorithm,
    MultiObjectivePlan as BlueDQNObjectivePlan,
)
from AlgTemplate.red.Algorithm.RuleAgentAlgorithm import (
    ReinforceAgentAlgorithm as RedRuleAlgorithm,
)
from AlgTemplate.blue.Algorithm.RuleAgentAlgorithm import (
    ReinforceAgentAlgorithm as BlueRuleAlgorithm,
)
from AlgTemplate.red.Algorithm.RLAgentAlgorithm import (
    ReinforceAgentAlgorithm as RedTemplateEntry,
)
from AlgTemplate.blue.Algorithm.RLAgentAlgorithm import (
    ReinforceAgentAlgorithm as BlueTemplateEntry,
)
from cssim.config import RuntimeConfig
from cssim.environment.env import CssimEnvironment
from cssim.environment.models import Entity, KeyObject, SimulationState, TeamState
from cssim.environment.observation import PerceptionModel
from cssim.logging_setup import SimulationLog
from cssim.protocol import (
    ALL_MOVE_DIRECTIONS,
    Action,
    ActionSetFactory,
    ChangeParentAction,
    DiscreteAction,
    RLActionSet,
    SUPPORTED_ACTION_COMMANDS,
)
from cssim.runtime.loader import AlgorithmLoader
from cssim.runtime.runner import Runner
from cssim.runtime.special_commands import SpecialCommandExecutor
from cssim.transport import GrpcClient
from cssim.transport.grpc_client import interface_pb2
from tools.analyze_protocol import ProtocolAnalyzer


class _Entity:
    def __init__(self, uid, team, team_index, entity_type="BP_BaseSoldier_C"):
        self.uid = uid
        self.team = team
        self.team_index = team_index
        self.entity_type = entity_type


def test_release_action_set_has_stable_fixed_entries():
    action_set = ActionSetFactory.standard(3)
    assert action_set.name == "cssim_standard_v1"
    assert action_set.fixed_count == 45
    assert action_set.n == 48
    assert action_set.decode(0).frame_action.command == "Idle"
    assert tuple(action_set.decode(index + 1).frame_action.direction for index in range(26)) == (
        ALL_MOVE_DIRECTIONS
    )


def test_actions_render_for_ue():
    roster = [_Entity(10, 0, 0), _Entity(20, 1, 0)]
    assert Action.move_to((1, 2, 3)).to_ue_string(10, roster) == "ActionSet2::Moving;;Points=[1.0, 2.0, 3.0];10"
    assert Action.guard_at((1, 2, 3)).to_ue_string(10, roster) == "ActionSet2::Guard;;Points=[1.0, 2.0, 3.0];10"
    assert Action.attack(roster[1]).to_ue_string(10, roster) == "ActionSet2::NormalAttacking;20;10"


def test_manual_rule_actions_render_for_ue():
    soldier = _Entity(10, 0, 0)
    uav = _Entity(11, 0, 1, "BP_Base_UAV_C")
    roster = [soldier, uav]
    assert len(ALL_MOVE_DIRECTIONS) == 26
    assert Action.move("+X-Y-Z").to_ue_string(10, roster) == "ActionSet2::Moving;+X-Y-Z;10"
    assert Action.guard("-X+Y").to_ue_string(10, roster) == "ActionSet2::Guard;-X+Y;10"
    assert Action.move_at((1, 2, 3)).to_ue_string(10, roster) == (
        "ActionSet2::Moving;;Points=[1.0, 2.0, 3.0];10"
    )
    assert Action.guard_position((1, 2, 3)).to_ue_string(10, roster) == (
        "ActionSet2::Guard;;Points=[1.0, 2.0, 3.0];10"
    )
    assert Action.grenade("+X+Y").to_ue_string(10, roster) == "ActionSet2::GrenadeAttacking;+X+Y;10"
    assert Action.plant_mine().to_ue_string(10, roster) == "ActionSet2::PlantMine;N/A;10"
    assert Action.detonate_mine().to_ue_string(10, roster) == "ActionSet2::DetonateMine;N/A;10"
    assert Action.self_destruct((1, 2, 3)).to_ue_string(11, roster) == (
        "ActionSet2::SelfDestruct;;Points=[1.0, 2.0, 3.0];11"
    )
    assert Action.board(20).to_ue_string(10, roster) == (
        "ActionSet2::Board;;Uid=20;10"
    )


def test_framework_does_not_filter_actions_by_entity_type():
    roster = [_Entity(10, 0, 0)]
    assert Action.self_destruct((1, 2, 3)).to_ue_string(10, roster) == (
        "ActionSet2::SelfDestruct;;Points=[1.0, 2.0, 3.0];10"
    )


def test_change_parent_action_repeats_and_validates_each_selection():
    def entity(uid, team, entity_type, x):
        item = Entity.from_property({
            "uId": uid, "agentTeam": team, "type": entity_type,
            "agentHp": 100, "initLocation": {"x": x, "y": 0, "z": 0},
        }, uid)
        item.raw = {"agentAlive": True, "agentHp": 100}
        return item

    child = entity(10, 0, "BP_Base_UAV_C", 0)
    near_soldier = entity(11, 0, "BP_BaseSoldier_C", 100)
    far_soldier = entity(12, 0, "BP_BaseSoldier_C", 500)
    enemy = entity(20, 1, "BP_BaseSoldier_C", 50)
    view = TeamState(
        team=0,
        action_mask=np.ones((3, 46), dtype=bool),
        agents=(child, near_soldier, far_soldier), visible_opponents=(enemy,),
        raw={}, episode=1, step=0, training=False,
    )
    assert view.nearest_friendly(child, "BP_BaseSoldier_C").uid == 11
    command = ChangeParentAction(near_soldier, child)
    assert (command.parent_uid, command.child_uid) == (11, 10)
    child.alive = False
    with pytest.raises(ValueError, match="child 必须是存活实体"):
        ChangeParentAction(near_soldier, child)
    child.alive = True

    class FakeClient:
        def __init__(self):
            self.calls = []

        async def change_actor_parent(self, parent_uid, child_uid):
            self.calls.append((parent_uid, child_uid))
            return {"success": True, "text": "ok"}

    async def exercise():
        client = FakeClient()
        executor = SpecialCommandExecutor(client)
        state = SimulationState(
            entities=(child, near_soldier, far_soldier, enemy), raw={},
            episode=1, step=0, training=False,
        )
        first = await executor.execute(0, state, (command,))
        child.raw["indexInTeam"] = near_soldier.uid
        repeated = await executor.execute(0, state, (command,))
        assert client.calls == [(11, 10), (11, 10)]
        assert first[0].success
        assert repeated[0].success
        child.raw["indexInTeam"] = far_soldier.uid
        await executor.execute(0, state, (command,))
        assert client.calls == [(11, 10), (11, 10), (11, 10)]
        with pytest.raises(ValueError, match="child 必须是本方"):
            await executor.execute(0, state, (ChangeParentAction(child, enemy),))

    asyncio.run(exercise())


def test_rule_parent_action_consumes_current_agent_action_slot():
    child = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_Base_UAV_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    parent = Entity.from_property({
        "uId": 11, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 100, "y": 0, "z": 0},
    }, 1)
    child.raw = {"agentAlive": True}
    parent.raw = {"agentAlive": True}
    state = TeamState(
        team=0,
        action_mask=np.ones((2, 45), dtype=bool), agents=(child, parent),
        visible_opponents=(), raw={}, episode=1, step=0, training=False,
    )

    class ParentRule(RuleAlgorithm):
        def decide(self, current):
            return (ChangeParentAction(parent, child), Action.move("+X"))

    algorithm = ParentRule(AlgorithmContext(
        0, state.agents, (), ActionSetFactory.standard(), "cpu", {}
    ))
    actions = algorithm.act(state)
    assert [item.command for item in actions] == ["Idle", "Moving"]
    assert algorithm.take_selected_special_commands() == (
        ChangeParentAction(parent, child),
    )


def test_rl_parent_slot_returns_action_object_and_sends_idle_frame():
    def entity(uid, team, entity_type, x):
        item = Entity.from_property({
            "uId": uid, "agentTeam": team, "type": entity_type,
            "agentHp": 100, "initLocation": {"x": x, "y": 0, "z": 0},
        }, uid)
        item.raw = {"agentAlive": True, "agentHp": 100}
        return item

    child = entity(10, 0, "BP_Base_UAV_C", 0)
    soldier = entity(11, 0, "BP_BaseSoldier_C", 100)
    standard = ActionSetFactory.standard(0)
    action_set = RLActionSet(
        "parent_test",
        1,
        (*standard.choices, DiscreteAction.dynamic(
            "change_parent:nearest_alive_parent",
            lambda state, current: ChangeParentAction(
                state.nearest_available_parent(current),
                current,
            ),
            available=lambda state, current: (
                state.nearest_available_parent(current) is not None
            ),
        )),
        include_target_attacks=True,
    )
    state = TeamState(
        team=0,
        action_mask=np.zeros((2, action_set.n), dtype=bool),
        agents=(child, soldier), visible_opponents=(), raw={},
        episode=1, step=0, training=False,
    )
    state = TeamState(
        **{**state.__dict__, "action_mask": action_set.action_mask(state)}
    )

    class ParentSelectingAlgorithm(ReinforcementAlgorithm):
        def select_actions(self, features, action_mask, training):
            assert action_mask[0, 45]
            assert not action_mask[1, 45]
            return (45, 0)

    context = AlgorithmContext(0, state.agents, (), action_set, "cpu", {})
    algorithm = ParentSelectingAlgorithm(context)
    actions = algorithm.act(state)
    commands = algorithm.take_selected_special_commands()

    assert action_set.fixed_count == 46
    assert actions[0].command == "Idle"
    assert actions[1].command == "Idle"
    assert commands == (ChangeParentAction(11, 10),)
    assert algorithm.take_selected_special_commands() == ()


def test_change_actor_parent_rpc_sends_uid_strings():
    class FakeStub:
        request = None

        async def InterfaceCall(self, request, timeout):
            self.request = request
            response = interface_pb2.ResultMessage()
            response.interfaceresult.success = True
            response.interfaceresult.text = b"ok"
            return response

    async def exercise():
        records = []
        client = GrpcClient(
            "127.0.0.1:21051",
            protocol_writer=lambda *items: records.append(items),
        )
        stub = FakeStub()
        client._interface_stub = stub
        result = await client.change_actor_parent(11, 10)
        assert result == {"success": True, "text": "ok"}
        assert stub.request.changeactorparent.parent == "11"
        assert stub.request.changeactorparent.child == "10"
        assert [item[0] for item in records] == ["python_to_ue", "ue_to_python"]

    asyncio.run(exercise())


def test_cpp_unsupported_actions_cannot_be_constructed():
    for command in ("Patrol", "Scout", "Hover", "Ascend", "Descend", "Alert", "ChangeHeight"):
        with pytest.raises(ValueError, match="当前 UE C\\+\\+ 不支持动作"):
            Action(command)


def test_standard_and_custom_rl_action_sets():
    standard = ActionSetFactory.standard(3)
    assert standard.name == "cssim_standard_v1"
    assert standard.fixed_count == 45
    assert standard.n == 48
    assert standard.decode(0).frame_action.command == "Idle"
    assert standard.decode(1).frame_action.direction == ALL_MOVE_DIRECTIONS[0]

    custom = RLActionSet(
        name="custom_test",
        version=1,
        choices=(
            DiscreteAction.fixed("idle", Action.idle()),
            DiscreteAction.dynamic("move:agent-x", lambda _state, agent: Action.move_to(
                (agent.position[0] + 10, 0, 0)
            )),
        ),
        include_target_attacks=True,
        target_count=1,
    )
    agent = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 5, "y": 0, "z": 0},
    }, 0)
    opponent = Entity.from_property({
        "uId": 20, "agentTeam": 1, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 50, "y": 0, "z": 0},
    }, 0)
    state = SimpleNamespace(
        agents=(agent,), visible_opponents=(opponent,),
        perceived_opponents=lambda _agent: (opponent,),
    )
    assert custom.decode(1, state, 0).frame_action.points == (15.0, 0.0, 0.0)
    assert custom.decode(2, state, 0).frame_action.target_uid == 20
    assert custom.action_mask(state).tolist() == [[True, True, True]]
    assert custom.signature != standard.signature


def test_reinforcement_base_owns_custom_action_set_lifecycle():
    custom = RLActionSet(
        name="contestant_actions",
        version=1,
        choices=(DiscreteAction.fixed("idle", Action.idle()),),
        include_target_attacks=True,
    )

    class ContestantActions(ActionSetDefinition):
        def define(self, context):
            return custom.choices

    class ContestantAlgorithm(ReinforcementAlgorithm):
        action_definition = ContestantActions("contestant_actions")

        def select_actions(self, features, action_mask, training):
            return np.zeros(len(features.self_features), dtype=np.int64)

    context = AlgorithmContext(
        0, (), (), ActionSetFactory.standard(3), "cpu", {}
    )
    algorithm = ContestantAlgorithm(context)

    assert algorithm.context.action_set is not custom
    assert algorithm.context.action_set.name == "contestant_actions"
    assert algorithm.context.action_set.target_count == 3
    assert algorithm.context.action_set.n == 4


def test_reinforcement_fixed_interfaces_build_actions_observation_and_training_data():
    agent = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "className": "士兵_0", "agentHp": 100,
        "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)

    class ContestantActions(ActionSetDefinition):
        def define(self, context):
            return (DiscreteAction.fixed("idle", Action.idle()),)

    class ContestantAlgorithm(ReinforcementAlgorithm):
        action_definition = ContestantActions(
            "fixed_interface_example", include_target_attacks=False
        )

        def __init__(self, context):
            self.selected_observation = None
            self.received_transition = None
            super().__init__(context)

        def build_observation(self, state):
            return {"agent_count": len(state.agents), "step": state.step}

        def select_actions(self, observation, action_mask, training):
            self.selected_observation = observation
            return np.zeros(observation["agent_count"], dtype=np.int64)

        def train_on_transition(self, transition):
            self.received_transition = transition

    context = AlgorithmContext(0, (agent,), (), ActionSetFactory.standard(3), "cpu", {})
    algorithm = ContestantAlgorithm(context)
    state = TeamState(
        team=0,
        action_mask=np.ones((1, 1), dtype=bool),
        agents=(agent,),
        visible_opponents=(),
        raw={},
        episode=1,
        step=7,
        training=True,
    )

    assert algorithm.context.action_set.name == "fixed_interface_example"
    assert algorithm.context.action_set.action_keys == ("idle",)
    assert algorithm.act(state) == [Action.idle()]
    algorithm._process_transition(EnvironmentTransition(
        state, np.asarray([0]), np.asarray([1.0], dtype=np.float32), state, False,
    ))

    assert algorithm.selected_observation == {"agent_count": 1, "step": 7}
    assert algorithm.received_transition.observation == algorithm.selected_observation
    assert algorithm.received_transition.next_observation == algorithm.selected_observation
    assert algorithm.received_transition.next_action_mask.tolist() == [[True]]


def test_reinforcement_rejects_action_disabled_by_mask():
    agent = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    action_set = RLActionSet(
        "mask_validation",
        1,
        (
            DiscreteAction.fixed("idle", Action.idle()),
            DiscreteAction.fixed("move:+X", Action.move("+X")),
        ),
        include_target_attacks=False,
    )

    class InvalidSelection(ReinforcementAlgorithm):
        def select_actions(self, observation, action_mask, training):
            return (1,)

    algorithm = InvalidSelection(AlgorithmContext(
        0, (agent,), (), action_set, "cpu", {}
    ))
    state = TeamState(
        team=0,
        action_mask=np.asarray([[True, False]], dtype=bool),
        agents=(agent,),
        visible_opponents=(),
        raw={},
        episode=1,
        step=0,
        training=False,
    )

    with pytest.raises(ValueError, match="选择了当前不可用的动作"):
        algorithm.act(state)


def test_runner_cleanup_continues_after_algorithm_hook_failure():
    calls = []

    class FailingAlgorithm:
        mode = "test"
        context = SimpleNamespace(team=0)

        def on_simulation_end(self):
            calls.append("first_hook")
            raise RuntimeError("cleanup failed")

    class HealthyAlgorithm:
        mode = "test"
        context = SimpleNamespace(team=1)

        def on_simulation_end(self):
            calls.append("second_hook")

    class Environment:
        async def connect(self):
            raise RuntimeError("simulation failed")

        async def close(self):
            calls.append("environment_close")

    runner = object.__new__(Runner)
    runner.environment = Environment()
    runner.algorithms = [FailingAlgorithm(), HealthyAlgorithm()]

    with pytest.raises(RuntimeError, match="simulation failed"):
        asyncio.run(runner.run())

    assert calls == ["first_hook", "second_hook", "environment_close"]

def test_dqn_checkpoint_rejects_same_width_but_different_action_semantics(tmp_path):
    red = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    context = AlgorithmContext(0, (red,), (), ActionSetFactory.standard(0), "cpu", {})
    first = RLActionSet(
        "same_width_a", 1,
        (DiscreteAction.fixed("idle", Action.idle()),),
        include_target_attacks=False,
    )
    second = RLActionSet(
        "same_width_b", 1,
        (DiscreteAction.fixed("move:+X", Action.move("+X")),),
        include_target_attacks=False,
    )
    model_path = tmp_path / "model.pt"

    class StaticActions(ActionSetDefinition):
        def __init__(self, action_set):
            super().__init__(
                action_set.name,
                action_set.version,
                action_set.include_target_attacks,
            )
            object.__setattr__(self, "choices", action_set.choices)

        def define(self, context):
            return self.choices

    class CustomDQN(DQNAlgorithm):
        action_definition = StaticActions(first)

    writer = CustomDQN(context, hidden_size=16, model_path=model_path)
    writer.save_checkpoint()
    checkpoint = torch.load(model_path, map_location="cpu")
    assert checkpoint["action_set_name"] == "same_width_a"
    assert checkpoint["action_signature"] == first.signature

    CustomDQN.action_definition = StaticActions(second)
    try:
        CustomDQN(context, hidden_size=16, model_path=model_path, load_model=True)
    except ValueError as exc:
        assert "模型动作集合不兼容" in str(exc)
    else:
        raise AssertionError("同宽但语义不同的动作集合不应加载同一模型")


def test_dqn_rejects_checkpoint_without_action_metadata(tmp_path):
    red = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    context = AlgorithmContext(0, (red,), (), ActionSetFactory.standard(0), "cpu", {})
    source = DQNAlgorithm(context, hidden_size=16)
    path = tmp_path / "old.pt"
    torch.save({
        "online": source.online.state_dict(),
        "target": source.target.state_dict(),
        "decision_count": 12,
        "update_count": 7,
    }, path)

    try:
        DQNAlgorithm(context, hidden_size=16, model_path=path, load_model=True)
    except ValueError as exc:
        assert "缺少动作集合元数据" in str(exc)
    else:
        raise AssertionError("发布版不应静默加载旧 checkpoint")


def test_dqn_load_model_fails_when_checkpoint_is_missing(tmp_path):
    context = AlgorithmContext(0, (), (), ActionSetFactory.standard(), "cpu", {})
    with pytest.raises(FileNotFoundError, match="找不到DQN模型"):
        DQNAlgorithm(
            context,
            hidden_size=16,
            model_path=tmp_path / "missing.pt",
            load_model=True,
        )


def test_dqn_inference_is_greedy_and_never_trains_or_saves():
    agent = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    context = AlgorithmContext(
        0, (agent,), (), ActionSetFactory.standard(), "cpu", {}
    )
    algorithm = DQNAlgorithm(context, hidden_size=16, batch_size=1)
    state = TeamState(
        team=0,
        action_mask=np.ones((1, 45), dtype=bool),
        agents=(agent,),
        visible_opponents=(),
        raw={},
        episode=1,
        step=0,
        training=False,
    )
    algorithm.select_actions(
        RedDQNObservation.from_state(state), state.action_mask, training=False
    )
    algorithm._process_transition(EnvironmentTransition(
        state, np.asarray([0]), np.asarray([1.0], dtype=np.float32), state, False,
    ))
    save_calls = []
    algorithm.save_checkpoint = lambda: save_calls.append(True)
    algorithm.update_count = 1
    algorithm.on_episode_end(10)
    algorithm.on_simulation_end()

    assert algorithm.training_enabled is False
    assert algorithm.online.training is False
    assert len(algorithm.replay) == 0
    assert save_calls == []


def test_commanders_and_dynamic_world_objects_are_exposed_without_combat_actions():
    response = {
        "propertyArr": [
            {
                "uId": 501, "agentTeam": 0, "type": "BaseCommander_C",
                "className": "BaseCommander_C_0", "agentHp": 100,
                "maxMoveSpeed": 0, "initLocation": {"x": 0, "y": 0, "z": 0},
            },
            {
                "uId": 601, "agentTeam": 1, "type": "BaseCommander_C",
                "className": "BaseCommander_C_1", "agentHp": 100,
                "maxMoveSpeed": 0, "initLocation": {"x": 100, "y": 0, "z": 0},
            },
            {
                "uId": 11, "agentTeam": 0, "type": "BP_BaseSoldier_C",
                "agentHp": 100, "maxMoveSpeed": 1,
                "initLocation": {"x": 0, "y": 0, "z": 0},
            },
            {
                "uId": 21, "agentTeam": 1, "type": "BP_BaseSoldier_C",
                "agentHp": 100, "maxMoveSpeed": 1,
                "initLocation": {"x": 10, "y": 0, "z": 0},
            },
        ]
    }
    environment = object.__new__(CssimEnvironment)
    environment._build_roster(response)
    assert [item.uid for item in environment.entities] == [11, 21]
    assert [item.uid for item in environment.commanders] == [501, 601]

    red, blue = environment.entities
    red.raw = {"agentPerception": [blue.uid]}
    target_a = KeyObject.from_raw(
        {"uId": 32768, "className": "CommandPost_A", "team": 1, "location": {"x": 50, "y": 0, "z": 0}}
    )
    target_b = KeyObject.from_raw(
        {"uId": 32769, "className": "CommandPost_B", "team": 0, "location": {"x": 5, "y": 0, "z": 0}}
    )
    state = TeamState(
        team=0,
        action_mask=np.ones((1, 46), dtype=bool),
        agents=(red,),
        visible_opponents=(blue,),
        raw={},
        episode=1,
        step=0,
        training=False,
        commanders=(environment.commanders[0],),
        key_objects=(target_a, target_b),
    )
    assert state.entity_by_uid(601) is None
    assert state.perceived_opponents(red) == (blue,)
    assert state.nearest_key_object(red).uid == 32769
    assert state.nearest_key_object(red, team=1).uid == 32768
    assert Action.attack(blue).to_ue_string(red.uid, environment.entities) == "ActionSet2::NormalAttacking;21;11"
    assert Action.move_to(target_b).to_ue_string(red.uid, environment.entities) == (
        "ActionSet2::Moving;;Points=[5.0, 0.0, 0.0];11"
    )


def test_team_state_uses_frame_snapshot_and_keeps_unified_attack_available():
    def entity(uid, team, entity_type, x):
        return Entity.from_property(
            {
                "uId": uid, "agentTeam": team, "type": entity_type,
                "agentHp": 100, "maxMoveSpeed": 1,
                "initLocation": {"x": x, "y": 0, "z": 0},
            },
            0,
        )

    live_red = entity(11, 0, "BP_BaseSoldier_C", 999)
    live_blue = entity(21, 1, "BP_Base_UAV_C", 999)
    snapshot_red = entity(11, 0, "BP_BaseSoldier_C", 10)
    snapshot_blue = entity(21, 1, "BP_Base_UAV_C", 20)
    snapshot_red.raw = {"agentPerception": [snapshot_blue.uid]}
    environment = object.__new__(CssimEnvironment)
    environment.entities = (live_red, live_blue)
    environment.config = SimpleNamespace(max_steps=200)
    state = SimulationState(
        entities=(snapshot_red, snapshot_blue),
        raw={}, episode=1, step=0, training=True,
    )

    view = environment.build_team_state(state, 0)
    assert view.agents[0] is not snapshot_red
    assert view.agents[0].uid == snapshot_red.uid
    assert view.visible_opponents[0] is not snapshot_blue
    assert view.visible_opponents[0].uid == snapshot_blue.uid
    assert not view.visible_opponents[0].raw and not view.visible_opponents[0].properties
    assert view.agents[0].position[0] == 10
    assert view.action_mask.shape == (1, 0)


def test_dead_entity_clears_stale_hp():
    entity = Entity.from_property(
        {
            "uId": 11, "agentTeam": 0, "type": "BP_BaseSoldier_C",
            "agentHp": 100, "maxMoveSpeed": 1,
            "initLocation": {"x": 0, "y": 0, "z": 0},
        },
        0,
    )
    entity.update({"uId": 11, "agentAlive": False})
    assert not entity.alive
    assert entity.hp == 0


def test_new_episode_refreshes_dynamic_uids_without_changing_action_dimensions():
    def entity(uid, team):
        return Entity.from_property(
            {
                "uId": uid, "agentTeam": team, "type": "BP_BaseSoldier_C",
                "agentHp": 100, "maxMoveSpeed": 1, "perceptionRange": 1000,
                "initLocation": {"x": 0, "y": 0, "z": 0},
            },
            0,
        )

    old_red, old_blue = entity(11, 0), entity(21, 1)
    context = AlgorithmContext(
        0, (old_red,), (old_blue,), ActionSetFactory.standard(1), "cpu", {}
    )
    runner = object.__new__(Runner)
    runner.algorithms = [SimpleNamespace(context=context)]
    runner.environment = object.__new__(CssimEnvironment)
    runner.environment.config = SimpleNamespace(max_steps=200)
    state = SimulationState(
        entities=(entity(111, 0), entity(121, 1)),
        raw={},
        episode=2,
        step=0,
        training=True,
    )
    runner._synchronize_algorithm_contexts_for_new_episode(state)
    assert runner.algorithms[0].context.agents[0].uid == 111
    assert runner.algorithms[0].context.visible_opponents[0].uid == 121
    assert runner.algorithms[0].context.action_set.n == context.action_set.n


def test_jsonc_config(tmp_path, monkeypatch):
    monkeypatch.delenv("CSSIM_LOG_LEVEL", raising=False)
    path = tmp_path / "task.jsonc"
    path.write_text("""
    {
      // UE 生成的 TaskConfig 允许行注释和末尾逗号。
      "config.py->GlobalConfig": {"num_worker_threads": 1},
      "Common.cssim_env_wrapper.py->ScenarioConfig": {
        "Algorithm_Names": ["a->A", "b->B"],
        "TcpAddr": "127.0.0.1",
        "CSSIMPort": 21051,
      },
    }
    """, encoding="utf-8")
    config = RuntimeConfig.load(path)
    assert config.host == "127.0.0.1"
    assert config.port == 21051
    assert config.log_level == "DEBUG"


def test_fixed_entry_loads_rule_mode():
    red = Entity.from_property(
        {
            "uId": 10, "agentTeam": 0, "type": "soldier", "maxMoveSpeed": 1,
            "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
        },
        0,
    )
    blue = Entity.from_property(
        {
            "uId": 20, "agentTeam": 1, "type": "soldier", "maxMoveSpeed": 1,
            "agentHp": 100, "initLocation": {"x": 1, "y": 0, "z": 0},
        },
        0,
    )
    context = AlgorithmContext(0, (red,), (blue,), ActionSetFactory.standard(1), "cpu", {})
    algorithm = AlgorithmLoader({}).load(
        "AlgTemplate.red.Algorithm.RLAgentAlgorithm->ReinforceAgentAlgorithm",
        context,
    )
    assert isinstance(algorithm, RuleAlgorithm)


def test_fixed_entry_can_switch_to_dqn():
    red = Entity.from_property(
        {
            "uId": 10, "agentTeam": 0, "type": "soldier", "maxMoveSpeed": 1,
            "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
        },
        0,
    )
    blue = Entity.from_property(
        {
            "uId": 20, "agentTeam": 1, "type": "soldier", "maxMoveSpeed": 1,
            "agentHp": 100, "initLocation": {"x": 1, "y": 0, "z": 0},
        },
        0,
    )
    context = AlgorithmContext(0, (red,), (blue,), ActionSetFactory.standard(1), "cpu", {})
    path = "AlgTemplate.red.Algorithm.RLAgentAlgorithm"
    raw_config = {
        f"{path}.py->AlgorithmConfig": {
            "mode": "reinforcement_learning",
            "gamma": 0.5,
        }
    }
    algorithm = AlgorithmLoader(raw_config).load(
        f"{path}->ReinforceAgentAlgorithm",
        context,
    )
    state = TeamState(
        team=0,
        action_mask=np.ones((1, 48), dtype=bool),
        agents=(red,),
        visible_opponents=(blue,),
        raw={},
        episode=1,
        step=0,
        training=False,
    )
    state = replace(state, action_mask=algorithm.build_action_mask(state))
    assert isinstance(algorithm, DQNAlgorithm)
    assert algorithm.context.action_set.name == "red_minimal_control"
    assert algorithm.context.action_set.fixed_count == 8
    assert algorithm.context.action_set.n == 8
    assert algorithm.gamma == 0.5
    assert len(algorithm.act(state)) == 1

    # 当前实例的TaskConfig覆盖不能污染下一次算法加载。
    default_algorithm = AlgorithmLoader({}).load(
        f"{path}->ReinforceAgentAlgorithm", context,
    )
    assert isinstance(default_algorithm, RuleAlgorithm)


def test_ue_default_red_and_blue_models_can_be_loaded():
    for team, model_name in ((0, "RedModel"), (1, "BlueModel")):
        agent = Entity.from_property(
            {
                "uId": 10 + team,
                "agentTeam": team,
                "type": "BP_BaseSoldier_C",
                "maxMoveSpeed": 1,
                "agentHp": 100,
                "initLocation": {"x": 0, "y": 0, "z": 0},
            },
            0,
        )
        context = AlgorithmContext(
            team, (agent,), (), ActionSetFactory.standard(), "cpu", {}
        )
        module = f"TeamAlg.{model_name}.Algorithm.RLAgentAlgorithm"
        pointer = "AlgorithmConfig_Red" if team == 0 else "AlgorithmConfig_Blue"
        loader = AlgorithmLoader({
            "config.py->GlobalConfig": {
                pointer: f"{module}.py->AlgorithmConfig",
            }
        })
        algorithm = loader.load(
            f"{module}->ReinforceAgentAlgorithm",
            context,
        )
        state = TeamState(
            team=team,
            action_mask=np.ones((1, 45), dtype=bool),
            agents=(agent,),
            visible_opponents=(),
            raw={},
            episode=1,
            step=0,
            training=False,
        )
        actions = list(algorithm.act(state))
        assert algorithm.mode == "rule"
        assert [item.command for item in actions] == ["Idle"]


def test_team_state_keeps_all_objectives_and_current_commander():
    red = Entity.from_property(
        {
            "uId": 10, "agentTeam": 0, "type": "BP_BaseSoldier_C",
            "agentHp": 100, "maxMoveSpeed": 600, "customSensorRange": 4321,
            "fireRange": 1000, "initLocation": {"x": 0, "y": 0, "z": 0},
        },
        0,
    )
    blue = Entity.from_property(
        {
            "uId": 20, "agentTeam": 1, "type": "FutureAgentType",
            "agentHp": 200, "maxMoveSpeed": 300,
            "initLocation": {"x": 100, "y": 0, "z": 0},
        },
        0,
    )
    red.update({
        "uId": 10, "agentAlive": True, "agentHp": 90,
        "agentLocation": {"x": 1, "y": 2, "z": 3},
        "agentVelocity": {"x": 0, "y": 0, "z": 0},
        "agentPerception": [20],
        "indexInTeam": 501, "rSVD1": "士兵_0;1",
    })
    commander = Entity.from_property({
        "uId": 501, "agentTeam": 0, "type": "BaseCommander_C",
        "agentHp": 100, "initLocation": {"x": 10, "y": 0, "z": 0},
    }, 0)
    objectives = tuple(
        KeyObject.from_raw({
            "uId": 32768 + index,
            "team": 1,
            "valid": index != 2,
            "location": {"x": index * 100, "y": 50, "z": 0},
        })
        for index in range(3)
    )
    state = TeamState(
        team=0, action_mask=np.ones((1, 1), dtype=bool), agents=(red,),
        visible_opponents=(blue,), raw={}, episode=1, step=5, training=False,
        commanders=(commander,), key_objects=objectives,
    )
    assert len(state.key_objects) == 3
    assert state.current_commander(red) is commander
    assert state.key_objects[2].valid is False
    assert red.properties["customSensorRange"] == 4321
    assert "customSensorRange" not in red.raw


def test_current_commander_can_be_an_ordinary_agent():
    leader = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 100, "y": 0, "z": 0},
    }, 0)
    follower = Entity.from_property({
        "uId": 11, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 1)
    follower.update({
        "uId": 11, "agentAlive": True, "agentHp": 100,
        "agentLocation": {"x": 0, "y": 0, "z": 0},
        "agentVelocity": {"x": 0, "y": 0, "z": 0},
        "indexInTeam": 10, "rSVD1": "士兵_1;1",
    })
    agents = (leader, follower)
    state = TeamState(
        team=0, action_mask=np.ones((2, 18), dtype=bool),
        agents=agents, visible_opponents=(), raw={}, episode=1, step=0, training=True,
    )

    assert state.current_commander(follower) is leader


def test_nearest_available_parent_uses_alive_soldier_or_initial_commander():
    def entity(uid, entity_type, x, index):
        return Entity.from_property({
            "uId": uid, "agentTeam": 0, "type": entity_type,
            "agentHp": 100, "initLocation": {"x": x, "y": 0, "z": 0},
        }, index)

    child = entity(10, "BP_Base_UAV_C", 0, 0)
    dead_soldier = entity(11, "BP_BaseSoldier_C", 10, 1)
    alive_soldier = entity(12, "BP_BaseSoldier_C", 100, 2)
    commander = entity(501, "BaseCommander_C", 50, 0)
    dead_soldier.alive = False
    agents = (child, dead_soldier, alive_soldier)
    state = TeamState(
        team=0,
        action_mask=np.ones((3, 45), dtype=bool),
        agents=agents,
        visible_opponents=(),
        raw={},
        episode=1,
        step=0,
        training=False,
        commanders=(commander,),
    )

    # 最近士兵已经阵亡，因此先回退到距离50的初始Commander。
    assert state.nearest_available_parent(child) is commander
    alive_soldier.position[0] = 20
    assert state.nearest_available_parent(child) is alive_soldier


def test_dqn_change_parent_remains_available_for_an_existing_parent():
    parent = Entity.from_property({
        "uId": 11, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    child = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_RoboDog_C",
        "agentHp": 100, "indexInTeam": 11,
        "initLocation": {"x": 10, "y": 0, "z": 0},
    }, 1)
    state = TeamState(
        team=0,
        action_mask=np.ones((2, 8), dtype=bool),
        agents=(child, parent),
        visible_opponents=(),
        raw={},
        episode=1,
        step=0,
        training=False,
    )
    context = AlgorithmContext(
        0, state.agents, (), ActionSetFactory.standard(), "cpu", {}
    )

    assert RedDQNActionSet().build(context).action_mask(state)[0, 7]
    assert BlueDQNActionSet().build(context).action_mask(state)[0, 7]


def test_dqn_accepts_variable_opponents_without_changing_action_order():
    def entity(uid, team, index):
        return Entity.from_property({
            "uId": uid, "agentTeam": team, "type": "BP_BaseSoldier_C",
            "agentHp": 100, "maxMoveSpeed": 600,
            "initLocation": {"x": uid, "y": 0, "z": 0},
        }, index)

    red = entity(10, 0, 0)
    two = (entity(20, 1, 0), entity(21, 1, 1))
    base_context = AlgorithmContext(
        0, (red,), two, ActionSetFactory.standard(2), "cpu", {}
    )
    context = replace(
        base_context, action_set=RedDQNActionSet().build(base_context)
    )
    algorithm = DQNAlgorithm(context, hidden_size=16, batch_size=2)
    for opponents in (two, (*two, entity(22, 1, 2))):
        mask = np.ones((1, 8), dtype=bool)
        state = TeamState(
            team=0, action_mask=mask, agents=(red,),
            visible_opponents=opponents, raw={}, episode=1, step=0,
            training=False,
        )
        actions = algorithm.select_actions(
            RedDQNObservation.from_state(state), mask, training=False
        )
        assert actions.shape == (1,)
        assert 0 <= actions[0] < 8


def test_dqn_learns_from_structured_feature_replay():
    red = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    blue = Entity.from_property({
        "uId": 20, "agentTeam": 1, "type": "BP_RoboDog_C",
        "agentHp": 100, "initLocation": {"x": 100, "y": 0, "z": 0},
    }, 0)
    objectives = tuple(
        KeyObject.from_raw({
            "uId": 32768 + index,
            "location": {"x": 1000 * index, "y": 0, "z": 0},
        })
        for index in range(2)
    )
    context = AlgorithmContext(0, (red,), (blue,), ActionSetFactory.standard(1), "cpu", {})
    algorithm = DQNAlgorithm(context, hidden_size=16, batch_size=2)
    state = TeamState(
        team=0, action_mask=np.ones((1, 46), dtype=bool),
        agents=(red,), visible_opponents=(blue,), raw={}, episode=1, step=0, training=True,
        key_objects=objectives,
    )
    transition = EnvironmentTransition(
        state=state,
        actions=np.asarray([0]),
        reward=np.asarray([0.1], dtype=np.float32),
        next_state=state,
        done=False,
    )
    algorithm._process_transition(transition)
    algorithm._process_transition(transition)
    assert algorithm.update_count == 1
    metrics = algorithm.training_metrics()
    assert metrics["replay_size"] == 2
    assert metrics["update_count"] == 1
    assert np.isfinite(metrics["loss"])
    assert algorithm.replay[0].observation["self"].dtype == np.float32
    assert algorithm.replay[0].next_observation["global"].dtype == np.float32


def test_templates_distribute_agents_across_multiple_command_posts():
    agents = tuple(
        Entity.from_property({
            "uId": 10 + index,
            "agentTeam": 0,
            "type": "BP_BaseSoldier_C",
            "agentHp": 100,
            "initLocation": {"x": 0, "y": 0, "z": 0},
        }, index)
        for index in range(4)
    )
    objectives = (
        KeyObject.from_raw({
            "uId": 32769, "team": 1,
            "location": {"x": 2000, "y": 0, "z": 0},
        }),
        KeyObject.from_raw({
            "uId": 32768, "team": 1,
            "location": {"x": 1000, "y": 0, "z": 0},
        }),
    )
    state = TeamState(
        team=0,
        action_mask=np.ones((4, 47), dtype=bool),
        agents=agents,
        visible_opponents=(),
        raw={},
        episode=1,
        step=0,
        training=False,
        key_objects=objectives,
    )
    assert [RedDQNObjectivePlan.target(state, item).uid for item in agents] == [
        32768, 32769, 32768, 32769,
    ]
    assert [BlueDQNObjectivePlan.target(state, item).uid for item in agents] == [
        32768, 32769, 32768, 32769,
    ]
    observation = RedDQNObservation.from_state(state)
    assert observation.objective.shape == (4, 5)
    assert observation.objective[0].tolist() == pytest.approx(
        [1.0, 1000.0, 0.0, 0.0, 1000.0]
    )
    assert observation.objective[1].tolist() == pytest.approx(
        [1.0, 2000.0, 0.0, 0.0, 2000.0]
    )

    context = AlgorithmContext(0, agents, (), ActionSetFactory.standard(0), "cpu", {})
    red_actions = RedDQNActionSet().build(context)
    blue_actions = BlueDQNActionSet().build(context)
    assert red_actions.decode(5, state, 0).frame_action.points == (1000.0, 0.0, 0.0)
    assert red_actions.decode(5, state, 1).frame_action.points == (2000.0, 0.0, 0.0)
    assert blue_actions.decode(5, state, 0).frame_action.points == (1000.0, 0.0, 0.0)
    assert blue_actions.decode(5, state, 1).frame_action.points == (2000.0, 0.0, 0.0)

    red_rule = RedRuleAlgorithm(context)
    blue_rule = BlueRuleAlgorithm(context)
    assert red_rule.formation_point(state, agents[0])[0] == pytest.approx(1350.0)
    assert red_rule.formation_point(state, agents[2])[0] == pytest.approx(650.0)
    assert blue_rule.defense_point(state, agents[0])[0] == pytest.approx(1700.0)
    assert blue_rule.defense_point(state, agents[2])[0] == pytest.approx(0.0)


def test_dqn_self_observation_accepts_unknown_entity_types_by_capability():
    agent = Entity.from_property({
        "uId": 10,
        "agentTeam": 0,
        "type": "BP_Future_Unlisted_Unit_C",
        "agentHp": 200,
        "maxMoveSpeed": 400,
        "weaponCD": 0.25,
        "fireRange": 10000,
        "guardRange": 5000,
        "perceptionRange": 20000,
        "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    agent.raw = {
        "uId": 10,
        "agentAlive": True,
        "agentHp": 100,
        "agentVelocity": {"x": 2000, "y": -1000, "z": 500},
        "weaponCD": 0.5,
        "agentPerception": [],
    }
    agent.update(agent.raw)
    state = TeamState(
        team=0,
        action_mask=np.ones((1, 8), dtype=bool),
        agents=(agent,),
        visible_opponents=(),
        raw={},
        episode=1,
        step=0,
        training=True,
    )

    observation = RedDQNObservation.from_state(state)

    assert observation.self_values.shape == (1, 10)
    assert observation.self_values[0].tolist() == pytest.approx([
        1.0, 100.0, 2000.0, -1000.0, 500.0, 400.0,
        0.5, 10000.0, 5000.0, 20000.0,
    ])


def test_dqn_does_not_filter_dead_agent_action():
    alive = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    dead = Entity.from_property({
        "uId": 11, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 1)
    dead.alive = False
    dead.hp = 0
    opponent = Entity.from_property({
        "uId": 20, "agentTeam": 1, "type": "BP_RoboDog_C",
        "agentHp": 100, "initLocation": {"x": 100, "y": 0, "z": 0},
    }, 0)
    agents = (alive, dead)
    context = AlgorithmContext(0, agents, (opponent,), ActionSetFactory.standard(1), "cpu", {})
    algorithm = DQNAlgorithm(context, hidden_size=16, batch_size=128)
    mask = np.ones((2, 46), dtype=bool)
    state = TeamState(
        team=0,
        action_mask=mask, agents=agents, visible_opponents=(opponent,), raw={},
        episode=1, step=0, training=True,
    )

    actions = algorithm.act(state)

    assert isinstance(actions[1], Action)
    assert algorithm.last_action_indices[1] >= 0
    assert algorithm.decision_count == 1


def test_dqn_epsilon_advances_once_per_environment_decision():
    agents = tuple(
        Entity.from_property({
            "uId": 10 + index, "agentTeam": 0, "type": "BP_BaseSoldier_C",
            "agentHp": 100, "initLocation": {"x": index, "y": 0, "z": 0},
        }, index)
        for index in range(2)
    )
    opponent = Entity.from_property({
        "uId": 20, "agentTeam": 1, "type": "BP_RoboDog_C",
        "agentHp": 100, "initLocation": {"x": 100, "y": 0, "z": 0},
    }, 0)
    algorithm = DQNAlgorithm(
        AlgorithmContext(0, agents, (opponent,), ActionSetFactory.standard(1), "cpu", {}),
        hidden_size=16,
        batch_size=128,
    )
    state = TeamState(
        team=0,
        action_mask=np.ones((2, 46), dtype=bool),
        agents=agents,
        visible_opponents=(opponent,),
        raw={}, episode=1, step=0, training=True,
    )

    algorithm.act(state)

    assert algorithm.decision_count == 1


def test_environment_does_not_filter_dead_agent_actions():
    red = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    red.alive = False
    red.hp = 0
    blue = Entity.from_property({
        "uId": 20, "agentTeam": 1, "type": "BP_RoboDog_C",
        "agentHp": 100, "initLocation": {"x": 100, "y": 0, "z": 0},
    }, 0)
    environment = object.__new__(CssimEnvironment)
    environment.config = SimpleNamespace(max_steps=200)
    state = SimulationState(
        entities=(red, blue), raw={}, episode=1, step=10, training=True,
    )

    view = environment.build_team_state(state, 0)

    assert view.action_mask.shape == (1, 0)


def test_environment_step_uses_task_timing_and_reply_timeout():
    red = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    blue = Entity.from_property({
        "uId": 20, "agentTeam": 1, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 1, "y": 0, "z": 0},
    }, 0)

    class Client:
        call = None

        async def frame(self, payload, sleep_time, timeout):
            self.call = (json.loads(payload), sleep_time, timeout)
            return json.dumps({
                "valid": True,
                "dataArr": [
                    {"uId": 10, "agentAlive": True, "agentHp": 100,
                     "agentLocation": {"x": 0, "y": 0, "z": 0}},
                    {"uId": 20, "agentAlive": True, "agentHp": 100,
                     "agentLocation": {"x": 1, "y": 0, "z": 0}},
                ],
                "dataGlobal": {"timeCnt": 1, "episodeDone": False, "events": []},
            })

    environment = object.__new__(CssimEnvironment)
    environment.config = SimpleNamespace(
        max_steps=200, step_game_time=0.5, reply_timeout=7.0
    )
    environment.client = Client()
    environment.entities = (red, blue)
    environment.commanders = ()
    environment.episode = 1
    environment.training = False

    result = asyncio.run(environment.step((
        Action.move("+X"),
        Action.idle(),
    ), 0))

    assert result.state.step == 1
    assert environment.client.call[1:] == (0.5, 7.0)
    assert environment.client.call[0]["StringActions"] == [
        "ActionSet2::Moving;+X;10",
        "ActionSet2::Idle;N/A;20",
    ]


def test_environment_max_step_fallback_uses_zero_based_time_count():
    red = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    blue = Entity.from_property({
        "uId": 20, "agentTeam": 1, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 1, "y": 0, "z": 0},
    }, 0)

    class Client:
        async def frame(self, payload, sleep_time, timeout):
            return json.dumps({
                "valid": True,
                "dataArr": [
                    {"uId": 10, "agentAlive": True, "agentHp": 100},
                    {"uId": 20, "agentAlive": True, "agentHp": 100},
                ],
                "dataGlobal": {"timeCnt": 1, "episodeDone": False, "events": []},
            })

    environment = object.__new__(CssimEnvironment)
    environment.config = SimpleNamespace(
        max_steps=2, step_game_time=0.0, reply_timeout=1.0
    )
    environment.client = Client()
    environment.entities = (red, blue)
    environment.commanders = ()
    environment.episode = 1
    environment.training = False

    result = asyncio.run(environment.step((Action.idle(), Action.idle()), 0))
    assert result.done


def test_dqn_skips_old_dead_agent_and_terminates_death_transition():
    def entity(uid, alive):
        item = Entity.from_property({
            "uId": uid, "agentTeam": 0, "type": "BP_BaseSoldier_C",
            "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
        }, uid)
        item.alive = alive
        item.hp = 100 if alive else 0
        return item

    opponent = Entity.from_property({
        "uId": 20, "agentTeam": 1, "type": "BP_RoboDog_C",
        "agentHp": 100, "initLocation": {"x": 100, "y": 0, "z": 0},
    }, 0)
    before_agents = (entity(10, True), entity(11, False))
    after_agents = (entity(10, False), entity(11, False))
    context = AlgorithmContext(
        0, before_agents, (opponent,), ActionSetFactory.standard(1), "cpu", {}
    )
    algorithm = DQNAlgorithm(context, hidden_size=16, batch_size=1)

    def state(agents, step):
        mask = np.ones((2, 46), dtype=bool)
        for row, agent in enumerate(agents):
            if not agent.alive:
                mask[row] = False
                mask[row, 0] = True
        return TeamState(
            team=0,
            action_mask=mask, agents=agents, visible_opponents=(opponent,), raw={},
            episode=1, step=step, training=True,
        )

    transition = EnvironmentTransition(
        state(before_agents, 0), np.asarray([0, 0]),
        np.asarray([0.1, 0.0], dtype=np.float32), state(after_agents, 1), False,
    )
    algorithm._process_transition(transition)

    assert len(algorithm.replay) == 1
    assert algorithm.replay[0].done is True
    assert algorithm.update_count == 1
    assert np.isfinite(algorithm.last_loss)


def test_dqn_progress_reward_requires_same_objective_uid():
    old_agent = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_RoboDog_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    new_agent = replace(old_agent, position=old_agent.position.copy())
    old_objective = KeyObject.from_raw({
        "uId": 32768, "team": 1,
        "location": {"x": 1000, "y": 0, "z": 0},
    })
    new_objective = KeyObject.from_raw({
        "uId": 32769, "team": 1,
        "location": {"x": 0, "y": 0, "z": 0},
    })

    def state(agent, objective, step):
        return TeamState(
            team=0,
            action_mask=np.ones((1, 8), dtype=bool), agents=(agent,),
            visible_opponents=(), raw={}, episode=1, step=step, training=True,
            key_objects=(objective,),
        )

    previous = state(old_agent, old_objective, 0)
    current = state(new_agent, new_objective, 1)
    red = object.__new__(RedDQNAgentAlgorithm)
    blue = object.__new__(BlueDQNAgentAlgorithm)
    assert red.calculate_reward(previous, current, ()) == pytest.approx([0.0])
    assert blue.calculate_reward(previous, current, ()) == pytest.approx([0.0])


def test_each_agent_only_receives_its_own_local_situation():
    def entity(uid, team, index, x):
        item = Entity.from_property({
            "uId": uid, "agentTeam": team, "type": "BP_BaseSoldier_C",
            "agentHp": 100, "perceptionRange": 1000,
            "initLocation": {"x": x, "y": 0, "z": 0},
        }, index)
        item.raw = {
            "uId": uid, "agentAlive": True, "agentHp": 100,
            "agentLocation": {"x": x, "y": 0, "z": 0},
            "agentVelocity": {"x": 0, "y": 0, "z": 0},
            "agentPerception": [],
        }
        return item

    red_a, red_b = entity(10, 0, 0, 0), entity(11, 0, 1, 100)
    blue_a, blue_b = entity(20, 1, 0, 200), entity(21, 1, 1, 300)
    red_a.raw["agentPerception"] = [20]
    red_b.raw["agentPerception"] = [21]
    entities = (red_a, red_b, blue_a, blue_b)

    environment = object.__new__(CssimEnvironment)
    environment.entities = entities
    environment.commanders = ()
    environment.config = SimpleNamespace(max_steps=200)
    state = SimulationState(
        entities=entities, raw={}, episode=1, step=0, training=False,
    )
    view = environment.build_team_state(state, 0)
    assert not hasattr(view, "features")
    assert {item.uid for item in view.visible_opponents} == {20, 21}
    assert {item.uid for item in view.perceived_opponents(view.agents[0])} == {20}
    assert {item.uid for item in view.perceived_opponents(view.agents[1])} == {21}
    assert all(not item.raw and not item.properties for item in view.visible_opponents)
    action_set = ActionSetFactory.standard(1)
    mask = action_set.action_mask(view)
    assert mask[:, 45:].tolist() == [[True], [True]]
    assert action_set.decode(45, view, 0).frame_action.target_uid == 20
    assert action_set.decode(45, view, 1).frame_action.target_uid == 21


def test_team_view_exposes_shared_contacts_but_hides_enemy_internal_fields():
    def entity(uid, team, index):
        return Entity.from_property({
            "uId": uid, "agentTeam": team, "type": "BP_BaseSoldier_C",
            "agentHp": 100, "fireRange": 1234,
            "initLocation": {"x": uid, "y": 0, "z": 0},
        }, index)

    red = entity(10, 0, 0)
    visible_blue = entity(20, 1, 0)
    hidden_blue = entity(21, 1, 1)
    red.raw = {"agentPerception": [20], "availActions": [20, 21]}
    red_commander = entity(500, 0, 0)
    blue_commander = entity(600, 1, 0)
    environment = object.__new__(CssimEnvironment)
    environment.config = SimpleNamespace(max_steps=200)
    state = SimulationState(
        entities=(red, visible_blue, hidden_blue),
        raw={
            "dataGlobal": {
                "timeCnt": 3,
                "events": ["secret"],
                "hiddenEnemyCount": 2,
            }
        },
        episode=1,
        step=3,
        training=True,
        commanders=(red_commander, blue_commander),
    )

    view = environment.build_team_state(state, 0)

    assert [item.uid for item in view.visible_opponents] == [20, 21]
    assert view.visible_opponents[0].properties == {}
    assert view.visible_opponents[0].raw == {}
    assert view.agents[0].raw == {"agentPerception": [20]}
    assert [item.uid for item in view.perceived_opponents(view.agents[0])] == [20]
    assert [item.uid for item in view.commanders] == [500]
    assert view.entity_by_uid(600) is None
    assert view.raw == {"dataGlobal": {"timeCnt": 3}}

    view.agents[0].raw["agentPerception"].append(21)
    view.visible_opponents[0].position[0] = -999
    assert red.raw["agentPerception"] == [20]
    assert visible_blue.position[0] != -999

    events = (
        {"Event": "Destroyed", "DamageCauser": "10", "Target": "20"},
        {"Event": "Destroyed", "DamageCauser": "21", "Target": "10"},
        {"Event": "Destroyed", "DamageCauser": "21", "Target": "20"},
        {"Event": "EndEpisode", "WinTeam": "0"},
    )
    visible_events = Runner._filter_visible_events(events, view, view)
    assert visible_events == events


def test_explicit_zero_perception_range_is_not_replaced_by_default():
    observer = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_MNWS_Vehicle_6x6UGV_C",
        "agentHp": 100, "perceptionRange": 0,
        "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    ally = Entity.from_property({
        "uId": 11, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 1, "y": 0, "z": 0},
    }, 1)

    assert not PerceptionModel.in_range(observer, ally)


def test_missing_perception_range_does_not_create_synthetic_visibility():
    observer = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    target = Entity.from_property({
        "uId": 20, "agentTeam": 1, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 1, "y": 0, "z": 0},
    }, 0)

    assert not PerceptionModel.can_observe(observer, target)


def test_team_state_keeps_raw_unit_type_and_objective_values():
    agent = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_Future_Helicopter_C",
        "agentHp": 100, "maxMoveSpeed": 600, "fireRange": 1000,
        "initLocation": {"x": 100, "y": 200, "z": 300},
    }, 0)
    objective = KeyObject.from_raw({
        "uId": 32768, "hp": -1, "location": {"x": 1000, "y": 0, "z": 0},
    })
    state = TeamState(
        team=0, action_mask=np.ones((1, 1), dtype=bool), agents=(agent,),
        visible_opponents=(), raw={}, episode=1, step=0, training=False,
        key_objects=(objective,),
    )

    assert state.agents[0].entity_type == "BP_Future_Helicopter_C"
    assert state.agents[0].properties["fireRange"] == 1000
    assert state.key_objects[0].hp == -1
    assert state.key_objects[0].position.tolist() == [1000, 0, 0]


def test_algorithm_config_pointer_must_match_algorithm_entry():
    red = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    blue = Entity.from_property({
        "uId": 20, "agentTeam": 1, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 1, "y": 0, "z": 0},
    }, 0)
    context = AlgorithmContext(0, (red,), (blue,), ActionSetFactory.standard(1), "cpu", {})
    raw_config = {
        "config.py->GlobalConfig": {
            "AlgorithmConfig_Red": (
                "TeamAlg.BlueModel.Algorithm.RLAgentAlgorithm.py->AlgorithmConfig"
            )
        }
    }

    try:
        AlgorithmLoader(raw_config).load(
            "TeamAlg.RedModel.Algorithm.RLAgentAlgorithm->ReinforceAgentAlgorithm",
            context,
        )
    except ValueError as exc:
        assert "AlgorithmConfig_Red 与 Algorithm_Names 不一致" in str(exc)
    else:
        raise AssertionError("配置指针与算法入口不一致时应拒绝启动")


def test_terminal_result_prefers_valid_ue_winner_and_uses_global_fallback():
    red = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    blue = Entity.from_property({
        "uId": 20, "agentTeam": 1, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 1, "y": 0, "z": 0},
    }, 0)
    environment = object.__new__(CssimEnvironment)
    environment.entities = (red, blue)

    winner, reason = environment._terminal_result(
        ({"Event": "EndEpisode", "WinTeam": "-1", "EndReason": "结束事件"},),
        True,
        {"teamWin": 1, "episodeEndReason": "全局结束"},
    )
    assert (winner, reason) == (1, "结束事件")

    winner, _ = environment._terminal_result(
        ({"Event": "EndEpisode", "WinTeam": "0"},), True, {"teamWin": 1}
    )
    assert winner == 0


def test_simulation_log_details_follow_log_level(tmp_path):
    record = {
        "episode": 1,
        "step": 2,
        "actions": 3,
        "entities": [{"uId": 10}],
        "action_strings": ["ActionSet2::Idle;Idle;10"],
    }
    info_log = SimulationLog(tmp_path / "info", detailed=False)
    info_log.write_simulation_record(record)
    info_log.close()
    info_record = json.loads(info_log.stats_path.read_text(encoding="utf-8"))
    assert info_record == {"episode": 1, "step": 2, "actions": 3}

    debug_log = SimulationLog(tmp_path / "debug", detailed=True)
    debug_log.write_protocol(
        "ue_to_python", "frame", "step",
        '{"dataArr":[{"availActions":[20]}]}',
    )
    debug_log.write_simulation_record(record)
    debug_log.close()
    debug_record = json.loads(debug_log.stats_path.read_text(encoding="utf-8"))
    assert debug_record["entities"] == [{"uId": 10}]
    assert debug_record["action_strings"] == ["ActionSet2::Idle;Idle;10"]
    protocol_record = json.loads(
        debug_log.protocol_path.read_text(encoding="utf-8")
    )
    assert protocol_record["payload"]["dataArr"][0]["availActions"] == [20]


def test_protocol_analyzer_profiles_fields_and_perception_relations(tmp_path):
    source = tmp_path / "protocol.jsonl"
    source.write_text(json.dumps({
        "direction": "ue_to_python",
        "channel": "frame",
        "type": "step",
        "payload": {
            "dataArr": [{
                "uId": 10, "agentTeam": 0, "agentAlive": True,
                "availActions": [20, 21], "agentPerception": [20],
            }, {
                "uId": 20, "agentTeam": 1, "agentAlive": True,
                "availActions": [], "agentPerception": [],
            }, {
                "uId": 21, "agentTeam": 1, "agentAlive": True,
                "availActions": [], "agentPerception": [],
            }],
            "dataGlobal": {"timeCnt": 3},
        },
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "report.md"

    analyzer = ProtocolAnalyzer.load(source)
    analyzer.write(output)

    report = output.read_text(encoding="utf-8")
    assert "perception⊆avail" in report
    assert "队伍共享池与存活敌方感知并集" in report
    assert "| 0 | 1 | 0 | 0 | 1 |" in report
    assert "team=0 `availActions` 首次非空帧：3" in report
    assert "payload.dataArr[].availActions" in report


def test_template_entries_only_select_rule_or_dqn():
    red = Entity.from_property({
        "uId": 10, "agentTeam": 0, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 0, "y": 0, "z": 0},
    }, 0)
    blue = Entity.from_property({
        "uId": 20, "agentTeam": 1, "type": "BP_BaseSoldier_C",
        "agentHp": 100, "initLocation": {"x": 1000, "y": 0, "z": 0},
    }, 0)
    cases = (
        (RedTemplateEntry, 0, red, RuleAlgorithm),
        (BlueTemplateEntry, 1, blue, RuleAlgorithm),
    )
    for entry, team, agent, expected_type in cases:
        context = AlgorithmContext(
            team, (agent,), (), ActionSetFactory.standard(), "cpu", {"mode": "rule"}
        )
        assert isinstance(entry(context), expected_type)

    dqn_cases = (
        (RedTemplateEntry, 0, red, "red_minimal_control"),
        (BlueTemplateEntry, 1, blue, "blue_minimal_control"),
    )
    for entry, team, agent, action_set_name in dqn_cases:
        context = AlgorithmContext(
            team,
            (agent,),
            (),
            ActionSetFactory.standard(),
            "cpu",
            {"mode": "reinforcement_learning", "hidden_size": 16},
        )
        algorithm = entry(context)
        assert isinstance(algorithm, ReinforcementAlgorithm)
        assert algorithm.context.action_set.name == action_set_name
        assert algorithm.context.action_set.action_keys == (
            "idle",
            "move:+X",
            "move:+Y",
            "move:-X",
            "move:-Y",
            "move:assigned_objective",
            "attack:nearest_local",
            "special:change_parent:nearest_alive_parent",
        )

    with pytest.raises(ValueError, match="未知参数"):
        RedTemplateEntry(AlgorithmContext(
            0,
            (red,),
            (),
            ActionSetFactory.standard(),
            "cpu",
            {"mode": "rule", "implementation": "action_coverage"},
        ))
