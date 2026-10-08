"""Read-only movement audit and counterfactual command validation.

Never contacts UE. Re-deciding recorded states cannot predict new scores.
Usage: PythonEnv/python tools/audit_assault_replay.py OUTPUT --protocol LOG --out JSON
"""

import argparse
from collections import Counter, defaultdict
import json
import logging
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "PythonCode"))
from cssim.algorithms import AlgorithmContext
from cssim.config import RuntimeConfig
from cssim.environment.env import CssimEnvironment
from cssim.protocol import ActionSetFactory
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RuleAgentAlgorithm import ReinforceAgentAlgorithm


def read_packets(path):
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def audit(output, protocol=None):
    old_actions = defaultdict(Counter)
    if protocol:
        for record in read_packets(protocol):
            if record.get("direction") != "python_to_ue" or record.get("channel") != "frame":
                continue
            for text in record.get("payload", {}).get("StringActions", []):
                parts = text.split(";")
                old_actions[int(parts[-1])][parts[0].split("::")[-1]] += 1
    config = RuntimeConfig(Path(output), {}, "127.0.0.1", 21051, ("red", "blue"),
                           200, 1, 1.0, 60.0, 0.5, 0, "cpu", "WARNING")
    env = CssimEnvironment(config)  # No connect/reset/step calls.
    games = []
    previous = None
    rule = None
    game = None
    for record in read_packets(output):
        if "direction" in record:
            if record.get("direction") != "ue_to_python" or not isinstance(record.get("payload"), dict):
                continue
            packet = record["payload"]
        else:
            packet = record
        if packet.get("propertyArr"):
            env._build_roster(packet)
        if packet.get("dataArr"):
            step = packet["dataGlobal"]["timeCnt"]
            if previous is None or step < previous:
                env.episode += 1
                game = {"episode": env.episode, "frames": 0, "scores_in_file_order": [],
                        "new_commands": Counter(), "new_specials": 0, "local_contact_rows": 0,
                        "contact_attacks": 0, "contact_moves": 0, "units": {},
                        "phase_changes": [], "first_strike_step": {}}
                games.append(game)
                rule = None
            previous = step
            snapshot = env._parse_state(packet)
            team = env.build_team_state(snapshot, 0)  # Same visibility filter as live play.
            if rule is None:
                rule = ReinforceAgentAlgorithm(AlgorithmContext(
                    0, team.agents, team.visible_opponents, ActionSetFactory.standard(), "cpu", {}, team.commanders))
            actions = rule.act(team)
            phase = rule._route.stage
            if phase != game.get("last_phase", 0):
                game["phase_changes"].append({"step": step, "stage": phase,
                    "objective": rule._route.objective(team).uid,
                    "garrison": len(rule._route._garrison)})
            game["last_phase"] = phase
            specials = rule.take_selected_special_commands()
            game["new_specials"] += len(specials)
            if game["frames"] == 0:
                game["initial_commands"] = dict(Counter(action.command for action in actions))
                game["initial_repairs"] = [command.parameters() for command in specials]
            game["frames"] += 1
            for agent, action in zip(team.agents, actions):
                # Validate serialization with the same full roster as live UE.
                action.to_ue_string(agent.uid, snapshot.entities)
                game["new_commands"][action.command] += 1
                if action.command == "SelfDestruct":
                    game["first_strike_step"].setdefault(agent.uid, step)
                if not agent.alive:
                    continue
                if team.perceived_opponents(agent):
                    game["local_contact_rows"] += 1
                    game["contact_attacks"] += action.command in ("NormalAttacking", "SelfDestruct")
                    game["contact_moves"] += action.command == "Moving"
                position = tuple(float(v) for v in agent.position)
                unit = game["units"].setdefault(agent.uid, {
                    "uid": agent.uid, "name": agent.class_name, "kind": agent.entity_type,
                    "first": position, "last": position, "path_cm": 0, "frames": 0,
                    "offline_frames": 0, "initial_parent": agent.commander_uid(),
                    "parent_changed": False, "old_commands": dict(old_actions[agent.uid]),
                })
                unit["path_cm"] += math.dist(unit["last"], position)
                unit["last"] = position
                unit["frames"] += 1
                unit["offline_frames"] += agent.communication_ok() is False
                unit["parent_changed"] |= agent.commander_uid() != unit["initial_parent"]
            game["last_objectives"] = [{key: obj.raw.get(key) for key in
                ("uId", "rSVD1", "redTeamNum", "blueteamNum", "percent")} for obj in team.key_objects]
        if "Score" in packet and game is not None:
            # Empty Events do not identify the side: preserve order, do not
            # drop zero-score results or mislabel them by guessing a side.
            events = packet.get("Events", [])
            side = events[0].get("摧毁方") if events else None
            kills = sum(int(value.get("SumWorth", 0)) for value in packet["Score"].values())
            reward = int(packet.get("Reward", 0) or 0)
            game["scores_in_file_order"].append({"side": side, "kills": kills, "reward": reward, "total": kills + reward})
    for game in games:
        for unit in game["units"].values():
            unit["net_m"] = round(math.dist(unit.pop("first"), unit.pop("last")) / 100, 1)
            unit["path_m"] = round(unit.pop("path_cm") / 100, 1)
        game["units"] = list(game["units"].values())
    return {"source": str(output), "protocol": str(protocol),
            "limitation": "Offline decisions on historical states, not a simulation of new outcomes. Special commands are not executed.",
            "episodes": games}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    result = audit(args.output, args.protocol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for episode in result["episodes"]:
        print(json.dumps({key: value for key, value in episode.items() if key not in ("units", "last_objectives")}, ensure_ascii=False))
