"""Audit scores and actual radio state in every episode of a CSSIM output.

Usage: python tools/analyze_command_links.py Output_*.txt --out summary.json
Scores include the reported Reward; link failures are observations, not an
assertion that every motion problem was caused by radio loss.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path


def point(item):
    loc = item.get("agentLocation", item.get("initLocation", {}))
    return tuple(float(loc.get(axis, 0)) for axis in ("x", "y", "z"))


def analyze(path):
    properties, games = {}, []
    game = None
    last_step = None
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip().startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            for item in payload.get("propertyArr", []):
                properties[int(item["uId"])] = item
            data = payload.get("dataArr")
            if data:
                step = payload.get("dataGlobal", {}).get("timeCnt")
                if game is None or (last_step is not None and step < last_step):
                    game = {"episode": len(games) + 1, "scores": {}, "units": {}, "frames": 0}
                    games.append(game)
                last_step = step
                game["frames"] += 1
                by_uid = {int(item["uId"]): item for item in data}
                for item in data:
                    if item.get("agentTeam") != 0 or not item.get("agentAlive"):
                        continue
                    if item.get("type") in ("BP_BaseSoldier_C", "BaseCommander_C"):
                        continue
                    uid = int(item["uId"])
                    unit = game["units"].setdefault(uid, {
                        "uid": uid, "type": item.get("type"), "samples": 0,
                        "offline_frames": 0, "first_offline_step": None,
                        "dead_parent_frames": 0, "over_1km_frames": 0,
                        "parent_uids": [], "start": point(item), "end": point(item),
                    })
                    unit["samples"] += 1
                    unit["end"] = point(item)
                    offline = str(item.get("rSVD1", "")).rpartition(";")[-1] == "0"
                    if offline:
                        unit["offline_frames"] += 1
                        if unit["first_offline_step"] is None:
                            unit["first_offline_step"] = step
                    parent_uid = item.get("indexInTeam")
                    if parent_uid not in unit["parent_uids"]:
                        unit["parent_uids"].append(parent_uid)
                    parent = by_uid.get(parent_uid, properties.get(parent_uid))
                    if parent:
                        unit["dead_parent_frames"] += int(parent.get("agentAlive") is False)
                        unit["over_1km_frames"] += int(math.dist(point(item), point(parent)) > 100_000)
                game["capture"] = [
                    {key: item.get(key) for key in ("rSVD1", "redTeamNum", "blueteamNum", "percent")}
                    for item in payload.get("dataGlobal", {}).get("keyObjArr", [])
                ]
            events = payload.get("Events")
            if game is not None and isinstance(events, list) and events:
                side = str(events[0].get("摧毁方", "")).lower()
                if side not in ("red", "blue"):
                    continue
                kills = sum(int(item.get("SumWorth", 0)) for item in payload.get("Score", {}).values()
                            if isinstance(item, dict))
                reward = int(payload.get("Reward", 0) or 0)
                game["scores"][side] = {"kills": kills, "reward": reward, "total": kills + reward}
    for game in games:
        units = []
        for unit in game["units"].values():
            unit["displacement_m"] = round(math.dist(unit.pop("start"), unit.pop("end")) / 100, 1)
            units.append(unit)
        game["units"] = units
        game["offline_units_by_type"] = dict(Counter(unit["type"] for unit in units if unit["offline_frames"]))
    return {"source": str(Path(path).resolve()), "episodes": games}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(args.output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for game in result["episodes"]:
        print(json.dumps({key: game[key] for key in ("episode", "scores", "offline_units_by_type")}, ensure_ascii=False))
