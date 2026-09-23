"""Read a CSSIM server Output_*.txt and summarize red/blue kill scores."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_POINTS = re.compile(r"([+-]?\d+)")


def _points(text: object) -> int:
    match = _POINTS.search(str(text or ""))
    return int(match.group(1)) if match else 0


def _sum_worth(score: object) -> tuple[int, dict[str, int]]:
    total = 0
    breakdown: dict[str, int] = {}
    if not isinstance(score, dict):
        return total, breakdown
    for item in score.values():
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or "")
        try:
            worth = int(item.get("SumWorth") or 0)
        except (TypeError, ValueError):
            worth = 0
        if name:
            breakdown[name] = breakdown.get(name, 0) + worth
        total += worth
    return total, breakdown


def parse_output(path: Path) -> dict:
    """Return kill scores and the final capture-point snapshot."""

    red_events: list[dict] = []
    blue_events: list[dict] = []
    red_worth = 0
    blue_worth = 0
    red_breakdown: dict[str, int] = {}
    blue_breakdown: dict[str, int] = {}
    capture: list[dict] = []
    saw_events = False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            if '"Events"' not in stripped and '"keyObjArr"' not in stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            events = payload.get("Events")
            if isinstance(events, list) and events:
                saw_events = True
                worth, breakdown = _sum_worth(payload.get("Score"))
                side = str(events[0].get("摧毁方") or "")
                rows = []
                for event in events:
                    if not isinstance(event, dict):
                        continue
                    rows.append({
                        "time": event.get("Time"),
                        "killer": event.get("摧毁方实体"),
                        "victim": event.get("被摧毁方实体"),
                        "points": _points(event.get("得分")),
                    })
                if side.lower() == "red":
                    red_events = rows
                    red_worth = worth
                    red_breakdown = breakdown
                else:
                    blue_events = rows
                    blue_worth = worth
                    blue_breakdown = breakdown

            key_objects = (payload.get("dataGlobal") or {}).get("keyObjArr")
            if isinstance(key_objects, list) and key_objects:
                capture = [
                    {
                        "name": item.get("rSVD1"),
                        "red": item.get("redTeamNum"),
                        "blue": item.get("blueteamNum"),
                        "percent": item.get("percent"),
                    }
                    for item in key_objects
                    if isinstance(item, dict)
                ]

    red_from_events = sum(item["points"] for item in red_events)
    blue_from_events = sum(item["points"] for item in blue_events)
    return {
        "output_file": str(path),
        "complete": saw_events,
        "red_score": red_worth if red_worth else red_from_events,
        "blue_score": blue_worth if blue_worth else blue_from_events,
        "red_breakdown": red_breakdown,
        "blue_breakdown": blue_breakdown,
        "red_kills": red_events,
        "blue_kills": blue_events,
        "capture": capture,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a CSSIM match output file")
    parser.add_argument("output", type=Path)
    parser.add_argument("--out", type=Path, help="Write the summary JSON here")
    args = parser.parse_args()
    result = parse_output(args.output)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(
        f"complete={result['complete']} red={result['red_score']} "
        f"blue={result['blue_score']} capture={json.dumps(result['capture'], ensure_ascii=False)}"
    )
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
