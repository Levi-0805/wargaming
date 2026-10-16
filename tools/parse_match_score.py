"""Summarize CSSIM Output or protocol JSON lines without guessing sides.

Final score = Score[*].SumWorth + Reward. Empty Events do not identify a side.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

_POINTS = re.compile(r"([+-]?\d+)")


def _points(text: object) -> int:
    match = _POINTS.search(str(text or ""))
    return int(match.group(1)) if match else 0


def _score_packet(packet: dict, line_number: int) -> dict | None:
    if not isinstance(packet.get("Score"), dict):
        return None
    total = 0
    breakdown: dict[str, int] = {}
    try:
        for item in packet["Score"].values():
            if not isinstance(item, dict):
                continue
            worth = int(item.get("SumWorth") or 0)
            name = str(item.get("Name") or "")
            if name:
                breakdown[name] = breakdown.get(name, 0) + worth
            total += worth
        reward = int(packet["Reward"]) if "Reward" in packet else None
    except (TypeError, ValueError, OverflowError):
        return None
    events = packet.get("Events")
    events = [e for e in events if isinstance(e, dict)] if isinstance(events, list) else []
    # The simulator duplicates some suicide events into BOTH teams' lists.
    # Neither their killer label nor their beneficiary identifies the packet.
    # Prefer consistent cross-team kills; exclude known same-team events from
    # attribution, but retain them verbatim in the event audit below.
    attribution_events = [event for event in events if not (
        str(event.get("摧毁方", "")).lower() in {"red", "blue"}
        and str(event.get("摧毁方", "")).lower() == str(event.get("被摧毁方", "")).lower())]
    beneficiaries = set()
    for event in attribution_events:
        text = str(event.get("得分", ""))
        if "红方" in text:
            beneficiaries.add("red")
        if "蓝方" in text:
            beneficiaries.add("blue")
    labels = {str(e.get("摧毁方", "")).strip().lower() for e in attribution_events}
    side = (next(iter(beneficiaries)) if len(beneficiaries) == 1 else None)
    evidence = "Events.得分 beneficiary" if side else "unknown: no consistent red/blue event label"
    if not beneficiaries and len(labels) == 1 and labels <= {"red", "blue"}:
        side = next(iter(labels))
        evidence = "Events.摧毁方"
    return {
        "side": side,
        "side_evidence": evidence,
        "line": line_number, "kill_score": total, "capture_score": reward,
        "total": total + reward if reward is not None else None,
        "breakdown": breakdown,
        "kills": [{"time": e.get("Time"), "killer": e.get("摧毁方实体"),
                   "killer_side": e.get("摧毁方"), "victim_side": e.get("被摧毁方"),
                   "victim": e.get("被摧毁方实体"), "points": _points(e.get("得分"))}
                  for e in events],
    }


def parse_output(path: Path, source_commit: str | None = None) -> dict:
    """Read a bounded snapshot; compatibility fields refer only to its last round.

    source_commit is caller-supplied provenance, never inferred from file dates
    or the current checkout. Historical logs may have no such binding.
    """
    path = Path(path)
    rounds: list[dict] = []
    current: dict | None = None
    previous_step = None
    previous_events: dict[str, list] = {}
    malformed_lines: list[int] = []
    digest = hashlib.sha256()
    limit = path.stat().st_size
    read = 0

    def new_round() -> dict:
        result = {"episode": len(rounds) + 1, "frames": 0, "last_step": None,
                  "score_packets": [], "capture": []}
        rounds.append(result)
        return result

    with path.open("rb") as handle:
        line_number = 0
        while read < limit:
            raw = handle.readline(limit - read)
            if not raw:
                break
            read += len(raw)
            digest.update(raw)
            line_number += 1
            stripped = raw.strip()
            if not stripped.startswith(b"{"):
                continue
            try:
                packet = json.loads(stripped)
            except (ValueError, UnicodeDecodeError):
                malformed_lines.append(line_number)
                continue
            if not isinstance(packet, dict):
                continue
            if "direction" in packet:
                if packet.get("direction") != "ue_to_python":
                    continue
                packet = packet.get("payload")
                if not isinstance(packet, dict):
                    continue

            # A new roster or backwards step starts a fresh round, including
            # a round that has begun but has not produced any scores yet.
            if packet.get("propertyArr"):
                if current is None or current["frames"] or current["score_packets"]:
                    current = new_round()
                previous_step = None
            global_data = packet.get("dataGlobal")
            global_data = global_data if isinstance(global_data, dict) else {}
            if packet.get("dataArr") or "timeCnt" in global_data:
                step = global_data.get("timeCnt")
                reset = (isinstance(step, (int, float)) and isinstance(previous_step, (int, float))
                         and step < previous_step)
                if current is None or reset:
                    current = new_round()
                current["frames"] += 1
                current["last_step"] = step
                previous_step = step
            key_objects = global_data.get("keyObjArr")
            if isinstance(key_objects, list):
                if current is None:
                    current = new_round()
                current["capture"] = [{"name": item.get("rSVD1"),
                    "red": item.get("redTeamNum"), "blue": item.get("blueteamNum"),
                    "percent": item.get("percent")} for item in key_objects if isinstance(item, dict)]

            score = _score_packet(packet, line_number)
            if score is not None:
                # This simulator may retain Events across rounds while Score
                # itself resets. Remove only an exactly matching prior prefix.
                all_events = score["kills"]
                prior = previous_events.get(score["side"], [])
                offset = len(prior) if prior and all_events[:len(prior)] == prior else 0
                score["event_history_prefix_removed"] = offset
                score["recorded_event_count"] = len(all_events)
                score["kills"] = all_events[offset:]
                if score["side"]:
                    previous_events[score["side"]] = all_events
                # Score-only extracts lack frame resets: preserve each pair.
                if current is None or len(current["score_packets"]) >= 2:
                    current = new_round()
                current["score_packets"].append(score)

    for episode in rounds:
        packets = episode["score_packets"]
        counts = {side: sum(p["side"] == side for p in packets) for side in ("red", "blue")}
        episode["scores"] = {side: next((p for p in packets if p["side"] == side), None)
                             if counts[side] == 1 else None for side in counts}
        episode["unknown_scores"] = [p for p in packets if p["side"] is None]
        episode["score_packets_complete"] = len(packets) == 2 and all(p["total"] is not None for p in packets)
        episode["complete"] = episode["score_packets_complete"] and all(counts[s] == 1 for s in counts)
    last = rounds[-1] if rounds else {}
    result = {
        "output_file": str(path), "source_sha256": digest.hexdigest(), "source_bytes": read,
        "source_commit": source_commit,
        "source_commit_evidence": "provided_by_caller" if source_commit else "unbound historical source",
        "score_definition": "Score[*].SumWorth + Reward; missing or ambiguous sides remain null",
        "complete": bool(last.get("complete")) and not malformed_lines,
        "malformed_json_lines": malformed_lines,
        "rounds": rounds, "capture": last.get("capture", []),
    }
    for side in ("red", "blue"):
        score = last.get("scores", {}).get(side)
        result[side + "_score"] = score["total"] if score else None
        result[side + "_kill_score"] = score["kill_score"] if score else None
        result[side + "_capture_score"] = score["capture_score"] if score else None
        result[side + "_breakdown"] = score["breakdown"] if score else {}
        result[side + "_kills"] = score["kills"] if score else []
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--out", type=Path, help="Write the summary JSON here")
    parser.add_argument("--source-commit", help="Verified source revision supplied by the match launcher")
    args = parser.parse_args()
    result = parse_output(args.output, args.source_commit)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"complete={result['complete']} rounds={len(result['rounds'])} "
          f"red={result['red_score']} blue={result['blue_score']}")
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
