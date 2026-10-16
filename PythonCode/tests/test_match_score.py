"""Result attribution tests using the observed CSSIM finish-packet structure."""
import hashlib
import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "parse_match_score", Path(__file__).resolve().parents[2] / "tools/parse_match_score.py")
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)


def score(side, kills=0, reward=0):
    return {"Events": [{"摧毁方": side, "得分": str(kills)}] if side else [],
            "Score": {"soldier": {"Name": "士兵", "SumWorth": kills}},
            "Command": [], "Reward": reward}


def read(tmp_path, packets, protocol=False):
    if protocol:
        packets = [{"direction": "ue_to_python", "channel": "finish", "payload": p} for p in packets]
    path = tmp_path / "output.jsonl"
    path.write_text("\n".join(json.dumps(p) for p in packets) + "\n", encoding="utf8")
    return parser.parse_output(path, "verified-revision"), path


def test_total_includes_capture_and_keeps_each_round(tmp_path):
    result, path = read(tmp_path, [score("red", 2670, 1600), score("Blue", 1700, 1600),
                                  score("red", 1530), score("Blue", 2660, 3200)], True)
    assert result["complete"]
    assert [r["scores"]["red"]["total"] for r in result["rounds"]] == [4270, 1530]
    assert result["red_score"] == 1530 and result["blue_score"] == 5860
    assert result["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["source_commit"] == "verified-revision"


def test_incomplete_last_round_never_uses_previous_scores(tmp_path):
    packets = [{"dataGlobal": {"timeCnt": 0}}, score("red", 1830, 2400), score("Blue", 2210, 800),
               {"propertyArr": [{"uId": 10}]}, {"dataGlobal": {"timeCnt": -1}}]
    result, _ = read(tmp_path, packets)
    assert len(result["rounds"]) == 2 and result["rounds"][0]["complete"]
    assert not result["complete"]
    assert result["red_score"] is None and result["blue_score"] is None


def test_empty_events_capture_is_retained_without_guessing_side(tmp_path):
    result, _ = read(tmp_path, [score(None, 0, 800), score("Blue", 1140, 2400)])
    assert not result["complete"] and result["red_score"] is None
    assert result["blue_score"] == 3540
    assert result["rounds"][0]["score_packets_complete"]
    assert result["rounds"][0]["unknown_scores"][0]["total"] == 800


def test_one_side_events_do_not_make_a_complete_result(tmp_path):
    result, _ = read(tmp_path, [score("red", 300, 1600)])
    assert not result["complete"] and result["blue_score"] is None


def test_zero_kills_with_known_side_are_valid_and_score_is_authoritative(tmp_path):
    red = score("red", 0, 1600)
    red["Events"][0]["得分"] = "100"
    result, _ = read(tmp_path, [red, score("Blue")])
    assert result["complete"] and result["red_score"] == 1600
    assert result["red_kill_score"] == 0


def test_missing_reward_or_malformed_tail_is_incomplete(tmp_path):
    red = score("red", 100)
    del red["Reward"]
    result, path = read(tmp_path, [red, score("Blue", 100)])
    assert not result["complete"] and result["red_score"] is None
    with path.open("a", encoding="utf8") as handle:
        handle.write('{"dataGlobal":')
    result = parser.parse_output(path)
    assert not result["complete"] and result["malformed_json_lines"] == [3]


def test_conflicting_event_labels_are_unknown_and_sent_packets_ignored(tmp_path):
    mixed = score("red", 100)
    mixed["Events"].append({"摧毁方": "Blue"})
    result, _ = read(tmp_path, [mixed, score("Blue", 200),
                              {"direction": "python_to_ue", "payload": score("red", 9999)}])
    assert not result["complete"] and len(result["rounds"]) == 1
    assert result["red_score"] is None


def test_step_reset_clears_capture_snapshot(tmp_path):
    result, _ = read(tmp_path, [{"dataGlobal": {"timeCnt": 10, "keyObjArr": [{"rSVD1": "HQ", "percent": 1}]}},
                              {"dataGlobal": {"timeCnt": -1}}, score("red", 100), score("Blue", 200)])
    assert len(result["rounds"]) == 2 and result["capture"] == []
    assert result["complete"]


def test_finish_can_arrive_before_last_frame_without_creating_phantom_round(tmp_path):
    result, _ = read(tmp_path, [{"dataGlobal": {"timeCnt": 277}},
        score("red", 2110), score("Blue", 3170, 3200), {"dataGlobal": {"timeCnt": 278}}])
    assert result["complete"] and len(result["rounds"]) == 1
    assert result["rounds"][0]["last_step"] == 278


def test_score_beneficiary_handles_suicide_and_cumulative_events(tmp_path):
    first = score("red", 100)
    first["Events"][0].update({"Time": "11:00:00/100", "得分": "（红方+100分)"})
    second = score("Blue", 10)
    second["Events"][0].update({"Time": "11:10:00/10", "得分": "（红方+10分)"})
    second["Events"].insert(0, first["Events"][0])
    result, _ = read(tmp_path, [first, score("Blue", 100), second, score("Blue", 200)])
    assert result["complete"] and result["red_score"] == 10
    assert len(result["red_kills"]) == 1
    assert result["red_kills"][0]["killer_side"] == "Blue"
    assert result["rounds"][1]["scores"]["red"]["event_history_prefix_removed"] == 1


def test_duplicated_same_team_events_do_not_pollute_packet_attribution(tmp_path):
    red, blue = score('red', 2430, 3200), score('Blue', 2880, 0)
    red['Events'][0].update({'被摧毁方': 'Blue', '得分': '红方+100分'})
    blue['Events'][0].update({'被摧毁方': 'red', '得分': '蓝方+100分'})
    shared = [{'摧毁方': 'Blue', '被摧毁方': 'Blue', '得分': '红方+10分'},
              {'摧毁方': 'red', '被摧毁方': 'red', '得分': '蓝方+10分'}]
    red['Events'].extend(shared)
    blue['Events'].extend(shared)
    result, _ = read(tmp_path, [red, blue])
    assert result['complete']
    assert result['red_score'] == 5630 and result['blue_score'] == 2880


def test_only_same_team_events_cannot_identify_score_packet(tmp_path):
    packet = score('Blue', 10)
    packet['Events'][0].update({'被摧毁方': 'Blue', '得分': '红方+10分'})
    result, _ = read(tmp_path, [packet, score('Blue', 100)])
    assert result['red_score'] is None and not result['complete']
