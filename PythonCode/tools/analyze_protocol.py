from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _example(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= 120 else text[:117] + "..."


def _uids(value: Any) -> frozenset[int]:
    result = set()
    for item in value if isinstance(value, list) else ():
        candidate = item.get("uId") if isinstance(item, dict) else item
        try:
            result.add(int(candidate))
        except (TypeError, ValueError):
            pass
    return frozenset(result)


@dataclass
class FieldStats:
    count: int = 0
    types: Counter[str] = field(default_factory=Counter)
    lengths: Counter[int] = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)

    def observe(self, value: Any) -> None:
        self.count += 1
        self.types[_type_name(value)] += 1
        if isinstance(value, (list, dict, str)):
            self.lengths[len(value)] += 1
        sample = _example(value)
        if sample not in self.examples and len(self.examples) < 3:
            self.examples.append(sample)


class ProtocolAnalyzer:
    """从 DEBUG 原始协议中生成字段画像和感知字段关系证据。"""

    def __init__(self, records: Iterable[dict[str, Any]]):
        self.records = tuple(records)
        self.fields: dict[str, FieldStats] = defaultdict(FieldStats)
        self.relation = Counter()
        self.first_nonempty: dict[tuple[int, str], int] = {}

    @staticmethod
    def load(path: Path) -> "ProtocolAnalyzer":
        records = (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        return ProtocolAnalyzer(records)

    def _walk(self, value: Any, path: str) -> None:
        self.fields[path].observe(value)
        if isinstance(value, dict):
            for name, item in value.items():
                self._walk(item, f"{path}.{name}")
        elif isinstance(value, list):
            for item in value:
                self._walk(item, f"{path}[]")

    def _observe_perception(self, payload: dict[str, Any]) -> None:
        entities = tuple(
            item for item in payload.get("dataArr", ()) if isinstance(item, dict)
        )
        if not entities:
            return
        step = int(payload.get("dataGlobal", {}).get("timeCnt", -1))
        for team in (0, 1):
            alive = tuple(
                item for item in entities
                if int(item.get("agentTeam", -1)) == team
                and bool(item.get("agentAlive", False))
            )
            if not alive:
                continue
            avail_sets = tuple(_uids(item.get("availActions")) for item in alive)
            perception_sets = tuple(_uids(item.get("agentPerception")) for item in alive)
            opponent_alive = frozenset(
                int(item["uId"])
                for item in entities
                if int(item.get("agentTeam", -1)) != team
                and bool(item.get("agentAlive", False))
                and item.get("uId") is not None
            )
            self.relation[f"team_{team}.frames"] += 1
            self.relation[f"team_{team}.alive_agent_samples"] += len(alive)
            self.relation[f"team_{team}.avail_total"] += sum(map(len, avail_sets))
            self.relation[f"team_{team}.perception_total"] += sum(
                map(len, perception_sets)
            )
            if len(set(avail_sets)) == 1:
                self.relation[f"team_{team}.frames_same_avail"] += 1
                team_pool = avail_sets[0]
                alive_perception_union = (
                    frozenset().union(*perception_sets) & opponent_alive
                )
                self.relation[f"team_{team}.pool_union_frames"] += 1
                if team_pool == alive_perception_union:
                    self.relation[f"team_{team}.pool_union_equal"] += 1
                if team_pool <= alive_perception_union:
                    self.relation[f"team_{team}.pool_subset_union"] += 1
                if alive_perception_union <= team_pool:
                    self.relation[f"team_{team}.union_subset_pool"] += 1
            if len(set(perception_sets)) == 1:
                self.relation[f"team_{team}.frames_same_perception"] += 1
            for avail, perception in zip(avail_sets, perception_sets):
                if avail == perception:
                    self.relation[f"team_{team}.equal"] += 1
                if perception <= avail:
                    self.relation[f"team_{team}.perception_subset_avail"] += 1
                if avail <= perception:
                    self.relation[f"team_{team}.avail_subset_perception"] += 1
            for name, sets in (
                ("availActions", avail_sets),
                ("agentPerception", perception_sets),
            ):
                if any(sets):
                    self.first_nonempty.setdefault((team, name), step)

    def analyze(self) -> None:
        for record in self.records:
            root = ".".join((
                str(record.get("direction", "unknown")),
                str(record.get("channel", "unknown")),
                str(record.get("type", "unknown")),
                "payload",
            ))
            payload = record.get("payload")
            self._walk(payload, root)
            if record.get("direction") == "ue_to_python" and isinstance(payload, dict):
                self._observe_perception(payload)

    def write(self, output: Path) -> None:
        self.analyze()
        lines = [
            "# CSSIM 原始协议字段画像",
            "",
            f"- 协议记录数：{len(self.records)}",
            "- 本报告只陈述日志中实际出现的数据类型和关系，不把字段名猜测当作正式语义。",
            "",
            "## availActions 与 agentPerception",
            "",
            "| 队伍 | 存活实体样本 | 平均 avail 数 | 平均 perception 数 | avail 全队相同帧 | perception 全队相同帧 | 两者相等样本 | perception⊆avail | avail⊆perception |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for team in (0, 1):
            prefix = f"team_{team}."
            samples = self.relation[prefix + "alive_agent_samples"]
            frames = self.relation[prefix + "frames"]
            average_avail = self.relation[prefix + "avail_total"] / max(samples, 1)
            average_perception = (
                self.relation[prefix + "perception_total"] / max(samples, 1)
            )
            lines.append(
                f"| {team} | {samples} | {average_avail:.3f} | "
                f"{average_perception:.3f} | "
                f"{self.relation[prefix + 'frames_same_avail']}/{frames} | "
                f"{self.relation[prefix + 'frames_same_perception']}/{frames} | "
                f"{self.relation[prefix + 'equal']} | "
                f"{self.relation[prefix + 'perception_subset_avail']} | "
                f"{self.relation[prefix + 'avail_subset_perception']} |"
            )
        lines.append("")
        lines.extend((
            "### 队伍共享池与存活敌方感知并集",
            "",
            "这里先将同队存活实体共有的 `availActions` 视为队伍池，再与全队 "
            "`agentPerception` 并集中的当帧存活敌方比较；这样不会把已阵亡但仍残留在感知列表中的 UID 当成当前候选。",
            "",
            "| 队伍 | 可比较帧 | 完全相等 | avail⊆感知并集 | 感知并集⊆avail |",
            "|---:|---:|---:|---:|---:|",
        ))
        for team in (0, 1):
            prefix = f"team_{team}."
            lines.append(
                f"| {team} | {self.relation[prefix + 'pool_union_frames']} | "
                f"{self.relation[prefix + 'pool_union_equal']} | "
                f"{self.relation[prefix + 'pool_subset_union']} | "
                f"{self.relation[prefix + 'union_subset_pool']} |"
            )
        lines.append("")
        for team in (0, 1):
            for name in ("availActions", "agentPerception"):
                value = self.first_nonempty.get((team, name), "未出现")
                lines.append(f"- team={team} `{name}` 首次非空帧：{value}")

        lines.extend((
            "",
            "## 全部字段",
            "",
            "| 路径 | 次数 | 类型 | 长度分布 | 样例 |",
            "|---|---:|---|---|---|",
        ))
        for name, stats in sorted(self.fields.items()):
            types = ", ".join(f"{key}:{value}" for key, value in stats.types.items())
            lengths = ", ".join(
                f"{key}:{value}" for key, value in sorted(stats.lengths.items())
            ) or "-"
            examples = "<br>".join(item.replace("|", "\\|") for item in stats.examples)
            lines.append(f"| `{name}` | {stats.count} | {types} | {lengths} | {examples} |")
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 CSSIM DEBUG protocol.jsonl")
    parser.add_argument("path", type=Path, help="protocol.jsonl 或包含它的日志目录")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    source = args.path / "protocol.jsonl" if args.path.is_dir() else args.path
    output = args.output or source.with_name("protocol_field_report.md")
    ProtocolAnalyzer.load(source).write(output)
    print(output.resolve())


if __name__ == "__main__":
    main()
