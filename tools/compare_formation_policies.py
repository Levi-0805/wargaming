"""Compare formation policies on the same recorded public observations.

Retired orders must reproduce the selected historical record before using it
as a control. Changed orders do not predict movement, survival, or score.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import logging
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'PythonCode'))
from cssim.algorithms import AlgorithmContext
from cssim.config import RuntimeConfig
from cssim.environment.env import CssimEnvironment
from cssim.protocol import ActionSetFactory
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RuleAgentAlgorithm import ReinforceAgentAlgorithm
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.VehicleFireControl import VehicleFireControl
from audit_vehicle_recovery import hashes, legacy_patrol

MODES = {'recorded_retired': (True, True), 'guard_fix_only': (True, True),
         'no_entry': (True, False), 'opening_only': (False, False)}


def compare(source, selected):
    before = hashes()
    config = RuntimeConfig(ROOT, {}, '127.0.0.1', 21051, ('red', 'blue'), 500, 1, 1., 60., .5, 0, 'cpu', 'WARNING')
    env = CssimEnvironment(config)
    rounds, current, previous, episode, rules, pending = [], None, None, 0, {}, {}
    limit, consumed, digest = source.stat().st_size, 0, hashlib.sha256()
    with source.open('rb') as stream:
        while consumed < limit:
            line = stream.readline(limit - consumed)
            if not line:
                break
            consumed += len(line)
            digest.update(line)
            row = json.loads(line)
            packet = row.get('payload', {})
            if not isinstance(packet, dict):
                continue
            if row.get('direction') == 'ue_to_python':
                if packet.get('propertyArr'):
                    env._build_roster(packet)
                    previous = None
                if not packet.get('dataArr'):
                    continue
                step = packet['dataGlobal']['timeCnt']
                if previous is None or step < previous:
                    episode += 1
                    env.episode = episode
                    current, rules, pending = None, {}, {}
                    if episode in selected:
                        current = {'episode': episode, 'frames': 0, 'recorded_matches': 0,
                                   'recorded_mismatches': 0, 'modes': {}}
                        rounds.append(current)
                        print('compare episode', episode, flush=True)
                previous = step
                if current is None:
                    continue
                snapshot = env._parse_state(packet)
                state = env.build_team_state(snapshot, 0)
                if not rules:
                    context = AlgorithmContext(0, state.agents, state.visible_opponents,
                                               ActionSetFactory.standard(), 'cpu', {}, state.commanders)
                    for name, (column, entry) in MODES.items():
                        rules[name] = ReinforceAgentAlgorithm(context, enable_continuous_column=column,
                                                             enable_entry_waves=entry)
                        current['modes'][name] = {'column_holds': 0, 'entry_holds': 0,
                            'changed_from_guard_fix': 0, 'changes_by_type': Counter(),
                            'pre_hq_diversions': 0, 'invalid_vehicle_fire': 0,
                            'invalid_uav_fire': 0, 'opening_released_at': None,
                            'column_first_wait': None, 'column_last_wait': None,
                            'objectives': [], 'changes': []}
                wires = {}
                for name, rule in rules.items():
                    context = (patch.object(VehicleFireControl, 'patrol', staticmethod(legacy_patrol))
                               if name == 'recorded_retired' else nullcontext())
                    with context:
                        actions = rule.act(state)
                        rule.take_selected_special_commands()
                    wires[name] = {u.uid: a.to_ue_string(u.uid, snapshot.entities) for u, a in zip(state.agents, actions)}
                    stats = current['modes'][name]
                    stats['column_holds'] += len(rule._column.held)
                    stats['entry_holds'] += len(rule._entry_waves.held)
                    if rule._column.held:
                        if stats['column_first_wait'] is None:
                            stats['column_first_wait'] = step
                        stats['column_last_wait'] = step
                    if rule._cohesion.released and stats['opening_released_at'] is None:
                        stats['opening_released_at'] = step
                    obj = rule._route.objective(state)
                    uid = obj.uid if obj else None
                    if not stats['objectives'] or stats['objectives'][-1]['uid'] != uid:
                        stats['objectives'].append({'step': step, 'uid': uid,
                            'name': obj.raw.get('rSVD1') if obj else None})
                    stats['pre_hq_diversions'] += bool(obj and not rule._route.hq_secured and not rule._route.is_hq(obj))
                    for unit, action in zip(state.agents, actions):
                        if unit.entity_type in rule.VEHICLES and action.command == 'NormalAttacking':
                            eligible = {e.uid for e in rule._vehicle_fire.eligible(state, unit)}
                            stats['invalid_vehicle_fire'] += action.target_uid not in eligible
                        if unit.entity_type == rule._UAV and action.command in {'NormalAttacking', 'SelfDestruct'}:
                            mission = rule._uav._missions.get(unit.uid)
                            target = mission[0] if action.command == 'SelfDestruct' and mission else action.target_uid
                            stats['invalid_uav_fire'] += target not in unit.perceived_uids()
                pending = wires['recorded_retired']
                for name, commands in wires.items():
                    if name == 'recorded_retired':
                        continue
                    stats = current['modes'][name]
                    for unit in state.agents:
                        old, new = wires['guard_fix_only'][unit.uid], commands[unit.uid]
                        if old != new:
                            stats['changed_from_guard_fix'] += 1
                            stats['changes_by_type'][unit.entity_type] += 1
                            if len(stats['changes']) < 6:
                                stats['changes'].append({'step': step, 'uid': unit.uid, 'old': old, 'new': new})
                current['frames'] += 1
            elif row.get('direction') == 'python_to_ue' and current:
                for action in packet.get('StringActions', []) or []:
                    uid = int(action.split(';')[-1])
                    if uid in pending:
                        current['recorded_matches' if action == pending[uid] else 'recorded_mismatches'] += 1
                pending = {}
    if hashes() != before:
        raise RuntimeError('Source changed during comparison; rerun.')
    return {'source': str(source), 'source_bytes': consumed, 'source_sha256': digest.hexdigest(),
            'candidate_file_sha256': before, 'episodes': rounds,
            'limitation': 'Ablations on fixed public observations; not a prediction of physical movement or new scores.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--episodes', type=int, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    result = compare(args.source, set(args.episodes))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    failed = len(result['episodes']) != len(set(args.episodes))
    for row in result['episodes']:
        failed |= not row['recorded_matches'] or bool(row['recorded_mismatches'])
        for name, stats in row['modes'].items():
            failed |= bool(stats['pre_hq_diversions'] or stats['invalid_vehicle_fire'] or stats['invalid_uav_fire'])
            print(row['episode'], name, {k: stats[k] for k in
                ('column_holds', 'entry_holds', 'changed_from_guard_fix', 'column_last_wait')})
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
