"""Compare the retired Guard rewrite with preserved movement on recorded views.

This validates transmitted commands, not engine movement or new match scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
from cssim.protocol import Action, ActionSetFactory
from cssim.protocol.actions import GUARD_DIRECTIONS
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RuleAgentAlgorithm import ReinforceAgentAlgorithm
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.VehicleFireControl import VehicleFireControl


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'PythonCode/TeamAlg/FsLutk7wD2RkNgbs/Algorithm').glob('*.py'))}


def legacy_patrol(action):
    if action.command == 'Moving':
        if action.points is not None:
            return Action.guard_at(action.points)
        if action.direction in GUARD_DIRECTIONS:
            return Action.guard(action.direction)
    return action


def audit(source, selected):
    before = hashes()
    config = RuntimeConfig(ROOT, {}, '127.0.0.1', 21051, ('red', 'blue'), 500, 1, 1., 60., .5, 0, 'cpu', 'WARNING')
    env = CssimEnvironment(config)
    previous, episode, current = None, 0, None
    legacy = candidate = None
    pending, rounds = {}, []
    size, consumed, digest = source.stat().st_size, 0, hashlib.sha256()
    with source.open('rb') as stream:
        while consumed < size:
            line = stream.readline(size - consumed)
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
                if not packet.get('dataArr'):
                    continue
                step = packet['dataGlobal']['timeCnt']
                if previous is None or step < previous:
                    episode += 1
                    env.episode = episode
                    legacy = candidate = current = None
                    pending = {}
                    if episode in selected:
                        print('audit episode', episode, flush=True)
                        current = {'episode': episode, 'frames': 0, 'legacy_recorded_matches': 0,
                            'legacy_recorded_mismatches': 0, 'changed_commands': 0,
                            'unexpected_changes': 0, 'invalid_vehicle_fire': 0,
                            'changed_by_unit': Counter(), 'examples': [], 'lynx_466': []}
                        rounds.append(current)
                previous = step
                if current is None:
                    continue
                snapshot = env._parse_state(packet)
                state = env.build_team_state(snapshot, 0)
                if legacy is None:
                    context = AlgorithmContext(0, state.agents, state.visible_opponents,
                        ActionSetFactory.standard(), 'cpu', {}, state.commanders)
                    legacy, candidate = ReinforceAgentAlgorithm(context), ReinforceAgentAlgorithm(context)
                with patch.object(VehicleFireControl, 'patrol', staticmethod(legacy_patrol)):
                    old = legacy.act(state)
                    legacy.take_selected_special_commands()
                new = candidate.act(state)
                candidate.take_selected_special_commands()
                pending = {u.uid: a.to_ue_string(u.uid, snapshot.entities) for u, a in zip(state.agents, old)}
                for unit, original, action in zip(state.agents, old, new):
                    wire = action.to_ue_string(unit.uid, snapshot.entities)
                    if unit.entity_type in candidate.VEHICLES and action.command == 'NormalAttacking':
                        eligible = {e.uid for e in candidate._vehicle_fire.eligible(state, unit)}
                        current['invalid_vehicle_fire'] += action.target_uid not in eligible
                    if original == action:
                        continue
                    current['changed_commands'] += 1
                    current['changed_by_unit'][unit.uid] += 1
                    expected = (unit.entity_type in candidate.VEHICLES and original.command == 'Guard'
                                and action.command == 'Moving' and original.points is None and action.points is None
                                and original.direction == action.direction)
                    current['unexpected_changes'] += not expected
                    sample = {'step': step, 'uid': unit.uid, 'type': unit.entity_type,
                        'position': [float(v) for v in unit.position], 'hp': unit.hp,
                        'ammo': VehicleFireControl.ammo(unit), 'old': pending[unit.uid], 'new': wire}
                    if len(current['examples']) < 8:
                        current['examples'].append(sample)
                    if episode == 1 and unit.uid == 466 and len(current['lynx_466']) < 8:
                        current['lynx_466'].append(sample)
                current['frames'] += 1
            elif row.get('direction') == 'python_to_ue' and current is not None:
                for action in packet.get('StringActions', []) or []:
                    uid = int(action.split(';')[-1])
                    if uid in pending:
                        key = 'legacy_recorded_matches' if action == pending[uid] else 'legacy_recorded_mismatches'
                        current[key] += 1
                pending = {}
    if hashes() != before:
        raise RuntimeError('Algorithm changed during audit; rerun after edits finish.')
    return {'source': str(source), 'source_bytes': consumed, 'source_sha256': digest.hexdigest(),
            'candidate_file_sha256': before, 'episodes': rounds,
            'limitation': 'Fixed observations cannot establish successful escape or score improvement.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--episodes', type=int, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-recorded-match', action='store_true')
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    result = audit(args.source, set(args.episodes))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    failed = not result['episodes']
    for ep in result['episodes']:
        print({k: v for k, v in ep.items() if k not in ('examples', 'lynx_466', 'changed_by_unit')})
        failed |= bool(ep['unexpected_changes'] or ep['invalid_vehicle_fire']
                       or args.require_recorded_match and ep['legacy_recorded_mismatches'])
    raise SystemExit(1 if failed else 0)
