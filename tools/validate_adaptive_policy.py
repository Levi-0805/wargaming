"""Validate candidate commands on recorded public observations, without UE.

Recorded scores describe the historical policy only. Replaying observations
does not predict the score or survival produced by these new commands.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'PythonCode'))

from cssim.algorithms import AlgorithmContext
from cssim.config import RuntimeConfig
from cssim.environment.env import CssimEnvironment
from cssim.protocol import ActionSetFactory
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RuleAgentAlgorithm import ReinforceAgentAlgorithm
from parse_match_score import _score_packet


def candidate_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'PythonCode/TeamAlg/FsLutk7wD2RkNgbs/Algorithm').glob('*.py'))}


def validate(path, selected=()):
    hashes = candidate_hashes()
    limit, consumed = path.stat().st_size, 0
    digest = hashlib.sha256()
    config = RuntimeConfig(ROOT, {}, '127.0.0.1', 21051, ('red', 'blue'),
                           500, 1, 1., 60., .5, 0, 'cpu', 'WARNING')
    env = CssimEnvironment(config)  # No connect/reset/step calls.
    rounds, previous, current, rule = [], None, None, None
    episode = 0
    with path.open('rb') as source:
        line_number = 0
        while consumed < limit:
            raw = source.readline(limit - consumed)
            if not raw:
                break
            consumed += len(raw)
            digest.update(raw)
            line_number += 1
            try:
                row = json.loads(raw)
            except ValueError:
                if consumed == limit and not raw.endswith(b'\n'):
                    break
                raise
            if row.get('direction') != 'ue_to_python':
                continue
            packet = row.get('payload')
            if not isinstance(packet, dict):
                continue
            if packet.get('propertyArr'):
                env._build_roster(packet)
            if packet.get('dataArr'):
                step = packet['dataGlobal']['timeCnt']
                if previous is None or step < previous:
                    episode += 1
                    env.episode = episode
                    current = None
                    rule = None
                    if not selected or episode in selected:
                        current = {'episode': episode, 'frames': 0,
                                   'commands': Counter(), 'objectives': [],
                                   'uav_fire_without_local_contact': 0,
                                   'invalid_vehicle_fire': 0,
                                   'pre_hq_diversion_frames': 0,
                                   'column_hold_commands': 0, 'entry_hold_commands': 0,
                                   'max_stalled_units_excluded': 0, 'entry_waves': [],
                                   'vehicle_frames_with_eligible_targets': 0,
                                   'vehicle_commands': Counter(),
                                   'initial_scout_groups': {}, 'initial_vehicle_front': [], 'initial_vehicle_reserve': {},
                                   'recorded_scores': []}
                        rounds.append(current)
                        print(f'validating episode {episode}', flush=True)
                previous = step
                if current is None:
                    continue
                snapshot = env._parse_state(packet)
                state = env.build_team_state(snapshot, 0)
                if rule is None:
                    rule = ReinforceAgentAlgorithm(AlgorithmContext(
                        0, state.agents, state.visible_opponents, ActionSetFactory.standard(),
                        'cpu', {}, state.commanders))
                actions = rule.act(state)
                assert len(actions) == len(state.agents)
                if current['frames'] == 0:
                    current['initial_scout_groups'] = dict(rule._uav._split_objectives)
                    current['initial_vehicle_front'] = sorted(rule._formation.front)
                    current['initial_vehicle_reserve'] = dict(rule._formation.reserve)
                objective = rule._route.objective(state)
                objective_uid = objective.uid if objective else None
                if objective is not None and not rule._route.hq_secured and not rule._route.is_hq(objective):
                    current['pre_hq_diversion_frames'] += 1
                current['column_hold_commands'] += len(rule._column.held)
                current['entry_hold_commands'] += len(rule._entry_waves.held)
                current['max_stalled_units_excluded'] = max(current['max_stalled_units_excluded'], len(rule._column.excluded))
                for release_step, ids in rule._entry_waves.waves:
                    wave = {'step': release_step, 'uids': list(ids)}
                    if wave not in current['entry_waves']:
                        current['entry_waves'].append(wave)
                if not current['objectives'] or current['objectives'][-1]['uid'] != objective_uid:
                    current['objectives'].append({'step': step, 'uid': objective_uid,
                                                 'name': objective.raw.get('rSVD1') if objective else None})
                for unit, action in zip(state.agents, actions):
                    action.to_ue_string(unit.uid, snapshot.entities)
                    current['commands'][action.command] += 1
                    if not unit.alive:
                        assert action.command == 'Idle'
                    if unit.entity_type in rule.VEHICLES and unit.alive:
                        current['vehicle_commands'][action.command] += 1
                        eligible = {enemy.uid for enemy in rule._vehicle_fire.eligible(state, unit)}
                        current['vehicle_frames_with_eligible_targets'] += bool(eligible)
                        if action.command == 'NormalAttacking':
                            if action.target_uid not in eligible:
                                current['invalid_vehicle_fire'] += 1
                    if unit.entity_type == 'BP_Base_UAV_C':
                        target_uid = action.target_uid
                        if action.command == 'SelfDestruct':
                            mission = rule._uav._missions.get(unit.uid)
                            target_uid = mission[0] if mission else None
                        if action.command in {'NormalAttacking', 'SelfDestruct'}:
                            if target_uid not in unit.perceived_uids():
                                current['uav_fire_without_local_contact'] += 1
                for command in rule.take_selected_special_commands():
                    command.parameters()
                current['frames'] += 1
            if current is not None:
                score = _score_packet(packet, line_number)
                if score:
                    current['recorded_scores'].append({key: score[key] for key in
                        ('side', 'kill_score', 'capture_score', 'total')})
    if candidate_hashes() != hashes:
        raise RuntimeError('Candidate source changed during validation; rerun after edits finish.')
    return {'source': str(path), 'source_bytes': consumed, 'source_sha256': digest.hexdigest(),
            'candidate_file_sha256': hashes, 'episodes': rounds,
            'limitation': 'Commands on historical public observations; no simulation of new scores, movement, or survival.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--episodes', type=int, nargs='*', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    result = validate(args.source, args.episodes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    uav_failures = sum(r['uav_fire_without_local_contact'] for r in result['episodes'])
    vehicle_failures = sum(r['invalid_vehicle_fire'] for r in result['episodes'])
    priority_failures = sum(r['pre_hq_diversion_frames'] for r in result['episodes'])
    failures = uav_failures + vehicle_failures + priority_failures
    print(f"episodes={len(result['episodes'])} frames={sum(r['frames'] for r in result['episodes'])} "
          f"invalid_uav_fire={uav_failures} invalid_vehicle_fire={vehicle_failures} pre_hq_diversions={priority_failures}")
    return 1 if failures or not result['episodes'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
