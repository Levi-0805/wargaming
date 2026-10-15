"""Replay extracted CSSIM public frames through the policy, without UE physics.

Usage: python tools/validate_cohesion_escort.py logs/<run>-stuck.json --output <json>
Input is the existing local protocol extractor's frames/rosters/scores format.
Never treat these decisions as a new match or a prediction of score/survival.
"""
import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'PythonCode'))

from cssim.config import RuntimeConfig
from cssim.environment.env import CssimEnvironment
from cssim.algorithms import AlgorithmContext
from cssim.protocol import ActionSetFactory
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RuleAgentAlgorithm import ReinforceAgentAlgorithm


def validate(source, baseline='ef538dcd7cd700e4d3fff7d83b8d112aa44387a3'):
    data = json.loads(source.read_text(encoding='utf-8'))
    out = {'source': data['source'], 'baseline_commit': baseline,
           'scope': 'Recorded public team views only. No UE physics, counterfactual score or survival prediction.',
           'recorded_scores': [], 'rounds': []}
    for row in data['scores'][::2]:
        score = row['score']
        kills = sum(v['SumWorth'] for v in score['Score'].values())
        out['recorded_scores'].append({'episode': row['ep'], 'kill_score': kills,
                                       'capture_score': score['Reward'], 'total': kills + score['Reward']})
    episodes = sorted({f['ep'] for f in data['frames']})
    for ep in episodes:
        env = CssimEnvironment(RuntimeConfig(ROOT, {}, '127.0.0.1', 21051, ('red', 'blue'),
                                            500, len(episodes), 4, 60, .5, 0, 'cpu', 'WARNING'))
        env.episode = ep
        env._build_roster(data['rosters'][ep-1])
        rule = None
        counts, air, streaks = Counter(), Counter(), Counter()
        last_station = {}
        first_flight = {}
        max_wait, release_step = 0, None
        starts = []
        for frame in data['frames']:
            if frame['ep'] != ep:
                continue
            snapshot = env._parse_state(frame['packet'])
            state = env.build_team_state(snapshot, 0)
            if rule is None:
                rule = ReinforceAgentAlgorithm(AlgorithmContext(0, state.agents, state.visible_opponents,
                                                ActionSetFactory.standard(), 'cpu', {}, state.commanders))
            rule._prepare(state)
            assembly = {u.uid: rule._cohesion.choose(state, u, rule._route) for u in state.agents}
            actions = rule.act(state)
            counts['frames'] += 1
            if rule._cohesion.released and release_step is None:
                release_step = state.step
            for unit, action in zip(state.agents, actions):
                action.to_ue_string(unit.uid, snapshot.entities)
                counts['serialized_actions'] += 1
                if not unit.alive:
                    continue
                waiting = assembly[unit.uid]
                streaks[unit.uid] = streaks[unit.uid]+1 if waiting is not None else 0
                max_wait = max(max_wait, streaks[unit.uid])
                if waiting is not None:
                    counts['assembly_unit_frames'] += 1
                    assert waiting.command == 'Guard' and waiting.points == tuple(unit.position)
                    assert not state.perceived_opponents(unit)
                garrison = rule._route.garrison_for(state, unit)
                if garrison and rule._route.inside_garrison(garrison, unit.position) and action.command == 'Moving':
                    assert action.points and rule._route.inside_garrison(garrison, action.points)
                if unit.entity_type != rule._UAV:
                    continue
                air[action.command] += 1
                obj = rule._route.objective(state)
                station = rule._uav._stations.get(unit.uid)
                key = (unit.uid, obj.uid if obj else None)
                if station:
                    assert key not in last_station or station[1] == last_station[key]
                    last_station[key] = station[1]
                    if action.command == 'Moving' and action.points and key not in first_flight:
                        distance = rule._horizontal(action.points, obj.position) / 100
                        assert 99 <= distance <= 124
                        first_flight[key] = {'step': state.step, 'uid': unit.uid, 'objective': obj.uid,
                                             'stand_off_m': round(distance, 1)}
                local = {e.uid: e for e in state.perceived_opponents(unit) if e.alive}
                if action.command == 'NormalAttacking':
                    assert action.target_uid in local
                    assert rule._uav._payload(unit) and rule._uav._clear(state, unit, local[action.target_uid])
                if action.command == 'SelfDestruct':
                    uid, point, start = rule._uav._missions[unit.uid]
                    assert uid in local and rule._uav._locked(state, unit, local[uid])
                    assert rule._uav._strike_supported(state, local[uid], obj, rule._route)
                    assert point == tuple(local[uid].position) == action.points
                    if start == state.step:
                        starts.append({'step': state.step, 'uid': unit.uid, 'target': uid})
        assert max_wait <= rule._cohesion.MAX_WAIT
        out['rounds'].append({'episode': ep, 'counts': dict(counts), 'max_assembly_wait': max_wait,
                              'assembly_started': rule._cohesion.started, 'assembly_released': release_step,
                              'release_reason': rule._cohesion.reason, 'uav_actions': dict(air),
                              'supported_dive_starts': starts, 'first_front_flights': list(first_flight.values())})
        print('VALIDATED', ep, dict(counts), 'assembly', rule._cohesion.started, release_step,
              rule._cohesion.reason, 'air', dict(air), flush=True)
    out['source_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (ROOT / 'PythonCode/TeamAlg/FsLutk7wD2RkNgbs/Algorithm').glob('*.py')}
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', default='ef538dcd7cd700e4d3fff7d83b8d112aa44387a3')
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    result = validate(args.source, args.baseline)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
