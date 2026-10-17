"""Read-only public-view replay, objective progress and recorded-command audit."""
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


def audit(path, selected):
    config = RuntimeConfig(ROOT, {}, '127.0.0.1', 21051, ('red', 'blue'),
                           500, 1, 1., 60., .5, 0, 'cpu', 'WARNING')
    env = CssimEnvironment(config)
    previous, episode, current, rule, pending = None, 0, None, None, {}
    rows, digest = [], hashlib.sha256()
    limit, consumed = path.stat().st_size, 0
    with path.open('rb') as source:
        for line in source:
            if consumed + len(line) > limit:
                break
            consumed += len(line)
            digest.update(line)
            record = json.loads(line)
            packet = record.get('payload')
            if not isinstance(packet, dict):
                continue
            if record.get('direction') == 'ue_to_python':
                if packet.get('propertyArr'):
                    env._build_roster(packet)
                if packet.get('dataArr'):
                    step = packet['dataGlobal']['timeCnt']
                    if previous is None or step < previous:
                        episode += 1
                        env.episode = episode
                        rule, current, pending = None, None, {}
                        if episode in selected:
                            current = {'episode': episode, 'frames': 0, 'matches': 0, 'mismatches': 0,
                                       'goals': [], 'snapshots': [], 'scores': [], 'mismatch_examples': []}
                            rows.append(current)
                            print('audit episode', episode, flush=True)
                    previous = step
                    if current is None:
                        continue
                    snapshot = env._parse_state(packet)
                    state = env.build_team_state(snapshot, 0)
                    if rule is None:
                        rule = ReinforceAgentAlgorithm(AlgorithmContext(0, state.agents, state.visible_opponents,
                            ActionSetFactory.standard(), 'cpu', {}, state.commanders))
                    actions = rule.act(state)
                    pending = {u.uid: a.to_ue_string(u.uid, snapshot.entities) for u, a in zip(state.agents, actions)}
                    rule.take_selected_special_commands()
                    route = rule._route
                    obj = route.objective(state)
                    current['frames'] += 1
                    if obj and (not current['goals'] or current['goals'][-1]['uid'] != obj.uid):
                        current['goals'].append({'step': step, 'uid': obj.uid, 'name': obj.raw.get('rSVD1')})
                    if step % 25 == 0 or step == 499:
                        ground = [u for u in state.agents if u.alive and u.entity_type in route._strategy.GROUND]
                        current['snapshots'].append({'step': step, 'stage': route.stage,
                            'alive': dict(Counter(u.entity_type for u in state.agents if u.alive)),
                            'objects': [{k:o.raw.get(k) for k in ('rSVD1','redTeamNum','blueteamNum','percent')} for o in state.key_objects],
                            'garrison': dict(Counter(route._garrison.values())),
                            'ground': [{'uid': u.uid, 'kind': u.entity_type, 'hp': round(u.hp,1),
                                        'pos': [round(float(v)) for v in u.position],
                                        'link':u.communication_ok(), 'guard':route._garrison.get(u.uid),
                                        'cursor':route._cursor.get(u.uid), 'action':pending[u.uid],
                                        'contact_count':len(state.perceived_opponents(u))} for u in ground]})
                if current is not None:
                    score = _score_packet(packet, 0)
                    if score:
                        current['scores'].append({k:score[k] for k in ('side','kill_score','capture_score','total')})
            elif record.get('direction') == 'python_to_ue' and current is not None:
                for action in packet.get('StringActions', []) or []:
                    uid = int(action.split(';')[-1])
                    if uid not in pending:
                        continue
                    match = action == pending[uid]
                    current['matches' if match else 'mismatches'] += 1
                    if not match and len(current['mismatch_examples']) < 5:
                        current['mismatch_examples'].append({'step':previous,'recorded':action,'generated':pending[uid]})
                pending = {}
    return {'source':str(path),'source_bytes':consumed,'source_sha256':digest.hexdigest(),
            'limitation':'Historical observations, not simulated outcomes of different actions.', 'episodes':rows}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--episodes', type=int, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    logging.disable(logging.CRITICAL)
    result = audit(args.source, set(args.episodes))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    for row in result['episodes']:
        print(json.dumps({k:row[k] for k in ('episode','matches','mismatches','goals','scores')},ensure_ascii=True))
