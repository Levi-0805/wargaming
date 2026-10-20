"""Retrospective round outcomes; full simulator state is never policy input."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from parse_match_score import parse_output

KINDS = {'BP_BaseSoldier_C': 'soldier', 'BP_RoboDog_C': 'dog',
         'BP_MNWS_Vehicle_Armored_C': 'wheel', 'BP_MNWS_Vehicle_6x6UGV_C': 'lynx',
         'BP_Base_UAV_C': 'uav', 'BP_Helicopter_C': 'heli'}


def objective_name(raw):
    name = raw.get('rSVD1', '')
    return 'A' if name.endswith('A') else 'B' if name.endswith('B') else 'HQ'


def brief(step, units, objects):
    live = [u for u in units if u['alive']]
    red = [u for u in live if u['team'] == 0]
    return dict(step=step, objectives=objects,
                red_alive=dict(Counter(u['kind'] for u in red)),
                red_foot=sum(u['kind'] in ('soldier', 'dog') for u in red),
                red_vehicles=[u['uid'] for u in red if u['kind'] in ('wheel', 'lynx')],
                blue_alive=dict(Counter(u['kind'] for u in live if u['team'] == 1)))


def analyze(source):
    scores = parse_output(source)
    digest, consumed, rounds, previous, roster, prior = hashlib.sha256(), 0, [], None, {}, {}
    limit = scores['source_bytes']
    with source.open('rb') as stream:
        while consumed < limit:
            line = stream.readline(limit - consumed)
            if not line:
                break
            digest.update(line)
            consumed += len(line)
            row = json.loads(line)
            p = row.get('payload')
            if row.get('direction') != 'ue_to_python' or not isinstance(p, dict):
                continue
            if p.get('propertyArr'):
                roster = {u['uId']: u for u in p['propertyArr']}
            if not p.get('dataArr'):
                continue
            step = p['dataGlobal']['timeCnt']
            if previous is None or step < previous:
                score = scores['rounds'][len(rounds)]
                red = score['scores']['red']
                current = dict(episode=score['episode'], complete=score['complete'],
                               score={k: red[k] for k in ('kill_score', 'capture_score', 'total')} if red else None,
                               frames=0, first={}, samples=[], vehicle_deaths=[], final=None,
                               cross_team_kill_events={})
                for side, other in (('red', 'blue'), ('blue', 'red')):
                    side_score = score['scores'][side]
                    if side_score is None:
                        continue
                    events = [e for e in side_score['kills']
                              if str(e['killer_side']).lower() == side and str(e['victim_side']).lower() == other]
                    current['cross_team_kill_events'][side] = dict(
                        all=len(events),
                        vehicle=sum('无人车' in str(e['killer']) for e in events),
                        lynx=sum('山猫' in str(e['killer']) for e in events))
                rounds.append(current)
                prior = {}
            previous = step
            current['frames'] += 1
            units = []
            for u in p['dataArr']:
                uid = u['uId']
                meta = roster.get(uid, {})
                unit = dict(uid=uid, team=u.get('agentTeam', meta.get('agentTeam')),
                            kind=KINDS.get(u.get('type', meta.get('type')), 'other'),
                            alive=bool(u.get('agentAlive')))
                old = prior.get(uid)
                if old and old['alive'] and not unit['alive'] and old['kind'] in ('wheel', 'lynx'):
                    current['vehicle_deaths'].append(dict(step=step, uid=uid, team=old['team'], kind=old['kind']))
                units.append(unit)
                prior[uid] = unit
            objects = {objective_name(o): dict(red=o.get('redTeamNum'), blue=o.get('blueteamNum'),
                                               percent=o.get('percent'))
                       for o in p['dataGlobal'].get('keyObjArr', ())}
            sample = brief(step, units, objects)
            for name, o in objects.items():
                # Entry, full progress and enemy-free control are different observations.
                entered = o['red'] is not None and o['red'] > 0
                full = o['percent'] is not None and o['percent'] >= 1
                for event, condition in (
                    ('entry', entered), ('full_bar', full),
                    ('enemy_free', entered and o['blue'] == 0),
                    ('secure', entered and full and o['blue'] == 0),
                ):
                    if condition:
                        current['first'].setdefault(f'{name}_{event}', sample)
            if step in (-1, 0, 50, 100, 125, 150, 175, 200, 250, 300, 400, 499):
                current['samples'].append(sample)
            current['final'] = sample
    if consumed != limit or digest.hexdigest() != scores['source_sha256']:
        raise RuntimeError('Source changed between score and observation reads.')
    if len(rounds) != len(scores['rounds']):
        raise RuntimeError('Score and observation episode boundaries differ.')
    return dict(source=str(source), source_bytes=consumed, source_sha256=digest.hexdigest(),
                source_commit=None, rounds=rounds,
                limitations='Retrospective full-state observations. Enemy-free and full-bar events are not terminal rewards. No counterfactual score prediction or inferred historical Git revision.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"rounds={len(result['rounds'])} source_sha256={result['source_sha256']}")
