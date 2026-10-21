"""Attribute observed losses to HQ phases, without inventing damage sources."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from analyze_objective_outcomes import KINDS, objective_name


def distance(a, b):
    return math.dist(a, b)


def analyze(outcomes_path):
    outcomes = json.loads(outcomes_path.read_text(encoding='utf-8'))
    source = Path(outcomes['source'])
    limit = outcomes['source_bytes']
    consumed, previous_step, episode = 0, None, 0
    digest, roster, previous, commands = hashlib.sha256(), {}, {}, {}
    rounds = []
    with source.open('rb') as stream:
        while consumed < limit:
            line = stream.readline(limit - consumed)
            if not line:
                break
            consumed += len(line)
            digest.update(line)
            row = json.loads(line)
            p = row.get('payload')
            if not isinstance(p, dict):
                continue
            if row.get('direction') == 'python_to_ue' and p.get('StringActions'):
                commands = {int(a.rsplit(';', 1)[1]): a.split(';')[0].split('::')[-1]
                            for a in p['StringActions']}
                continue
            if row.get('direction') != 'ue_to_python':
                continue
            if p.get('propertyArr'):
                roster = {u['uId']: u for u in p['propertyArr']}
            if not p.get('dataArr'):
                continue
            step = p['dataGlobal']['timeCnt']
            if previous_step is None or step < previous_step:
                summary = outcomes['rounds'][episode]
                episode += 1
                previous, commands = {}, {}
                current = dict(episode=episode, complete=summary['complete'], score=summary['score'],
                               phase_deaths={}, hq_damage=Counter(), deaths=[], snapshots=[])
                rounds.append(current)
                events = summary['first']
                entry = events.get('HQ_entry', {}).get('step', math.inf)
                full = events.get('HQ_full_bar', {}).get('step', math.inf)
                secure = events.get('HQ_secure', {}).get('step', math.inf)
                sample_steps = {-1, 100, 125, 150, 175, 200, 250, entry, full, secure,
                                entry + 20, entry + 40, entry + 60}
            previous_step = step
            hq = next((o for o in p['dataGlobal'].get('keyObjArr', ()) if objective_name(o) == 'HQ'), None)
            if hq is None:
                raise ValueError('HQ missing from observed state')
            center = tuple(hq['location'][k] for k in 'xyz')
            units = {}
            for u in p['dataArr']:
                uid, old = u['uId'], previous.get(u['uId'], {})
                meta = roster.get(uid, {})
                pos = u.get('agentLocation', {})
                unit = dict(uid=uid, name=meta.get('className'),
                            team=u.get('agentTeam', meta.get('agentTeam')),
                            kind=KINDS.get(u.get('type', meta.get('type')), 'other'),
                            alive=bool(u.get('agentAlive')), hp=float(u.get('agentHp', 0)),
                            position=tuple(pos.get(k, old.get('position', (0, 0, 0))[i]) for i, k in enumerate('xyz')),
                            ammo=u.get('interaction', []), local=u.get('agentPerception', []))
                units[uid] = unit
                if not old or not old['alive'] or unit['team'] != 0:
                    continue
                phase = ('before_entry' if step < entry else 'entry_to_full' if step < full
                         else 'full_to_secure' if step <= secure else 'after_secure')
                if not unit['alive']:
                    bucket = current['phase_deaths'].setdefault(phase, Counter())
                    bucket[unit['kind']] += 1
                    current['deaths'].append(dict(step=step, uid=uid, name=unit['name'], kind=unit['kind'],
                                                  phase=phase, distance_hq_m=round(distance(old['position'], center)/100, 2),
                                                  position=old['position'], prior_command=commands.get(uid),
                                                  prior_ammo=old['ammo'], prior_local=old['local']))
                loss = max(0, old['hp'] - (unit['hp'] if unit['alive'] else 0))
                if (loss and unit['kind'] in ('soldier', 'dog') and step <= secure
                        and distance(old['position'], center) <= 18000):
                    stats = current['hq_damage']
                    stats['hp_lost'] += loss
                    stats['hit_unit_steps'] += 1
                    stats['command_' + commands.get(uid, 'unknown')] += 1
                    local_enemies = [v for v in old['local'] if v in previous and previous[v]['team'] == 1 and previous[v]['alive']]
                    stats['with_local_enemy' if local_enemies else 'without_local_enemy'] += 1
                    cars = [v for v in previous.values() if v['team'] == 0 and v['alive'] and v['kind'] in ('wheel', 'lynx')]
                    if not cars or min(distance(old['position'], v['position']) for v in cars) > 6000:
                        stats['no_live_vehicle_within_60m'] += 1
            if step in sample_steps:
                live = [u for u in units.values() if u['alive'] and u['team'] == 0]
                cars = [u for u in live if u['kind'] in ('wheel', 'lynx')]
                current['snapshots'].append(dict(step=step, red_foot=sum(u['kind'] in ('soldier', 'dog') for u in live),
                    red_cars=len(cars), red_uavs=sum(u['kind']=='uav' for u in live),
                    hq_red=hq.get('redTeamNum'), hq_blue=hq.get('blueteamNum'),
                    cars=[dict(uid=u['uid'],kind=u['kind'],distance_hq_m=round(distance(u['position'],center)/100,2),
                               hp=u['hp'],ammo=u['ammo']) for u in cars]))
            previous = units
    if consumed != limit or digest.hexdigest() != outcomes['source_sha256']:
        raise RuntimeError('Source snapshot differs from outcomes; refusing mismatched analysis.')
    return dict(source=str(source),source_bytes=consumed,source_sha256=digest.hexdigest(),source_commit=None,
                rounds=rounds, limitations='Observed HP losses and alive-to-dead transitions only. Phase is temporal, not killer attribution. Prior commands do not prove physical firing or absence of autonomous Guard fire. 60m vehicle proximity is descriptive, not a proven support range.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outcomes', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.outcomes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('rounds=',len(result['rounds']),'source_sha256=',result['source_sha256'])
