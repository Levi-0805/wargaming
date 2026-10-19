"""Measure observed arrivals and surviving units; never simulate new scores."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

SAMPLES = {0, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 499}
KINDS = {'BP_BaseSoldier_C': 'soldier', 'BP_RoboDog_C': 'dog',
         'BP_MNWS_Vehicle_Armored_C': 'wheeled', 'BP_MNWS_Vehicle_6x6UGV_C': 'lynx',
         'BP_Base_UAV_C': 'uav'}


def audit(source):
    rounds, current, previous, roster = [], None, None, {}
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
            if row.get('direction') != 'ue_to_python' or not isinstance(packet, dict):
                continue
            if packet.get('propertyArr'):
                roster = {u['uId']: u for u in packet['propertyArr']}
                previous = None
                current = None
            if not packet.get('dataArr'):
                continue
            step = packet['dataGlobal']['timeCnt']
            if previous is None or step < previous:
                current = {'episode': len(rounds) + 1, 'frames': 0, 'first_hq_entry': None,
                           'first_hq_clear_full': None, 'max_hq_red': 0,
                           'max_hq_red_when_contested': 0, 'first_red_entry': {}, 'samples': [],
                           'hq_arrival_samples': [], 'uav_zero_step': None}
                rounds.append(current)
            previous = step
            current['frames'] += 1
            alive = {0: Counter(), 1: Counter()}
            for unit in packet['dataArr']:
                known = roster.get(unit['uId'], {})
                kind = KINDS.get(unit.get('type') or known.get('type'))
                team = unit.get('agentTeam', known.get('agentTeam'))
                if kind and team in alive and unit.get('agentAlive'):
                    alive[team][kind] += 1
            objects = packet['dataGlobal'].get('keyObjArr', [])
            hq = next((o for o in objects if o.get('rSVD1') == '指挥所'), None)
            hq_state = {k: hq.get(k) for k in ('percent', 'redTeamNum', 'blueteamNum')} if hq else None
            sample = {'step': step, 'red_alive': dict(alive[0]), 'blue_alive': dict(alive[1]), 'hq': hq_state}
            for obj in objects:
                name = obj.get('rSVD1', str(obj.get('uId')))
                if obj.get('redTeamNum', 0) > 0 and name not in current['first_red_entry']:
                    current['first_red_entry'][name] = step
            if hq:
                red, blue = hq.get('redTeamNum', 0), hq.get('blueteamNum', 0)
                current['max_hq_red'] = max(current['max_hq_red'], red)
                if blue > 0:
                    current['max_hq_red_when_contested'] = max(current['max_hq_red_when_contested'], red)
                event = False
                if red > 0 and current['first_hq_entry'] is None:
                    current['first_hq_entry'], event = step, True
                if red > 0 and blue == 0 and hq.get('percent', 0) >= 1 and current['first_hq_clear_full'] is None:
                    current['first_hq_clear_full'], event = step, True
                if event:
                    current['hq_arrival_samples'].append(sample)
            if alive[0]['uav'] == 0 and current['uav_zero_step'] is None:
                current['uav_zero_step'] = step
            if step in SAMPLES:
                current['samples'].append(sample)
    return {'source': str(source), 'source_bytes': consumed, 'source_sha256': digest.hexdigest(),
            'limitations': ['Full HQ bar plus positive red and zero blue is an observed clear snapshot, not a new-policy score.',
                           'UAV disappearance includes destroyed and self-destructing aircraft; cause is not inferred.'],
            'rounds': rounds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    results = []
    for source in args.sources:
        result = audit(source)
        results.append(result)
        print(source.parent.name, 'rounds', len(result['rounds']), 'HQ clears',
              sum(r['first_hq_clear_full'] is not None for r in result['rounds']), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'runs': results}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
