"""Read-only comparison of public capture counters and diagnostic positions."""
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'logs/learning/capture-geometry-audit.json'
BASE = ROOT / 'logs/autotrain/learning-20261031-162920-r0002-0001/logs/20261031_205034'

def run():
    decoder = json.JSONDecoder()
    snapshots = {}
    previous, episode = None, 0
    protocol = BASE / 'protocol.jsonl'
    size = protocol.stat().st_size
    consumed = 0
    with protocol.open('rb') as stream:
        while consumed < size:
            line = stream.readline(size-consumed)
            consumed += len(line)
            if b'"ue_to_python"' not in line or b'"dataArr"' not in line:
                continue
            match = re.search(rb'"timeCnt"\s*:\s*(\d+)', line)
            index = line.find(b'"keyObjArr"')
            if match is None or index < 0:
                continue
            step = int(match[1])
            if previous is None or step < previous:
                episode += 1
            previous = step
            # Decode only the tiny capture array, avoiding the full distance matrix.
            tail = line[index:].decode('utf-8')
            snapshots[episode, step] = decoder.raw_decode(tail, tail.index('['))[0]
    variants = [(factor, inset, z, rotated)
                for factor in (0.5, 1.0)
                for inset in (0, 400)
                for z in (False, True)
                for rotated in (False, True)]
    metrics = collections.defaultdict(lambda: [0, 0, 0, 0])
    summaries = collections.defaultdict(lambda: {'frames':0,'full_frames':0,'red_zero_full_frames':0,
                                               'lost_progress_steps':[], 'examples':[]})
    types = collections.Counter()
    frame_count = 0
    previous_bars = {}
    with (BASE / 'simulation.jsonl').open(encoding='utf-8-sig') as stream:
        for line in stream:
            frame = json.loads(line)
            if 'entities' not in frame:
                continue
            ep, step = frame['episode'], frame['step']
            objects = snapshots.get((ep, step))
            if objects is None:
                continue
            frame_count += 1
            units = [unit for unit in frame['entities'] if unit['alive']]
            if step == 0:
                types.update(unit['type'] for unit in units)
            eligible = [u for u in units if 'Soldier' in u['type'] or 'Dog' in u['type']]
            for obj in objects:
                oid = obj['uId']
                loc, scale = obj['location'], obj['scale']
                yaw = math.radians(obj.get('rotation',{}).get('yaw',0))
                projected = []
                for u in eligible:
                    p = u['position']
                    dx, dy, dz = p[0]-loc['x'], p[1]-loc['y'], p[2]-loc['z']
                    projected.append((u, dx, dy, dz, dx*math.cos(yaw)+dy*math.sin(yaw),
                                      -dx*math.sin(yaw)+dy*math.cos(yaw)))
                for factor, inset, check_z, rotated in variants:
                    counts = collections.Counter()
                    for u, dx, dy, dz, rx, ry in projected:
                        x,y = (rx,ry) if rotated else (dx,dy)
                        if (abs(x)<=scale['x']*factor-inset and abs(y)<=scale['y']*factor-inset
                            and (not check_z or abs(dz)<=scale['z']*factor)):
                            counts[u['team']] += 1
                    for team, field in ((0,'redTeamNum'),(1,'blueteamNum')):
                        actual = obj.get(field,0)
                        # Focus on nonempty observations; avoid travel zeros inflating fit.
                        if not actual and not counts[team]:
                            continue
                        key = f'{oid}:team{team}:factor{factor}:inset{inset}:z{check_z}:rotate{rotated}'
                        m = metrics[key]
                        m[0] += 1; m[1] += abs(actual-counts[team]); m[2] += actual==counts[team]
                        m[3] += counts[team]-actual
                s = summaries[ep,oid]
                bar = obj.get('percent',0)
                s['frames'] += 1
                s['full_frames'] += bar >= 1
                s['red_zero_full_frames'] += bar>=1 and obj.get('redTeamNum',0)==0
                oldbar = previous_bars.get((ep,oid),0)
                if oldbar>=1 and bar<1:
                    s['lost_progress_steps'].append(step)
                previous_bars[ep,oid] = bar
                # The code's own footprint predicate, without Z, with 4m inset.
                claimed = [dict(uid=u['uid'],type=u['type'],position=u['position'],
                                offset=[round(rx),round(ry),round(dz)])
                           for u,dx,dy,dz,rx,ry in projected
                           if u['team']==0 and abs(rx)<=scale['x']-400 and abs(ry)<=scale['y']-400]
                if oid==32769 and obj.get('redTeamNum',0)==0 and claimed and len(s['examples'])<3 and step>100:
                    s['examples'].append(dict(step=step,bar=bar,server_red=0,server_blue=obj.get('blueteamNum'),
                                              code_inside=claimed))
    ranked = collections.defaultdict(list)
    for k,(n,error,exact,bias) in metrics.items():
        scope, variant = k.split(':factor',1)
        ranked[scope].append(dict(variant='factor'+variant,n=n,mae=round(error/n,4),
                                  exact=round(exact/n,4),bias=round(bias/n,4)))
    for values in ranked.values():
        values.sort(key=lambda v:(v['mae'],-v['exact']))
    result = dict(read_only=True,source_commit='8ac55f2131c72a2725c0cef10db06cdb6b9c9f30',
                  model_sha256='3e36fb18875258f58065cfa748b57c8947675cded629654ce3b79696a8863f4f',
                  protocol=str(protocol),protocol_snapshot_bytes=size,frames=frame_count,
                  caveats=['Position/counter lag may affect instantaneous fit.',
                           'Diagnostic full state is never fed into the red policy.',
                           'Candidate volume hypotheses are not platform specification.'],
                  eligible_types=dict(types),fits=dict(ranked),
                  summaries={f'{ep}:{oid}':v for (ep,oid),v in summaries.items()},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__ == '__main__':
    try:
        run()
    except Exception:
        OUT.with_suffix('.error.txt').write_text(traceback.format_exc(),encoding='utf-8')
