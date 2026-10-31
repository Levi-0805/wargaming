"""Frozen-state decision audit; never controls a live unit or alters source."""
import collections
import hashlib
import json
import logging
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'PythonCode'), str(ROOT/'tools')]
from cssim.algorithms import AlgorithmContext
from cssim.environment.env import CssimEnvironment
from cssim.protocol import ActionSetFactory
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.LearnedTargeting import HybridIQLAgent
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RLAgentAlgorithm import AlgorithmConfig
from train_target_iql import config

BASE=ROOT/'logs/autotrain/learning-20261031-162920-r0002-0001/logs/20261031_205034'
OUT=ROOT/'logs/learning/hq-battle-audit.json'
DETAIL=OUT.with_suffix('.jsonl')
FOOT={'BP_BaseSoldier_C','BP_RoboDog_C'}
CAR={'BP_MNWS_Vehicle_6x6UGV_C','BP_MNWS_Vehicle_Armored_C'}

def run():
    logging.disable(logging.CRITICAL)
    observed={}
    with (BASE/'simulation.jsonl').open(encoding='utf-8-sig') as f:
        for line in f:
            r=json.loads(line)
            if r.get('entities') and r['episode'] in {1,3,6,8}:
                observed[r['episode'],r['step']]=r
    env=CssimEnvironment(config())
    protocol=BASE/'protocol.jsonl'
    limit=protocol.stat().st_size
    previous=None;episode=0;rule=None;consumed=0
    summaries={}
    with protocol.open('rb') as stream, DETAIL.open('w',encoding='utf-8') as detail:
        while consumed<limit:
            line=stream.readline(limit-consumed);consumed+=len(line)
            row=json.loads(line);p=row.get('payload',{})
            if row.get('direction')!='ue_to_python' or not isinstance(p,dict):continue
            if p.get('propertyArr'):env._build_roster(p)
            if not p.get('dataArr'):continue
            step=p['dataGlobal']['timeCnt']
            if previous is None or step<previous:
                episode+=1;rule=None;env.episode=episode
            previous=step
            if episode not in {1,3,6,8}:continue
            snap=env._parse_state(p);state=env.build_team_state(snap,0)
            if rule is None:
                context=AlgorithmContext(0,state.agents,state.visible_opponents,ActionSetFactory.standard(),'cpu',{},state.commanders)
                rule=HybridIQLAgent(context,AlgorithmConfig(mode='hybrid_iql',iql_model=str(ROOT/'PythonCode/TeamAlg/FsLutk7wD2RkNgbs/Models/target_iql_v1.json'),iql_shadow=False))
                summaries[episode]=dict(frames=0,compared_commands=0,different_commands=0,transitions=[],timeline=[])
            actions=rule.act(state)
            wire={a.uid:act.to_ue_string(a.uid,snap.entities) for a,act in zip(state.agents,actions)}
            # The simulation logs the commands that produced the *next* state.
            # Compare a decision from protocol step N with simulation N+1.
            observed_frame=observed.get((episode,step+1),{})
            actual={int(s.rsplit(';',1)[1]):s for s in observed_frame.get('action_strings',[])}
            summary=summaries[episode];summary['frames']+=1
            summary['compared_commands']+=len(wire.keys()&actual.keys())
            summary['different_commands']+=sum(v!=actual[k] for k,v in wire.items() if k in actual)
            route=rule._route;hq=state.key_object_by_uid(route.hq_uid)
            ground=[a for a in state.agents if a.alive and a.entity_type in FOOT]
            vehicles=[a for a in state.agents if a.alive and a.entity_type in CAR]
            def unit(a):
                return dict(uid=a.uid,name=a.raw.get('rSVD1',''),type=a.entity_type,hp=float(a.hp),
                            position=[round(float(x),1) for x in a.position],
                            ammo=a.raw.get('interaction'),communication=a.communication_ok(),
                            hq_distance=round(route.distance(a.position,hq.position)),
                            mobile_foot_distance=min((round(route.distance(a.position,f.position)) for f in ground if f.uid not in route._garrison),default=None),
                            request=actual.get(a.uid),replay_request=wire.get(a.uid))
            guards=[a for a in ground if route._garrison.get(a.uid)==hq.uid]
            hq_units=[a for a in ground if route.inside_garrison(hq,a.position)]
            current=dict(step=step,stage=route.stage,objective=route.objective(state).uid,
                         hq_secured=route.hq_secured,bar=hq.raw.get('percent'),red=hq.raw.get('redTeamNum'),
                         blue=hq.raw.get('blueteamNum'),red_foot=len(ground),red_car=len(vehicles),
                         hq_foot=len(hq_units),column_enabled=rule.enable_continuous_column,
                         column_held=list(rule._column.held),formation_front=list(rule._formation.front),
                         guards=[unit(a) for a in guards],
                         vehicles=[unit(a) for a in vehicles])
            detail.write(json.dumps(dict(episode=episode,**current),ensure_ascii=False)+'\n')
            if step%10==0:summary['timeline'].append({k:v for k,v in current.items() if k not in ('guards','vehicles')})
            transition=(current['stage'],current['objective'],current['hq_secured'])
            if not summary['transitions'] or tuple(summary['transitions'][-1]['key'])!=transition:
                summary['transitions'].append(dict(key=transition,**current))
    progress=json.loads((BASE.parents[1]/'progress.json').read_text(encoding='utf-8-sig'))
    scores=[]
    for r in progress['score_summary']['rounds']:
        packets=[]
        for p in r.get('score_packets',[]):
            by_killer=collections.Counter()
            for kill in p.get('kills',[]):
                killer=kill['killer'].rsplit('_',1)[0];by_killer[killer]+=1
            packets.append(dict(side=p['side'],kill=p['kill_score'],capture=p['capture_score'],total=p['total'],
                                kills_by_shooter_type=dict(by_killer),victim_score_by_type=p.get('breakdown')))
        scores.append(dict(episode=r['episode'],packets=packets))
    result=dict(recorded_source_commit='8ac55f2131c72a2725c0cef10db06cdb6b9c9f30',
                replay_policy_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'PythonCode/TeamAlg/FsLutk7wD2RkNgbs/Algorithm').glob('*.py')},
                replay_model_sha256=hashlib.sha256((ROOT/'PythonCode/TeamAlg/FsLutk7wD2RkNgbs/Models/target_iql_v1.json').read_bytes()).hexdigest(),
                protocol_bytes=limit,replay_not_new_games=True,action_alignment='protocol N -> simulation N+1',
                summaries=summaries,scores=scores,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__ == '__main__':
    try:run()
    except Exception:OUT.with_suffix('.error.txt').write_text(traceback.format_exc(),encoding='utf-8')
