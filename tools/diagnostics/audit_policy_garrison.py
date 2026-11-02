"""Explain guards on verified completed public states; replay is not a match."""
import argparse
from collections import defaultdict
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'PythonCode'),str(ROOT/'tools')]
from cssim.algorithms import AlgorithmContext
from cssim.environment.env import CssimEnvironment
from cssim.protocol import ActionSetFactory
from train_target_iql import config
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RLAgentAlgorithm import ReinforceAgentAlgorithm

def numeric(value):
    if hasattr(value,'item'):return value.item()
    raise TypeError(type(value).__name__)

def audit(report, ordinals, output, *, baseline='6cb5a79'):
    games=json.loads(Path(report).read_text(encoding='utf-8-sig'))['games']
    groups=defaultdict(list)
    for game in games:
        if game['ordinal'] in ordinals:groups[game['protocol']].append(game)
    if sum(map(len,groups.values()))!=len(ordinals):raise ValueError('Missing selected game')
    results=[];output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix('.jsonl').open('w',encoding='utf-8') as details:
        for source,selected in groups.items():
            chosen={g['episode']:g for g in selected};observed={}
            with Path(source).with_name('simulation.jsonl').open(encoding='utf-8-sig') as stream:
                for line in stream:
                    row=json.loads(line)
                    if row.get('episode') in chosen and 'action_strings' in row:
                        observed[row['episode'],row['step']]=row['action_strings']
            env=CssimEnvironment(config());previous=None;episode=0;policy=None;summary=None
            prior=set();prior_key=None;digest=hashlib.sha256()
            with Path(source).open('rb') as stream:
                for line in stream:
                    digest.update(line);row=json.loads(line);p=row.get('payload',{})
                    if row.get('direction')!='ue_to_python' or not isinstance(p,dict):continue
                    if p.get('propertyArr'):env._build_roster(p)
                    if not p.get('dataArr'):continue
                    step=p['dataGlobal']['timeCnt']
                    if previous is None or step<previous:
                        episode+=1;policy=None;env.episode=episode;prior=set();prior_key=None
                    previous=step
                    if episode not in chosen:continue
                    snapshot=env._parse_state(p);state=env.build_team_state(snapshot,0)
                    if policy is None:
                        options=dict(mode='hybrid_iql',policy_baseline=baseline,iql_shadow=False)
                        policy=ReinforceAgentAlgorithm(AlgorithmContext(0,state.agents,state.visible_opponents,
                            ActionSetFactory.standard(),'cpu',options,state.commanders))
                        summary=dict(ordinal=chosen[episode]['ordinal'],episode=episode,source=source,
                            source_commit=chosen[episode]['source_commit'],score=chosen[episode]['red'],
                            frames=0,compared=0,mismatches=0,changed_by_type={},command_changes=[],changes=[],transitions=[])
                        results.append(summary)
                    actions=policy.act(state);policy.take_selected_special_commands();route=policy._route
                    hq=state.key_object_by_uid(route.hq_uid)
                    wire={u.uid:a.to_ue_string(u.uid,snapshot.entities) for u,a in zip(state.agents,actions)}
                    actual={int(a.rsplit(';',1)[1]):a for a in observed.get((episode,step+1),[])}
                    summary['frames']+=1;summary['compared']+=len(wire.keys()&actual.keys())
                    summary['mismatches']+=sum(wire[k]!=actual[k] for k in wire.keys()&actual.keys())
                    for unit in state.agents:
                        if unit.uid in actual and wire[unit.uid] != actual[unit.uid]:
                            kind=unit.entity_type
                            summary['changed_by_type'][kind]=summary['changed_by_type'].get(kind,0)+1
                            summary['command_changes'].append(dict(step=step,uid=unit.uid,type=kind,
                                garrison=route._garrison.get(unit.uid),requested=wire[unit.uid],
                                actual_next=actual[unit.uid]))
                    guards={uid for uid,obj in route._garrison.items() if obj==hq.uid}
                    agents={u.uid:u for u in state.agents}
                    def unit(uid):
                        u=agents.get(uid)
                        if u is None:return dict(uid=uid,present=False)
                        return dict(uid=uid,type=u.entity_type,alive=u.alive,hp=float(u.hp),
                            link=u.communication_ok(),inside=route.inside_garrison(hq,u.position),
                            distance=round(route.distance(u.position,hq.position)),
                            requested=wire.get(uid),actual_next=actual.get(uid))
                    record=dict(ordinal=chosen[episode]['ordinal'],step=step,stage=route.stage,
                        objective=route.objective(state).uid,hq_secured=route.hq_secured,
                        bar=hq.raw.get('percent'),red=hq.raw.get('redTeamNum'),blue=hq.raw.get('blueteamNum'),
                        remnant=route._remnant.active,needed=route.garrison_needed(state,hq),
                        guards=[unit(uid) for uid in sorted(guards)],removed=[unit(uid) for uid in sorted(prior-guards)])
                    details.write(json.dumps(record,ensure_ascii=False,default=numeric)+'\n')
                    if prior!=guards:summary['changes'].append(record)
                    key=(route.stage,record['objective'],record['hq_secured'])
                    if key!=prior_key:summary['transitions'].append(record)
                    prior=guards;prior_key=key
            for game in selected:
                if digest.hexdigest()!=game['protocol_sha256']:raise ValueError('Completed protocol SHA mismatch')
    result=dict(baseline=baseline,episodes=results,
        limitation='Fixed public-state replay; N+1 command equivalence required before interpreting decisions. No new returns.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=numeric)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path);parser.add_argument('--ordinals',nargs='+',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline',choices=('6cb5a79','6cb5a79_hq_hold', '6cb5a79_hq_guard_entry'),default='6cb5a79')
    args=parser.parse_args();logging.disable(logging.CRITICAL)
    result=audit(args.report,set(args.ordinals),args.out,baseline=args.baseline)
    print(json.dumps([dict(ordinal=x['ordinal'],compared=x['compared'],mismatches=x['mismatches']) for x in result['episodes']]))
