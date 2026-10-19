"""Attribute recorded orders to formation gates; ablations are not new battles."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'PythonCode'))
from cssim.algorithms import AlgorithmContext
from cssim.config import RuntimeConfig
from cssim.environment.env import CssimEnvironment
from cssim.protocol import ActionSetFactory
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RuleAgentAlgorithm import ReinforceAgentAlgorithm


class Probe:
    def __init__(self, state, mode):
        self.rule = ReinforceAgentAlgorithm(AlgorithmContext(
            0, state.agents, state.visible_opponents, ActionSetFactory.standard(), 'cpu', {}, state.commanders))
        self.reason = {}
        self.counts = Counter()
        self.vehicle_counts = Counter()
        self.holds_under_fire = Counter()
        self.streaks, self.max_streaks = {}, {}
        self.pending = {}
        self.matches = self.mismatches = 0
        self.samples = []
        for attr, name in (('_vehicle_fire', 'fire'), ('_formation', 'formation'),
                           ('_cohesion', 'opening'), ('_hq_entry', 'entrance_route')):
            control = getattr(self.rule, attr)
            original = control.choose
            def choose(state, agent, *args, _original=original, _name=name, **kwargs):
                if mode == 'no_static_reserve' and _name == 'formation':
                    assigned = self.rule._formation.reserve.pop(agent.uid, None)
                    try:
                        result = _original(state, agent, *args, **kwargs)
                    finally:
                        if assigned is not None:
                            self.rule._formation.reserve[agent.uid] = assigned
                else:
                    result = _original(state, agent, *args, **kwargs)
                if result is not None:
                    self.reason[agent.uid] = _name
                return result
            control.choose = choose
        if mode == 'no_entry_waves':
            self.rule._entry_waves.begin = lambda *a: None
        for attr, name in (('_motion_recovery', 'recovery'), ('_entry_waves', 'entry_wait'), ('_column', 'column_wait')):
            control = getattr(self.rule, attr)
            original = control.apply
            def apply(state, agent, action, *args, _original=original, _name=name, **kwargs):
                if mode == 'no_column' and _name == 'column_wait':
                    if action.command == 'Moving':
                        self.rule._column._moved.add(agent.uid)
                    return action
                result = _original(state, agent, action, *args, **kwargs)
                if result != action:
                    self.reason[agent.uid] = _name
                return result
            control.apply = apply

    def act(self, state, snapshot):
        self.reason = {}
        actions = self.rule.act(state)
        self.rule.take_selected_special_commands()
        self.pending = {u.uid: a.to_ue_string(u.uid, snapshot.entities) for u,a in zip(state.agents, actions)}
        sample = []
        for unit, action in zip(state.agents, actions):
            if not unit.alive:
                continue
            reason = self.reason.get(unit.uid, 'navigation_or_combat')
            self.counts[reason] += 1
            if unit.entity_type not in self.rule.VEHICLES:
                continue
            self.vehicle_counts[reason] += 1
            holding = reason in {'column_wait', 'entry_wait', 'formation'} and action.command == 'Guard'
            self.streaks[unit.uid] = self.streaks.get(unit.uid, 0)+1 if holding else 0
            self.max_streaks[unit.uid] = max(self.max_streaks.get(unit.uid, 0), self.streaks[unit.uid])
            if holding and unit.uid in self.rule._formation._under_fire:
                self.holds_under_fire[reason] += 1
            if state.step % 10 == 0 and state.step <= 210:
                sample.append({'uid': unit.uid, 'type': unit.entity_type, 'hp': unit.hp,
                    'ammo': self.rule._vehicle_fire.ammo(unit), 'pos': [round(float(x)) for x in unit.position],
                    'reason': reason, 'action': self.pending[unit.uid],
                    'eligible': len(self.rule._vehicle_fire.eligible(state,unit)),
                    'progress': self.rule._column.progress.get(unit.uid),
                    'front': unit.uid in self.rule._formation.front,
                    'reserve': self.rule._formation.reserve.get(unit.uid)})
        if sample:
            obj=self.rule._route.objective(state)
            self.samples.append({'step':state.step,'anchor':self.rule._column.anchor,
                'foot_reference':len(self.rule._column.foot),'garrison_count':len(self.rule._route._garrison),
                'entry_waves':list(self.rule._entry_waves.waves),'objective':obj.uid if obj else None,'vehicles':sample})

    def result(self):
        return {'recorded_matches':self.matches,'recorded_mismatches':self.mismatches,
                'all_reasons':dict(self.counts),'vehicle_reasons':dict(self.vehicle_counts),
                'vehicle_holds_under_fire':dict(self.holds_under_fire),
                'max_vehicle_hold_streak':self.max_streaks,'samples':self.samples}


def audit(source, selected, modes):
    config=RuntimeConfig(ROOT,{},'127.0.0.1',21051,('red','blue'),500,1,1.,60.,.5,0,'cpu','WARNING')
    env=CssimEnvironment(config)
    previous=None;episode=0;probes={};rounds=[];consumed=0;digest=hashlib.sha256()
    limit=source.stat().st_size
    with source.open('rb') as stream:
        while consumed<limit:
            line=stream.readline(limit-consumed)
            if not line:break
            consumed+=len(line);digest.update(line)
            row=json.loads(line);packet=row.get('payload',{})
            if not isinstance(packet,dict):continue
            if row.get('direction')=='ue_to_python':
                if packet.get('propertyArr'):env._build_roster(packet)
                if not packet.get('dataArr'):continue
                step=packet['dataGlobal']['timeCnt']
                if previous is None or step<previous:
                    if probes:rounds.append({'episode':episode,'modes':{m:p.result() for m,p in probes.items()}})
                    episode+=1;env.episode=episode;probes={}
                    if episode in selected:print('audit episode',episode,flush=True)
                previous=step
                if episode not in selected:continue
                snapshot=env._parse_state(packet);state=env.build_team_state(snapshot,0)
                if not probes:probes={m:Probe(state,m) for m in modes}
                for probe in probes.values():probe.act(state,snapshot)
            elif row.get('direction')=='python_to_ue':
                for probe in probes.values():
                    for action in packet.get('StringActions',[]) or []:
                        uid=int(action.split(';')[-1])
                        if uid in probe.pending:
                            if action==probe.pending[uid]:probe.matches+=1
                            else:probe.mismatches+=1
                    probe.pending={}
    if probes:rounds.append({'episode':episode,'modes':{m:p.result() for m,p in probes.items()}})
    return {'source':str(source),'source_bytes':consumed,'source_sha256':digest.hexdigest(),
            'checkout_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'candidate_file_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT/'PythonCode/TeamAlg/FsLutk7wD2RkNgbs/Algorithm').glob('*.py'))},
            'limitation':'Fixed recorded observations. Ablations measure changed orders, not causal score changes.',
            'episodes':rounds}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--episodes',type=int,nargs='+',required=True)
    parser.add_argument('--modes',nargs='+',default=['baseline','no_column','no_entry_waves','no_static_reserve'])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();logging.disable(logging.CRITICAL)
    result=audit(args.source,set(args.episodes),args.modes)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for ep in result['episodes']:
        for mode,r in ep['modes'].items():
            print(ep['episode'],mode,{k:v for k,v in r.items() if k!='samples'})
