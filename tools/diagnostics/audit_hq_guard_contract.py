import argparse, hashlib, importlib.util, json, logging, sys, types
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description='Frozen HQ guard contract comparison; not new match outcomes.')
parser.add_argument('--baseline-route',type=Path,required=True)
parser.add_argument('--candidate-route',type=Path,default=ROOT/'PythonCode/TeamAlg/FsLutk7wD2RkNgbs/Algorithm/AssaultRoute.py')
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--episode',type=int,default=1)
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();OUT=args.out.parent;OUT.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(ROOT/'PythonCode'),str(ROOT/'PythonCode/tests'),str(ROOT/'tools')]
from test_adaptive_assault import opening, advance
from test_fslutk_rule import guards_at, state
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.LearnedTargeting import HybridIQLAgent
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RLAgentAlgorithm import AlgorithmConfig
from cssim.algorithms import AlgorithmContext
from cssim.environment.env import CssimEnvironment
from cssim.protocol import ActionSetFactory
from train_target_iql import config
spec=importlib.util.spec_from_file_location('TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRouteCandidate',args.candidate_route)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
base_spec=importlib.util.spec_from_file_location('TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRouteBaseline',args.baseline_route)
baseline=importlib.util.module_from_spec(base_spec);base_spec.loader.exec_module(baseline)
def setup(patched):
    _,objects,route=opening();units=guards_at(objects[0],count=8)
    objects[0].raw.update(percent=1,redTeamNum=8,blueteamNum=0)
    advance(route,units,objects,range(4));assert route.stage==1 and len(route._garrison)==4
    route._assign_garrison=types.MethodType((mod if patched else baseline).AssaultRoute._assign_garrison,route)
    return units,objects,route
def checks():
    cases=[]
    for patched in [False,True]:
      units,objects,route=setup(patched);old=dict(route._garrison)
      objects[0].raw.update(percent=1,redTeamNum=4,blueteamNum=4)
      s=state(units,objectives=objects,step=4);route._assign_garrison(s)
      assert len(route._garrison)==(4 if patched else 0)
      assert not route.owned(s,objects[0]);assert route.stage==1
      if patched:assert route._garrison==old
      cases.append(dict(case='tie_preserves_only_existing_guards',variant=patched,guards=len(route._garrison)))
    for case in ['below_full','explicit_enemy_owner','no_previous','dead_guard']:
      units,objects,route=setup(True);old=dict(route._garrison)
      objects[0].raw.update(percent=1,redTeamNum=4,blueteamNum=4)
      if case=='below_full':objects[0].raw['percent']=.9
      if case=='explicit_enemy_owner':objects[0]=replace(objects[0],team=1)
      if case=='no_previous':route._garrison={}
      if case=='dead_guard':
        dead=next(u for u in units if u.uid in old);units=[replace(u,alive=False) if u.uid==dead.uid else u for u in units]
      route._assign_garrison(state(units,objectives=objects,step=4))
      assert len(route._garrison)==(3 if case=='dead_guard' else 0)
      cases.append(dict(case=case,guards=len(route._garrison)))
    return cases
def replay():
    path=args.source
    env=CssimEnvironment(config());rules=None;prev=None;ep=0;rows=[];changes=[];safety=dict(invalid_vehicle_fire=0,pre_hq_diversions=0,changed_air_commands=0);objective_changes=0
    for line in path.open('rb'):
      rec=json.loads(line);p=rec.get('payload',{})
      if rec.get('direction')!='ue_to_python' or not isinstance(p,dict):continue
      if p.get('propertyArr'):env._build_roster(p)
      if not p.get('dataArr'):continue
      step=p['dataGlobal']['timeCnt']
      if prev is None or step<prev:ep+=1;env.episode=ep
      prev=step
      if ep>args.episode:break
      if ep!=args.episode:continue
      snap=env._parse_state(p);s=env.build_team_state(snap,0)
      if rules is None:
        ctx=AlgorithmContext(0,s.agents,s.visible_opponents,ActionSetFactory.standard(),'cpu',{},s.commanders)
        cfg=AlgorithmConfig(mode='hybrid_iql',iql_model=str(args.model),iql_shadow=False)
        rules=[HybridIQLAgent(ctx,cfg),HybridIQLAgent(ctx,cfg)]
        rules[0]._route._assign_garrison=types.MethodType(baseline.AssaultRoute._assign_garrison,rules[0]._route)
        rules[1]._route._assign_garrison=types.MethodType(mod.AssaultRoute._assign_garrison,rules[1]._route)
      wires=[];guards=[]
      for rule in rules:
        acts=rule.act(s);wires.append({u.uid:a.to_ue_string(u.uid,snap.entities) for u,a in zip(s.agents,acts)});rule.take_selected_special_commands()
        guards.append(dict(rule._route._garrison))
        if rule is rules[1]:
          objective=rule._route.objective(s)
          safety['pre_hq_diversions'] += bool(objective and not rule._route.hq_secured and not rule._route.is_hq(objective))
          for unit,action in zip(s.agents,acts):
            if unit.entity_type in rule.VEHICLES and action.command=='NormalAttacking':
              safety['invalid_vehicle_fire'] += action.target_uid not in {e.uid for e in rule._vehicle_fire.eligible(s,unit)}
      diff=[dict(uid=u,baseline=wires[0][u],candidate=wires[1][u]) for u in wires[0] if wires[0][u]!=wires[1][u]]
      types_by_uid={u.uid:u.entity_type for u in s.agents}
      safety['changed_air_commands'] += sum(types_by_uid[x['uid']] in {'BP_Base_UAV_C','BP_Scout_UAV_C','BP_Helicopter_C'} for x in diff)
      objective_changes += rules[0]._route.objective(s).uid != rules[1]._route.objective(s).uid
      if diff:changes.append(dict(step=step,changes=diff))
      if 213<=step<=223:
        hq=s.key_object_by_uid(rules[0]._route.hq_uid)
        rows.append(dict(step=step,bar=hq.raw.get('percent'),red=hq.raw.get('redTeamNum'),blue=hq.raw.get('blueteamNum'),guards=guards,changed=diff))
    return dict(source=str(path),episode=args.episode,timeline=rows,safety=safety,objective_changes=objective_changes,changed_frames=len(changes),changed_commands=sum(len(x['changes']) for x in changes),changes=changes[:20],limitation='Same frozen states, two policies. No simulated consequences or new scores; baseline includes current B join change, not exact historical source.')
if __name__=='__main__':
  logging.disable(logging.CRITICAL)
  try:
    result=dict(cases=checks(),replay=replay(),candidate_sha=hashlib.sha256(args.candidate_route.read_bytes()).hexdigest(),baseline_sha=hashlib.sha256(args.baseline_route.read_bytes()).hexdigest(),model_sha=hashlib.sha256(args.model.read_bytes()).hexdigest())
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
  except Exception:
    import traceback
    args.out.with_suffix('.error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    raise
