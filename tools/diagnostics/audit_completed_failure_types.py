"""Read only verified game selections; state flags are evidence, not causal labels."""
import collections, hashlib, json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'logs/learning/failure-taxonomy'
REPORTS=['recover-20261031-round1-last1','learning-20261031-162920-r0002','learning-1101-014143-r0003','learning-1101-014143-r0004']
CAP={'BP_BaseSoldier_C','BP_RoboDog_C'}
GROUND=CAP|{'BP_MNWS_Vehicle_Armored_C','BP_MNWS_Vehicle_6x6UGV_C'}
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def audit(path, games):
    selected={g['episode']:g for g in games}; rows={}; roster={}; ep=0; prev=None
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
      for line in f:
        digest.update(line); record=json.loads(line); p=record.get('payload',{})
        if record.get('direction')!='ue_to_python' or not isinstance(p,dict): continue
        if p.get('propertyArr'): roster={u['uId']:u for u in p['propertyArr']}
        if not p.get('dataArr'): continue
        step=p['dataGlobal']['timeCnt']
        if prev is None or step<prev: ep+=1
        prev=step
        if ep not in selected: continue
        r=rows.setdefault(ep,dict(frames=0,points={},ties=[],loss_windows=[],timeline=[]))
        units=[u for u in p['dataArr'] if u.get('agentAlive') and u.get('agentTeam',roster.get(u['uId'],{}).get('agentTeam'))==0]
        alive=collections.Counter(u.get('type',roster.get(u['uId'],{}).get('type')) for u in units)
        cap=sum(alive[t] for t in CAP); ground=sum(alive[t] for t in GROUND)
        r['frames']+=1
        frame=dict(step=step,capturers=cap,ground=ground,points={})
        for o in p['dataGlobal'].get('keyObjArr',[]):
          name=o.get('rSVD1',str(o.get('uId'))); bar=o.get('percent'); red=o.get('redTeamNum'); blue=o.get('blueteamNum')
          if not all(isinstance(v,(int,float)) for v in (bar,red,blue)): continue
          z=r['points'].setdefault(name,dict(first_red_entry=None,first_full=None,full_count=0,first_reversal=None,last=None,tied_full_positive=0))
          if red>0 and z['first_red_entry'] is None:z['first_red_entry']=step
          if bar>=1:
            z['full_count']+=1
            if z['first_full'] is None:z['first_full']=step
          elif z['first_full'] is not None and z['first_reversal'] is None:z['first_reversal']=step
          if bar>=1 and red==blue and red>0:
            z['tied_full_positive']+=1
            if len(r['ties'])<6:r['ties'].append(dict(step=step,point=name,red=red,blue=blue,bar=bar))
          state=dict(bar=bar,red=red,blue=blue);z['last']=state;frame['points'][name]=state
        r['timeline'].append(frame)
        if len(r['timeline'])>20:
          old=r['timeline'][-21]
          if old['capturers']>=6 and old['capturers']-cap>=max(5,math.ceil(old['capturers']*.3)):
            if not r['loss_windows'] or step-r['loss_windows'][-1]['end']>20:
              r['loss_windows'].append(dict(start=old['step'],end=step,before=old['capturers'],after=cap,points=frame['points']))
    sha=digest.hexdigest()
    for g in games:
      assert sha==g['protocol_sha256'], 'Protocol changed or provenance mismatch'
    result=[]
    for e,g in selected.items():
      r=rows[e];hq=next((v for k,v in r['points'].items() if k=='指挥所'),None)
      flags=[]
      if hq:
        if hq['first_full'] is None:flags.append('HQ_full_red_bar_never_observed')
        elif hq['first_reversal'] is not None:flags.append('HQ_full_bar_then_progress_reversed')
        if hq['first_full'] is not None and hq['first_reversal'] is None and g['red']['capture_score']<3200: flags.append('HQ_bar_retained_secondary_capture_incomplete')
      if r['loss_windows']:flags.append('rapid_capturer_loss_observed')
      stalled=[]
      for name,v in r['points'].items():
        if name=='指挥所':continue
        seq=[f for f in r['timeline'] if f['points'].get(name,{}).get('red',0)>0 and f['points'][name]['bar']<1]
        # Consecutive occupied/unfinished frames: dwell, not proof of pathfinding/fire bug.
        longest=run=0;last=None
        for f in seq:
          run=run+1 if last is not None and f['step']==last+1 else 1;last=f['step'];longest=max(longest,run)
        if longest>=60:stalled.append(dict(point=name,consecutive_frames=longest))
      if stalled:flags.append('secondary_point_long_occupied_but_incomplete')
      result.append(dict(ordinal=g['ordinal'],episode=e,protocol=str(path),protocol_sha256=sha,source_commit=g['source_commit'],model_sha256=g.get('model_sha256'),score=g['red'],flags=flags,points=r['points'],rapid_losses=r['loss_windows'][:5],tied_full_examples=r['ties'],secondary_dwell=stalled,frames=r['frames']))
    return result
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rounds=[]
    for n,ident in enumerate(REPORTS,1):
      report=read(ROOT/'results/batches'/f'{ident}-cumulative.json'); grouped=collections.defaultdict(list)
      for g in report['games']:grouped[g['protocol']].append(g)
      games=[]
      for path,gs in grouped.items():games.extend(audit(path,gs))
      games.sort(key=lambda x:x['ordinal']);assert len(games)==10
      rounds.append(dict(round=n,report=f'results/batches/{ident}-cumulative.json',games=games,counts=dict(collections.Counter(flag for g in games for flag in g['flags']))))
      (OUT/f'round-{n:04d}.json').write_text(json.dumps(rounds[-1],ensure_ascii=False,indent=2),encoding='utf-8')
    result=dict(rounds=rounds,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limits=['Only confirmed selections; no running game is read.','Bar=1 is observed full red progress, not proof of secure ownership or cleared enemies.','Reversal is progress reversal, not necessarily final loss.','Loss-window thresholds and dwell are diagnostic flags, not established algorithm defects.','No counterfactual returns or paired defender control.'])
    (OUT/'completed.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':
    try:main()
    except Exception:
      import traceback
      (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
      raise
