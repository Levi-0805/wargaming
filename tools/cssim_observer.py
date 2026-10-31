"""External evidence recorder: reads logs; only director/zoom viewer inputs.

Never edits match source/config, commits Git, commands a unit, or calls an AI API.
Kept outside runtime source while a version-bound round is in progress; archive
this source, tests and SHA at the next completed-round review gate.
"""
from __future__ import annotations
import ast
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists() and (p/'PythonCode').is_dir())
sys.path.insert(0, str(ROOT/'tools'))
import batch_training as files
from autotrain_runtime import SessionLock
from cssim_campaign import WindowsProcesses, same_process

OUT = ROOT/'logs/adaptive-observer'
LEARNING = ROOT/'logs/learning/state.json'
CAR = {'BP_MNWS_Vehicle_Armored_C', 'BP_MNWS_Vehicle_6x6UGV_C'}
FOOT = {'BP_BaseSoldier_C', 'BP_RoboDog_C'}
GROUND = CAR | FOOT
ANCHORS = {'HQ':(-41810,32275), 'B':(-45450,-30945), 'A':(-3000,-27530)}


def distance(a,b): return math.dist(a[:2],b[:2])
def read(path): return files.read_json(Path(path))
def compact(unit):
    return {k:unit.get(k) for k in ('uid','label','team','type','alive','hp','position','velocity','previous_action','request','protocol_sample')}


def movement_request(wire, position):
    if not wire or ('::Moving;' not in wire and '::Guard;;Points=' not in wire): return False
    if 'Points=' not in wire: return '::Moving;' in wire
    try:
        points=ast.literal_eval(wire.split('Points=',1)[1].rsplit(';',1)[0])
        return len(points)>=3 and distance(position,points[:3])>1500
    except (ValueError,SyntaxError,TypeError): return False


def ammo(raw):
    values=raw.get('interaction')
    if not isinstance(values,list) or not values: return None
    try: values=[float(x) for x in values]
    except (TypeError,ValueError): return None
    return sum(values) if all(math.isfinite(v) and v>=0 for v in values) else None


class Evidence:
    def __init__(self, output):
        self.out=Path(output);self.out.mkdir(parents=True,exist_ok=True)
        self.ids=set()
        if (self.out/'events.jsonl').exists():
            with (self.out/'events.jsonl').open(encoding='utf-8') as f:
                for line in f: self.ids.add(json.loads(line)['id'])
        self.previous={};self.watches={};self.units={};self.frame=None
        self.recent=deque(maxlen=9);self.history=deque(maxlen=8)
        self.objectives={};self.proto={};self.proto_order=deque()
        self.statistics={};self.counts=Counter();self.recent_events=deque(maxlen=18)
        self.last_event={};self.urgent_until=0.;self.last_critical_at=None
        self.event_serial=0;self.batch=None;self.context=None;self.pending_after=[]
        self.initial_ground=None;self.last_live_event=None

    def emit(self,kind,frame,detail,severity='warning',*,backfill=False,unit=None,cooldown=0):
        key=(frame['batch_id'],frame['episode'],kind,unit or detail.get('objective','group'))
        if frame['step']-self.last_event.get(key,-100000)<cooldown:return
        self.last_event[key]=frame['step']
        token='|'.join(map(str,(*key,frame['step'])))
        identity=hashlib.sha256(token.encode()).hexdigest()[:20]
        if identity in self.ids:return
        self.ids.add(identity);self.event_serial+=1;self.counts[kind]+=1
        event=dict(id=identity,kind=kind,severity=severity,frame=dict(frame),detail=detail,
                   recorded_at=files.now(),backfilled=backfill,
                   limitation='Diagnostic candidate, not an established cause or proof of weapon effect.')
        selected=[compact(u) for u in self.units.values() if u.get('team')==0 and
                  (u.get('uid')==unit or u.get('type') in CAR)]
        if unit in self.units:
            point=self.units[unit]['position']
            selected += [compact(u) for u in sorted(self.units.values(),key=lambda u:distance(u['position'],point))
                         if u.get('alive')][:12]
        evidence=dict(event=event,selected_units=selected[:24],unit_context_frame=self.frame,
                      previous_frames=list(self.history)[-4:],context=self.context)
        path=self.out/'evidence'/f'{identity}.json'
        files.write_json(path,evidence)
        event['evidence']=str(path)
        with (self.out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
        self.recent_events.append(event)
        if not backfill:
            self.last_live_event=event
            if severity=='critical':
                self.urgent_until=max(self.urgent_until,time.monotonic()+120)
                self.last_critical_at=files.now()
            if severity in ('critical','warning'):
                self.pending_after.append((dict(frame),identity,unit))

    def protocol_frame(self,payload,batch,episode,*,backfill=False):
        data=payload.get('dataGlobal',{});step=data.get('timeCnt')
        if not isinstance(step,(int,float)):return
        frame=dict(batch_id=batch,episode=episode,step=step)
        sample={}
        for raw in payload.get('dataArr',[]):
            sample[raw['uId']]={k:raw.get(k) for k in ('interaction','weaponCD','agentPerception','agentAlive','rSVD1')}
        key=(batch,episode,step)
        self.proto[key]=sample;self.proto_order.append(key)
        while len(self.proto_order)>600:self.proto.pop(self.proto_order.popleft(),None)
        for raw in data.get('keyObjArr',[]):
            uid=raw.get('uId');location=raw.get('location') or {}
            point=(location.get('x',0),location.get('y',0))
            label=min(ANCHORS,key=lambda name:distance(point,ANCHORS[name]))
            if distance(point,ANCHORS[label])>1000:label=str(uid)
            try:
                percent=float(raw['percent']);red=float(raw['redTeamNum']);blue=float(raw['blueteamNum'])
                if not all(math.isfinite(v) and v>=0 for v in (percent,red,blue)):continue
            except (KeyError,TypeError,ValueError):continue
            slot=self.objectives.setdefault((batch,episode,uid),dict(hold=0,held=False,fall=0,empty=0,last=None))
            explicit=next((raw[k] for k in ('team','agentTeam','ownerTeam') if raw.get(k) is not None),None)
            ours=(explicit==0) if explicit is not None else percent>=.999 and red>blue
            slot['hold']=slot['hold']+1 if ours else 0
            now=dict(percent=percent,red=red,blue=blue,explicit_owner=explicit,step=step)
            if slot['hold']>=3 and not slot['held']:
                slot['held']=True
                self.emit('point_held_observed',frame,dict(objective=label,observation=now,
                          ownership_evidence='explicit_owner' if explicit is not None else 'full_progress_and_red_advantage'),
                          'info',backfill=backfill)
            if slot['held']:
                reversed_bar=explicit==1 or percent<.95
                slot['fall']=slot['fall']+1 if reversed_bar else 0
                slot['empty']=slot['empty']+1 if red==0 else 0
                if slot['fall']==3:
                    self.emit('held_point_progress_reversed',frame,dict(objective=label,before=slot['last'],after=now,
                        hypothesis='Check loss of holding / contested progress; an absent owner field prevents definitive ownership attribution.'),
                        'critical',backfill=backfill,cooldown=40)
                    slot['held']=False;slot['hold']=0
                elif slot['empty']==4:
                    self.emit('holding_without_red_occupants',frame,dict(objective=label,observation=now),
                              'critical' if label=='HQ' else 'warning',backfill=backfill,cooldown=40)
            slot['last']=now

    def simulation_frame(self,row,batch,*,backfill=False):
        if not row.get('entities'):return
        frame=dict(batch_id=batch,episode=row['episode'],step=row['step'])
        same=self.frame and (self.frame['batch_id'],self.frame['episode'])==(batch,row['episode'])
        if not same:
            self.previous={};self.watches={};self.recent.clear();self.history.clear();self.statistics={};self.pending_after=[]
        self.batch=batch
        wires={}
        for text in row.get('action_strings',[]):
            try:wires[int(text.rsplit(';',1)[1])]=text
            except (ValueError,IndexError):pass
        samples=self.proto.get((batch,row['episode'],row['step']),{})
        units={u['uid']:dict(u) for u in row['entities']}
        self.units=units;self.frame=frame
        red=[u for u in units.values() if u['team']==0]
        ground=[u for u in red if u['type'] in GROUND and u['alive']]
        if not same:self.initial_ground=len(ground)
        lost=[];hp_loss=0
        for u in red:
            uid=u['uid'];wire=wires.get(uid);u['request']=wire
            raw=samples.get(uid)
            if raw is not None:
                u['label']=(raw.get('rSVD1') or '').split(';',1)[0]
                u['protocol_sample']=dict(step=row['step'],ammunition=ammo(raw),weapon_cd=raw.get('weaponCD'),
                                          local_contacts=raw.get('agentPerception'))
            prior=self.previous.get(uid)
            watch=self.watches.setdefault(uid,dict(stall=0,fire=0,target=None))
            stat=self.statistics.setdefault(uid,dict(uid=uid,type=u['type'],moving_frames=0,waypoint_guard_frames=0,
                directional_guard_frames=0,explicit_fire_frames=0,ammo_used_observed=0.,ammo_observed_frames=0,
                target_hp_drop_seen=0.,stall_flags=0,ineffective_flags=0,lost_at_step=None))
            if wire and '::Guard;;Points=' in wire:stat['waypoint_guard_frames']+=1
            elif wire and '::Guard;' in wire:stat['directional_guard_frames']+=1
            if wire and '::NormalAttacking;' in wire:stat['explicit_fire_frames']+=1
            if prior is None:continue
            if prior.get('alive') and not u['alive']:
                stat['lost_at_step']=row['step']
                if u['type'] in GROUND:lost.append(uid)
                self.emit('red_unit_lost',frame,dict(uid=uid,type=u['type'],before=compact(prior),after=compact(u)),
                          'critical' if u['type'] in CAR else 'warning',backfill=backfill,unit=uid)
            drop=max(0.,float(prior.get('hp') or 0)-float(u.get('hp') or 0))
            if u['type'] in GROUND:hp_loss+=drop
            current_ammo=(u.get('protocol_sample') or {}).get('ammunition')
            previous_ammo=(prior.get('protocol_sample') or {}).get('ammunition')
            if current_ammo is not None:
                stat['ammo_observed_frames']+=1
                if previous_ammo is not None:stat['ammo_used_observed']+=max(0.,previous_ammo-current_ammo)
            moving=u['alive'] and u['type'] in GROUND and movement_request(wire,u['position'])
            if moving:stat['moving_frames']+=1
            contiguous=same and row['step']==self.previous_step+1
            stationary=distance(u['position'],prior['position'])<25
            watch['stall']=watch['stall']+1 if contiguous and moving and stationary else 0
            if watch['stall']==24:
                stat['stall_flags']+=1
                self.emit('possible_movement_stall',frame,dict(uid=uid,frames=24,request=wire,
                    displacement_threshold_cm=25,protocol_sample=u.get('protocol_sample')),
                    'warning',backfill=backfill,unit=uid,cooldown=60)
            pieces=wire.split(';') if wire else []
            target=int(pieces[1]) if len(pieces)>2 and pieces[0].endswith('NormalAttacking') and pieces[1].isdigit() else None
            enemy=units.get(target);old_enemy=self.previous.get(target)
            if target is not None and enemy and old_enemy:
                damage=max(0.,float(old_enemy['hp'])-float(enemy['hp']))
                stat['target_hp_drop_seen']+=damage
                no_damage=enemy['alive'] and damage==0 and watch['target']==target and contiguous
            else:no_damage=False
            watch['fire']=watch['fire']+1 if no_damage else 0;watch['target']=target
            if watch['fire']==8:
                stat['ineffective_flags']+=1
                self.emit('possible_ineffective_fire_request',frame,dict(uid=uid,target=target,frames=8,
                    target_position=enemy['position'],target_hp=enemy['hp'],request=wire,
                    protocol_sample=u.get('protocol_sample'),
                    caution='Requested fire and target HP correlation do not establish actual shot execution or attribution.'),
                    'warning',backfill=backfill,unit=uid,cooldown=40)
        self.recent.append(dict(step=row['step'],losses=lost,hp_loss=hp_loss))
        window=[item for item in self.recent if row['step']-item['step']<=8]
        deaths=[uid for item in window for uid in item['losses']]
        damage=sum(item['hp_loss'] for item in window)
        if len(deaths)>=4 or damage>=300:
            self.emit('rapid_red_ground_attrition',frame,dict(lost_uids=deaths,hp_loss=damage,
                window_steps=8,red_ground_alive=len(ground)), 'critical',backfill=backfill,cooldown=40)
        if self.initial_ground and self.initial_ground>=6 and len(ground)<=max(2,self.initial_ground*.25):
            self.emit('red_ground_force_depleted',frame,dict(initial_ground=self.initial_ground,
                red_ground_alive=len(ground)), 'critical',backfill=backfill,cooldown=10000)
        for anchor,point in ANCHORS.items():
            near=[u['uid'] for u in ground if distance(u['position'],point)<20000]
            if len(near)>=4:
                self.emit('ground_group_near_objective',frame,dict(objective=anchor,near_uids=near),
                          'info',backfill=backfill,cooldown=10000)
        totals=dict(red_ground_alive=len(ground),red_alive=sum(u['alive'] for u in red),hp_loss=hp_loss,
                    unit_positions={u['uid']:u['position'] for u in ground[:12]})
        self.history.append(dict(frame=frame,totals=totals))
        for origin,identity,uid in list(self.pending_after):
            if origin['batch_id']==batch and origin['episode']==row['episode'] and row['step']>=origin['step']+12:
                files.write_json(self.out/'evidence'/f'{identity}-after.json',dict(frame=frame,
                    selected_units=[compact(u) for u in units.values() if u['uid']==uid or u['team']==0 and u['type'] in CAR],
                    totals=totals))
                self.pending_after.remove((origin,identity,uid))
        self.previous={uid:dict(u) for uid,u in units.items()};self.previous_step=row['step']
        if row.get('done'):
            files.write_json(self.out/'episodes'/f'{batch}-ep{row["episode"]:02d}.json',dict(frame=frame,
                context=self.context,unit_statistics=list(self.statistics.values()),
                final_red_units=[compact(u) for u in red],limitations=self.summary()['limitations']))

    def summary(self):
        return dict(updated_at=files.now(),frame=self.frame,events_total=len(self.ids),new_events_this_process=self.event_serial,
            event_counts=dict(self.counts),urgent=time.monotonic()<self.urgent_until,
            poll_seconds=5 if time.monotonic()<self.urgent_until else 20,
            desired_codex_minutes=5 if time.monotonic()<self.urgent_until else 15,
            last_critical_at=self.last_critical_at,recent_events=list(self.recent_events)[-8:],
            last_live_event=self.last_live_event,
            red_ground_alive=sum(u['team']==0 and u['type'] in GROUND and u['alive'] for u in self.units.values()),
            limitations=['Flags are hypotheses; full director observations never feed the policy.',
                        'Blue bot Python Idle is not interpreted as executed enemy inactivity.',
                        'Absent ammunition/cooldown is unknown, not zero.'])


class Incremental:
    def __init__(self):self.offsets={};self.episodes=defaultdict(int);self.previous={};self.reset={}
    def lines(self,path):
        key=str(path)
        with Path(path).open('rb') as f:
            f.seek(self.offsets.get(key,0))
            for _ in range(12000):
                line=f.readline()
                if not line or not line.endswith(b'\n'):break
                self.offsets[key]=f.tell()
                yield json.loads(line)
    def protocol(self,path,engine,batch,backfill):
        key=str(path)
        for row in self.lines(path):
            p=row.get('payload')
            if row.get('direction')!='ue_to_python' or not isinstance(p,dict):continue
            if p.get('propertyArr'):self.reset[key]=True
            if not p.get('dataArr'):continue
            step=p.get('dataGlobal',{}).get('timeCnt')
            if step is None:continue
            if key not in self.previous or self.reset.get(key) or step<self.previous[key]:self.episodes[key]+=1
            self.reset[key]=False;self.previous[key]=step
            engine.protocol_frame(p,batch,self.episodes[key],backfill=backfill)


def write_digest(engine,learning,view=None):
    summary=engine.summary();summary.update(round_number=learning['round_number'],round_limit=10)
    number=learning['round_number']
    summary['desired_codex_minutes']=(15 if summary['urgent'] else 30) if number<=2 else (30 if summary['urgent'] else 60) if number<=5 else 120
    if view:summary['latest_view']=view
    elif (OUT/'latest-view.json').exists():summary['latest_view']=read(OUT/'latest-view.json')
    files.write_json(OUT/'summary.json',summary)
    files.write_json(OUT/'latest-units.json',dict(frame=engine.frame,units=list(engine.units.values()),
        vehicle_and_unit_statistics=list(engine.statistics.values()),
        scope='Diagnostic full simulation, not the red policy observation; command is a request, not proof of execution.'))
    labels={'rapid_red_ground_attrition':'短时间大面积受损','red_ground_force_depleted':'地面兵力严重不足',
            'held_point_progress_reversed':'已占据点进度回落，需核查失守','holding_without_red_occupants':'占领区缺少红方驻守',
            'possible_movement_stall':'移动/巡逻指令下疑似停滞','possible_ineffective_fire_request':'连续攻击请求但目标未掉血'}
    lines=[f'前10轮重点观察：第{learning["round_number"]}轮',f'更新：{summary["updated_at"]}',
           '当前帧：'+str(summary['frame']),f'当前红方存活地面单位：{summary["red_ground_alive"]}',
           f'脚本采样间隔：{summary["poll_seconds"]}秒；重大事件加密2分钟，平稳后回到20秒。',
           '最新待核查事件（告警不等同于根因）：']
    for event in summary['recent_events']:
        lines.append(f'{labels.get(event["kind"],event["kind"])}；局/步{event["frame"]["episode"]}/{event["frame"]["step"]}；'
                     f'级别{event["severity"]}；历史补录={event["backfilled"]}；证据{event["evidence"]}')
    lines += ['', '提前拟定的排查方案（不在对局中发布）：',
        '据点进度回落：对齐驻守人数、主力转场与敌方回流时间，先核查实际留守和支援距离。',
        '集中减员：对齐首辆车、步兵/机器狗到达与敌方接敌位置，检查车辆支援是否脱节和拐角逐个暴露。',
        '停滞/攻击无效：区分巡逻目的地、警戒方向、弹药与冷却、目标高度及动作回显；不把攻击请求当成命中。',
        '只有对应证据经复核后才修改；源码和模型保持到同版本累计满十局。',
        '截图使用导演视角及单位标签密集区近景；未提供所有者字段时据点丢失仅为待核查信号。']
    folder=Path(learning['reports_dir']);folder.mkdir(exist_ok=True,parents=True)
    (folder/'重点观察-实时摘要.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    if summary['urgent']:
        files.write_json(OUT/'attention.json',dict(updated_at=files.now(),frame=engine.frame,
            event=engine.last_live_event,suggested_codex_minutes=summary['desired_codex_minutes']))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with SessionLock(OUT/'observer.lock'):
        source_sha=files.sha(Path(__file__))
        files.write_json(OUT/'launch.json',dict(started_at=files.now(),identity=WindowsProcesses().identity(os.getpid()),
            script=str(Path(__file__).resolve()),script_sha256=source_sha,first_rounds=10,
            normal_poll_seconds=20,alert_poll_seconds=5,alert_duration_seconds=120,
            normal_view_seconds=600,alert_view_seconds=60,max_sets_per_game=6,
            git_head=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            camera_helper_sha256=files.sha(Path(__file__).with_name('cssim_observer_camera.py')),ocr_helper_sha256=files.sha(Path(__file__).with_name('cssim_observer_ocr.ps1')),
            no_ai_api_calls=True,no_source_or_config_writes=True))
        from cssim_observer_camera import Viewer as FocusViewer
        engine=Evidence(OUT);reader=Incremental();viewer=FocusViewer(OUT)
        visited=set();errors=0;last_campaign=None
        while not (OUT/'STOP').exists() and not (ROOT/'logs/learning/STOP').exists():
            try:
                learning=read(LEARNING)
                if learning['round_number']>10 or learning['phase'] in ('stopped','needs_attention'):break
                if learning['phase'] not in ('collecting',):time.sleep(20);continue
                state=read(ROOT/'logs/campaigns'/learning['active_campaign']/'state.json')
                batch=state.get('current_attempt');protocol=state.get('runtime',{}).get('batch_status',{}).get('protocol')
                if state['state']!='collecting' or not batch or not protocol:time.sleep(10);continue
                path=Path(protocol);simulation=path.with_name('simulation.jsonl')
                if not simulation.exists():time.sleep(5);continue
                engine.context=dict(campaign=learning['active_campaign'],round_number=learning['round_number'],
                    source_commit=state['source_commit'],protocol=str(path),observer_sha256=source_sha)
                backfill=batch not in visited
                reader.protocol(path,engine,batch,backfill)
                for row in reader.lines(simulation):engine.simulation_frame(row,batch,backfill=backfill)
                visited.add(batch)
                view=None
                fresh=time.time()-simulation.stat().st_mtime<30
                if fresh:view=viewer.capture(learning,state,engine,force=backfill)
                write_digest(engine,learning,view)
                errors=0
            except Exception as exc:
                errors+=1
                files.write_json(OUT/'last-error.json',dict(at=files.now(),error=f'{type(exc).__name__}: {exc}',consecutive=errors))
                if errors>=8:break
            time.sleep(5 if time.monotonic()<engine.urgent_until else 20)
        files.write_json(OUT/'finished.json',dict(at=files.now(),frame=engine.frame,errors=errors,
            reason='stop/round_limit/collection_attention' if errors<8 else 'repeated_observer_errors'))


if __name__=='__main__':main()
