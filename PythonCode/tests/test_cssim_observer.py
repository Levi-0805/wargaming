import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('adaptive_observer',Path(__file__).resolve().parents[2]/'tools/cssim_observer.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
CAR='BP_MNWS_Vehicle_6x6UGV_C'
FOOT='BP_BaseSoldier_C'


def unit(uid,team=0,hp=100,position=(0,0,0),kind=FOOT):
    return dict(uid=uid,team=team,type=kind,alive=hp>0,hp=hp,position=list(position),velocity=[0,0,0],previous_action=-1)


def row(step,units,wires=(),episode=1):
    return dict(episode=episode,step=step,entities=units,action_strings=list(wires))


def events(engine,kind):return [e for e in engine.recent_events if e['kind']==kind]


def test_guard_movement_distinguished_from_stationary_directional_guard():
    assert module.movement_request('ActionSet2::Guard;;Points=[10000,0,0];1',[0,0,0])
    assert not module.movement_request('ActionSet2::Guard;;Points=[0,0,0];1',[0,0,0])
    assert not module.movement_request('ActionSet2::Guard;+X;1',[0,0,0])
    assert not module.movement_request('ActionSet2::Idle;N/A;1',[0,0,0])


def test_stall_uses_consecutive_frames_and_ignores_stationary_guard(tmp_path):
    engine=module.Evidence(tmp_path)
    for step in range(30):
        engine.simulation_frame(row(step,[unit(1,kind=CAR),unit(2)],
            ['ActionSet2::Guard;;Points=[10000,0,0];1','ActionSet2::Guard;+X;2']),'batch')
    assert len(events(engine,'possible_movement_stall'))==1
    assert events(engine,'possible_movement_stall')[0]['detail']['uid']==1
    assert engine.statistics[2]['stall_flags']==0


def test_episode_reset_never_compares_two_different_rosters_as_casualties(tmp_path):
    engine=module.Evidence(tmp_path)
    engine.simulation_frame(row(499,[unit(1)]),'batch')
    engine.simulation_frame(row(0,[unit(1,hp=0)],episode=2),'batch')
    assert not events(engine,'red_unit_lost')


def test_mass_casualties_trigger_fast_sampling_but_backfill_does_not(tmp_path):
    for backfill in (False,True):
        engine=module.Evidence(tmp_path/str(backfill))
        engine.simulation_frame(row(0,[unit(i) for i in range(6)]),'batch',backfill=backfill)
        engine.simulation_frame(row(1,[unit(i,hp=0 if i<4 else 100) for i in range(6)]),'batch',backfill=backfill)
        assert events(engine,'rapid_red_ground_attrition')
        assert engine.summary()['urgent'] is (not backfill)
        assert engine.summary()['poll_seconds']==(20 if backfill else 5)


def test_fire_request_without_damage_is_a_hypothesis_not_a_confirmed_miss(tmp_path):
    engine=module.Evidence(tmp_path)
    for step in range(12):
        engine.simulation_frame(row(step,[unit(1,kind=CAR),unit(20,team=1,position=(3000,0,0))],
            ['ActionSet2::NormalAttacking;20;1']),'batch')
    result=events(engine,'possible_ineffective_fire_request')
    assert len(result)==1
    assert 'not establish' in result[0]['detail']['caution']
    assert engine.statistics[1]['target_hp_drop_seen']==0


def test_objective_loss_requires_confirmed_prior_hold_and_progress_reversal(tmp_path):
    engine=module.Evidence(tmp_path)
    for step in range(6):
        obj=dict(uId=99,location=dict(x=-41810,y=32275),percent=1 if step<3 else .5,
                 redTeamNum=8 if step<3 else 1,blueteamNum=0 if step<3 else 4)
        engine.protocol_frame(dict(dataGlobal=dict(timeCnt=step,keyObjArr=[obj]),dataArr=[]),'batch',1)
    result=events(engine,'held_point_progress_reversed')
    assert len(result)==1 and result[0]['severity']=='critical'
    assert result[0]['detail']['objective']=='HQ'
    assert 'prevents definitive' in result[0]['detail']['hypothesis']


def test_missing_ownership_values_do_not_mean_red_point_lost(tmp_path):
    engine=module.Evidence(tmp_path)
    engine.protocol_frame(dict(dataGlobal=dict(timeCnt=1,keyObjArr=[dict(uId=1,location={})]),dataArr=[]),'batch',1)
    assert not list(engine.recent_events)


def test_absent_and_invalid_ammo_remains_unknown():
    for raw in ({},{'interaction':[]},{'interaction':['bad']},{'interaction':[float('nan')]}):
        assert module.ammo(raw) is None
    assert module.ammo({'interaction':[0]})==0
    assert module.ammo({'interaction':[10,20]})==30


def test_incomplete_trailing_line_not_consumed_and_no_replay_duplicates(tmp_path):
    path=tmp_path/'data.jsonl';path.write_bytes(b'{"step":1}\n{"step":')
    reader=module.Incremental()
    assert list(reader.lines(path))==[dict(step=1)]
    with path.open('ab') as f:f.write(b'2}\n')
    assert list(reader.lines(path))==[dict(step=2)]
    assert list(reader.lines(path))==[]


def test_event_evidence_survives_restart_without_duplicate_alerts(tmp_path):
    for _ in range(2):
        engine=module.Evidence(tmp_path)
        engine.simulation_frame(row(0,[unit(1)]),'batch')
        engine.simulation_frame(row(1,[unit(1,hp=0)]),'batch')
    lines=(tmp_path/'events.jsonl').read_text(encoding='utf-8').splitlines()
    assert len(lines)==1
    saved=json.loads(lines[0])
    assert Path(saved['evidence']).exists()


def test_native_ocr_normalization_preserves_unit_identity():
    spec=importlib.util.spec_from_file_location('camera_view',Path(__file__).resolve().parents[2]/'tools/cssim_observer_camera.py')
    camera=importlib.util.module_from_spec(spec);spec.loader.exec_module(camera)
    assert camera.normalize('十 兵 一 ] 1')=='士兵11'
    assert camera.normalize('“山猫” 无 人 车 _ 0')=='山猫无人车0'
    assert camera.normalize('士兵_1')!=camera.normalize('士兵_11')
