import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
import cssim_cumulative as module
from cssim_campaign import Settings


def evidence(batch, count):
    manifest = dict(batch_id=batch, source_commit='code-a', model_sha256='model-a')
    progress = dict(status=dict(protocol=batch, done_episodes=list(range(1,count+1))),
                    score_summary=dict(malformed_json_lines=0, source_sha256=batch+'-sha',
                    rounds=[dict(episode=i, complete=True,
                        scores={s:dict(kill_score=100,capture_score=1600,total=1700)
                                for s in ('red','blue')}) for i in range(1,count+1)]))
    return manifest, progress


def test_accumulates_completed_games_without_duplicates_or_incomplete_rounds():
    games=[]
    first,p=evidence('failed-seven',7)
    assert module.append_completed(games,first,p)==7
    assert module.append_completed(games,first,p)==0
    second,p=evidence('remaining-three',4)
    p['score_summary']['rounds'][-1]['complete']=False
    assert module.append_completed(games,second,p)==3
    assert len(games)==10 and [g['ordinal'] for g in games]==list(range(1,11))


def test_cross_version_and_budget_overrun_are_rejected():
    games=[];m,p=evidence('one',7)
    module.append_completed(games,m,p)
    other,p=evidence('two',4)
    with pytest.raises(ValueError,match='budget'):
        module.append_completed(games,other,p)
    other,p=evidence('two',3);other['source_commit']='different'
    with pytest.raises(ValueError,match='different'):
        module.append_completed(games,other,p)
    assert len(games)==7


@pytest.mark.parametrize('counts,expected_code,expected_budgets', [
    ([7,3],0,[10,3]), ([4,4,2],0,[10,6,2]), ([0,0,0],2,[10,10,10])])
def test_controller_retries_only_remaining_games_and_stops_at_total_ten(
        tmp_path,monkeypatch,counts,expected_code,expected_budgets):
    budgets=[];statuses={}
    class Backend:
        def __init__(self,s): self.s=s
        def free_bytes(self,p): return 100*1024**3
        def preflight(self): return 'code-a'
        def launch_apps(self,c): budgets.append(c['episode_budget'])
        def wait_client(self,*a): pass
        def flow(self,*a): pass
        def check_platform(self,*a): pass
        def resolve_auto_python(self,*a): pass
        def bind(self,c):
            m,p=evidence(c['batch_id'],counts[len(budgets)-1])
            path=c['directory']/'manifest.json';receipt=c['directory']/'runtime-exit.json'
            m.update(manifest=str(path),runtime_exit_path=str(receipt))
            c['manifest']=m
            module.batches.write_json(path,m)
            module.batches.write_json(path.with_name('progress.json'),p)
            module.batches.write_json(receipt,dict(exit_code=1))
            statuses[str(path)]=p['status']
        def start_runtime(self,c): c['supervisor']=dict(process=SimpleNamespace(poll=lambda:1,returncode=1))
        def wait_ready(self,*a): pass
        def cleanup(self,*a): pass
        def commit_report(self,result): return 'report-commit'
    monkeypatch.setattr(module,'RealBackend',Backend)
    monkeypatch.setattr(module.batches,'status',lambda p:statuses[str(p)])
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    collector=module.CumulativeCollection(Settings('case',tmp_path/'recipe',repo=tmp_path))
    assert collector.run()==expected_code
    assert budgets==expected_budgets
    if expected_code==0:
        assert collector.state['completed_games']==10
        assert collector.state['state']=='completed'
        report=json.loads(Path(collector.state['successful_reports'][0]).read_text(encoding='utf-8'))
        assert all(a['runtime_receipt']['exit_code']==1 for a in report['attempts'])
        assert report['summary']['mean']==1700
    else:
        assert collector.state['completed_games']==0
        assert 'without any new' in collector.state['error']
