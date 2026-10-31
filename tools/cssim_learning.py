"""Local ten-game collection, offline IQL, desktop reports and reviewed iteration.

Runs independently of Codex through collection/training/reporting, then waits
at a durable review gate. Codex's scheduled chat reads this gate and publishes
the next version. Closing Codex never pretends an AI review has occurred.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import batch_training as batches
from autotrain_runtime import SessionLock
from cssim_campaign import Settings, RealBackend, WindowsProcesses, checked_git, same_process
from cssim_cumulative import CumulativeCollection
from cssim_annealing import metrics, assess, propose

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'logs/learning/state.json'
STOP = STATE.with_name('STOP')
PROFILE = ROOT/'PythonCode/TeamAlg/FsLutk7wD2RkNgbs/Algorithm/defense_profiles.json'
MODEL = ROOT/'PythonCode/TeamAlg/FsLutk7wD2RkNgbs/Models/target_iql_v1.json'
KINDS = ('unknown', 'hq_only', 'hq_a', 'hq_b', 'hq_a_b')
LABELS = dict(unknown='未知', hq_only='指挥部集中', hq_a='指挥部+A', hq_b='指挥部+B', hq_a_b='三点分守')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(state, **updates):
    state.update(updates, updated_at=batches.now())
    batches.write_json(STATE, state)
    if state.get('reports_dir'):
        folder = Path(state['reports_dir'])
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'训练状态.txt').write_text(
            f'状态：{state.get("phase")}\n轮次：{state.get("round_number",0)}\n'
            f'采集任务：{state.get("active_campaign", "尚未启动")}\n更新时间：{state["updated_at"]}\n'
            f'错误：{state.get("error", "无")}\n机器记录：{STATE}\n', encoding='utf-8-sig')


def settings(state, campaign):
    return Settings(campaign_id=campaign, repo=ROOT, recipe=ROOT/'tools/cssim_flow.json',
                    cssim_root=Path(state['cssim_root']), server_pid=state.get('server_pid'), max_batches=1)


def pythonw(state):
    return str(Path(state['cssim_root'])/'Client/Data/AlgData/PythonEnv/pythonw.exe')


def desktop_reports():
    # Resolve redirected/OneDrive desktops using the Windows known shell folder.
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
            r'Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders') as key:
        desktop = os.path.expandvars(winreg.QueryValueEx(key, 'Desktop')[0])
    return Path(desktop)/'迭代报告'


def live(identity):
    try:
        return bool(identity and same_process(identity, WindowsProcesses().identity(identity['pid'])))
    except (OSError, RuntimeError):
        return False


def child(state, role, arguments):
    directory = STATE.parent/f"round-{state['round_number']:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/f'{role}.log').open('ab') as output:
        process = subprocess.Popen([pythonw(state), *map(str, arguments)], cwd=ROOT,
            stdout=output, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    identity = WindowsProcesses().identity(process.pid)
    save(state, child=dict(role=role, identity=identity, arguments=list(map(str, arguments))))
    while process.poll() is None:
        if STOP.exists():
            WindowsProcesses().stop(identity)
            raise RuntimeError('STOP requested during '+role)
        time.sleep(5)
    save(state, child=None)
    if process.returncode:
        raise RuntimeError(f'{role} failed ({process.returncode}); see {directory/ (role+".log")}')


def checkpoint(state, number):
    folder = STATE.parent/f'round-{number:04d}'
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MODEL, folder/'model.json')
    shutil.copyfile(PROFILE, folder/'profiles.json')
    return dict(model=str(folder/'model.json'), profiles=str(folder/'profiles.json'),
                model_sha256=batches.sha(MODEL), profile_sha256=batches.sha(PROFILE),
                source_commit=checked_git(ROOT, 'rev-parse', 'HEAD'),
                experiment=read(state['next_review']) if state.get('next_review') else None)


def defence_samples(games, expected_profile=None):
    histograms = {}
    for protocol in dict.fromkeys(g['protocol'] for g in games):
        counters = defaultdict(Counter)
        simulation = Path(protocol).with_name('simulation.jsonl')
        if simulation.exists():
            with simulation.open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get('kind') == 'reset' and expected_profile:
                        policies = row.get('policies') or []
                        if not policies or policies[0].get('defense_profile_sha256') != expected_profile:
                            raise ValueError('Actual reset policy profile differs from round checkpoint')
                    entries = row.get('training_metrics') or []
                    code = entries[0].get('defense_kind_code') if entries else None
                    if type(code) is int and 0 <= code < len(KINDS):
                        counters[row['episode']][KINDS[code]] += 1
        for ep, counts in counters.items():
            histograms[(protocol, ep)] = counts
    result = []
    for game in games:
        counts = histograms.get((game['protocol'], game['episode']), Counter())
        known = {k: v for k, v in counts.items() if k != 'unknown'}
        kind = max(known, key=known.get) if known else 'unknown'
        result.append(dict(**game, observed_defense=kind, defense_frames=dict(counts)))
    return result


def summarize(state, report):
    games = defence_samples(report['games'], state['checkpoint']['profile_sha256'])
    grouped = defaultdict(list)
    for game in games:
        grouped[game['observed_defense']].append(game['red']['total'])
    scores = [g['red']['total'] for g in games]
    capture = Counter(g['red']['capture_score'] for g in games)
    return dict(round_number=state['round_number'], campaign_id=report['campaign_id'],
        source_commit=report['source_commit'], model_sha256=report['model_sha256'],
        profile_sha256=state['checkpoint']['profile_sha256'], games=games, metrics=metrics(scores),
        defense_groups={kind: metrics(values) for kind, values in grouped.items()},
        capture_scores=dict(capture), actual_exit_codes=[a.get('runtime_receipt', {}).get('exit_code')
            for a in report['attempts'] if a.get('runtime_receipt')],
        problems=[f'{sum(s<5000 for s in scores)}/10局低于5000；不能用平均分代表稳定性。',
            f'{sum(g["red"]["capture_score"]<3200 for g in games)}/10局终局占领分不足3200；需核查未占领和失守的时间线。',
            '防守分型仅来自红方公开观察，不代表敌方隐藏初始部署；分组少样本不能证明措施因果。'],
        plan=['保留最佳版本；以实际十局均值、低分比例、尾部均值与方差评估退火候选。',
              '每次只试验一类战术参数或一份IQL权重，冻结到累计十局；新模型先做接入回放。',
              'IQL只学习地面目标选择；路线和队形仍由有界策略控制，不宣称端到端自训练。'])


def write_reports(state, summary):
    folder = Path(state['reports_dir'])
    folder.mkdir(parents=True, exist_ok=True)
    number = summary['round_number']
    stem = f'第{number:04d}轮'
    m = summary['metrics']
    lines = [f'CSSIM 第{number}轮：累计十局', '',
        f'源码：{summary["source_commit"]}', f'模型 SHA：{summary["model_sha256"]}',
        f'战术 SHA：{summary["profile_sha256"]}', '',
        f'均分 {m["mean"]:.1f}；最低 {m["minimum"]}；最高 {m["maximum"]}；低于5000：{m["under_5000"]}/10。',
        f'最低两局均分 {m["bottom_20_mean"]:.1f}；标准差 {m["stddev"]:.1f}。',
        '逐局成绩（终局摧毁分+Reward=总分）：',
        *[f'{index}：{g["red"].get("kill_score", "未知")} + {g["red"].get("capture_score", "未知")} = {g["red"]["total"]}'
          for index, g in enumerate(summary['games'], 1)],
        '实际进程退出码：'+str(summary['actual_exit_codes']), '', '主要问题：', *summary['problems'], '',
        '下一版修改方案（发布前审查）：', *summary['plan'], '', '公开敌情分组：']
    for kind, values in summary['defense_groups'].items():
        lines.append(f'{LABELS[kind]}：{values["count"]}局，均分{values["mean"]:.1f}，低于5000共{values["under_5000"]}局。')
    if summary.get('proposed_change'):
        change = summary['proposed_change']
        lines += ['', f'待审战术候选：{LABELS[change["kind"]]}，{change["field"]}由{change["before"]}改为{change["after"]}。',
                  f'自然日窗口：{change["starts_on"]}至{change["ends_on"]}；温度：{change["temperature"]}。',
                  f'本轮退火接受结果：{summary["annealing"]["accepted"]}；最佳版本仍保留。']
    if summary.get('candidate_review'):
        candidate = read(summary['candidate_review'])
        lines += ['', '离线IQL候选已生成，尚无该候选的新实战成绩。',
                  f'候选SHA：{candidate["candidate_model_sha256"]}',
                  '训练/验证/测试源角色：'+json.dumps(candidate['split_sessions'],ensure_ascii=False),
                  '最后两源分别留给验证和测试，不参加本次梯度训练。']
    elif summary.get('candidate_error'):
        lines += ['', '本轮候选训练异常：'+summary['candidate_error']]
    lines += ['', '分数=终局摧毁分+Reward。异常退出不抹去完整终局；离线回放/一致率不是新对局分数。',
              '本文件可手动上传；Codex巡检通过本地待审队列读取，不伪称自动上传附件。']
    (folder/(stem+'.txt')).write_text('\n'.join(lines)+'\n', encoding='utf-8-sig')
    if number % 4 == 0:
        block = [*state['rounds'][-3:], summary]
        aggregate = dict(rounds=[r['round_number'] for r in block],
            overall=metrics([g['red']['total'] for r in block for g in r['games']]),
            versions=[dict(round_number=r['round_number'], source_commit=r['source_commit'],
                           metrics=r['metrics']) for r in block],
            limitation='不同版本、非配对防守；总体均分不是单一策略的成绩，也不是因果检验。')
        (folder/f'四轮大报-{number:04d}.txt').write_text('\n'.join([
            f'CSSIM 第{number-3}–{number}轮大报', '',
            f'四轮共{aggregate["overall"]["count"]}局；总体均分{aggregate["overall"]["mean"]:.1f}；低于5000共{aggregate["overall"]["under_5000"]}局。',
            *[f'第{r["round_number"]}轮：均分{r["metrics"]["mean"]:.1f}，低于5000共{r["metrics"]["under_5000"]}/10，版本{r["source_commit"]}' for r in block],
            '', aggregate['limitation']])+'\n', encoding='utf-8-sig')
    (folder/'使用说明.txt').write_text(
        '双击桌面“CSSIM训练”启动。勿重复手动操作对局界面。电脑须保持开机、登录且桌面可交互。\n'
        '双击“停止CSSIM训练”发出STOP，保留完整成绩。不会关闭借用的服务器管理器。\n'
        '每个版本累计十局后写简报、训练离线IQL候选，等待审查。每四轮有大报。\n'
        'Codex开启时由巡检接续审查/发布/下一轮；关闭时本地仍完成当前轮及候选并写报告，随后等待。\n'
        '重开Codex后巡检读取本地待审文件；不是文件自动上传API，也不保证启动瞬间触发。\n'
        '原始大日志在wargaming/logs，精简成绩和发布审查推送GitHub。\n', encoding='utf-8-sig')


def collect_and_train(state):
    number = len(state['rounds'])+1
    campaign = state['session_id']+f'-r{number:04d}'
    save(state, phase='collecting', round_number=number, active_campaign=campaign,
         checkpoint=checkpoint(state, number), pending=None)
    collector = CumulativeCollection(settings(state, campaign))
    if collector.run():
        save(state, phase='stopped' if STOP.exists() else 'needs_attention', error=collector.state.get('error'))
        return False
    report_path = collector.state['successful_reports'][0]
    report = read(report_path)
    if report['model_sha256'] != state['checkpoint']['model_sha256']:
        raise ValueError('Collected model differs from pre-round checkpoint')
    summary = summarize(state, report)
    summary['collection_report'] = report_path
    write_reports(state, summary)  # Feedback exists before any candidate/model change.
    save(state, phase='training', pending=summary)
    candidate_id = campaign+'-iql'
    args = [ROOT/'tools/train_batch_candidate.py', '--candidate-id', candidate_id,
        '--batches', *state['seed_reports'], *[r['collection_report'] for r in state['rounds']], report_path,
        '--sources', *state['historical_sources'], '--steps', str(state['train_steps'])]
    try:
        child(state, 'offline-training', args)
        summary['candidate_review'] = str(ROOT/'logs/autotrain-candidates'/candidate_id/'review.json')
    except RuntimeError as exc:
        summary['candidate_error'] = str(exc)
        if STOP.exists():
            raise
    observation = dict(metrics=summary['metrics'], checkpoint=state['checkpoint'],
                       round_number=number, source_commit=summary['source_commit'],
                       defense_groups=summary['defense_groups'])
    experiment = state['checkpoint'].get('experiment') or {}
    if experiment.get('chosen') == 'bounded_tactical_parameter':
        observation['tested_kind'] = experiment['proposal']['kind']
    search = assess(state.get('search', {}), observation, round_number=number, anchor=state['anchor_date'])
    base = read(search['current']['checkpoint']['profiles'])
    eligible = [k for k, group in summary['defense_groups'].items() if k != 'unknown' and group['count'] >= 3]
    proposal, change = propose(base, round_number=number, anchor=state['anchor_date'], eligible=eligible)
    summary.update(annealing=search['last_acceptance'], proposed_change=change)
    batches.write_json(STATE.parent/f'round-{number:04d}'/'proposed-profiles.json', proposal)
    # Do not append before writing the four-round report; it needs previous three.
    write_reports(state, summary)
    summary_path = ROOT/'results/learning'/f'{campaign}-review.json'
    batches.write_json(summary_path, summary)
    # Reports only; no source changes occur until the external reviewed publish step.
    RealBackend(settings(state, campaign)).commit_report(dict(result_path=str(summary_path), batch_id=campaign+'-review'))
    save(state, phase='awaiting_review', rounds=[*state['rounds'], summary], pending=summary,
         search=search, pending_summary=str(summary_path))
    return True


def prepare_next(state, *, use_candidate=False):
    if state['phase'] != 'awaiting_review' or STOP.exists():
        raise ValueError('Next version may only be staged at a completed ten-game review gate')
    if checked_git(ROOT, 'status', '--porcelain'):
        raise ValueError('Preserve and review existing uncommitted work before staging a candidate')
    number = state['round_number']
    current = state['search']['current']['checkpoint']
    chosen_model = Path(current['model'])
    proposal = STATE.parent/f'round-{number:04d}'/'proposed-profiles.json'
    review = dict(round_number=number, created_at=batches.now(), live_score_improvement_unproven=True,
                  rollback_checkpoint=current, chosen='bounded_tactical_parameter',
                  proposal=state['pending']['proposed_change'])
    if use_candidate:
        candidate = read(state['pending']['candidate_review'])
        chosen_model = Path(candidate['candidate_model'])
        if batches.sha(chosen_model) != candidate['candidate_model_sha256']:
            raise ValueError('Candidate model checksum changed')
        # One low and one high terminal game; replay verifies interface, not returns.
        ordered = sorted(state['pending']['games'], key=lambda g: g['red']['total'])
        checks = []
        for label, game in (('low', ordered[0]), ('high', ordered[-1])):
            out = STATE.parent/f'round-{number:04d}'/f'integration-{label}.json'
            child(state, 'integration-'+label, [ROOT/'tools/evaluate_target_iql.py', game['protocol'],
                '--model', chosen_model, '--episodes', str(game['episode']), '--out', out])
            check = read(out)
            if check['model_sha256'] != candidate['candidate_model_sha256']:
                raise ValueError('Integration replay checked a different model')
            checks.append(str(out))
        review.update(chosen='offline_iql_trial', integration_checks=checks,
                      candidate_review=state['pending']['candidate_review'],
                      split_sessions=candidate['split_sessions'])
        # Isolate the weight trial from tactical mutations.
        proposal = Path(current['profiles'])
    shutil.copyfile(chosen_model, MODEL)
    shutil.copyfile(proposal, PROFILE)
    expected = dict(model_sha256=batches.sha(MODEL), profile_sha256=batches.sha(PROFILE))
    review.update(expected)
    output = ROOT/'results/learning'/f'{state["active_campaign"]}-next-version.json'
    batches.write_json(output, review)
    save(state, next_expected=expected, next_review=str(output))
    return review


def release(state, commit):
    if state['phase'] != 'awaiting_review' or STOP.exists() or not state.get('next_expected'):
        raise ValueError('Release requires a staged, reviewed next version and no STOP')
    actual = RealBackend(settings(state, state['active_campaign'])).preflight()
    if actual != commit or actual == state['pending']['source_commit']:
        raise ValueError('Release commit must identify the newly published and deployed version')
    expected = dict(model_sha256=batches.sha(MODEL), profile_sha256=batches.sha(PROFILE))
    if expected != state['next_expected']:
        raise ValueError('Published model/tactics differ from reviewed artifacts')
    batches.write_json(STATE.with_name('release.json'), dict(round_number=state['round_number'],
        commit=commit, **expected, released_at=batches.now()))


def run(state):
    with SessionLock(STATE.with_name('learning.lock')):
        save(state, worker=WindowsProcesses().identity(os.getpid()))
        try:
            if state['phase'] not in ('created', 'awaiting_review', 'released'):
                raise ValueError('Interrupted coordinator requires recovery; do not discard cumulative games')
            while not STOP.exists():
                if state['phase'] in ('created', 'released'):
                    if not collect_and_train(state):
                        return 2
                # The external reviewer writes next_expected; do not use a stale
                # in-memory gate snapshot or overwrite the reviewer's updates.
                state = read(STATE)
                release_path = STATE.with_name('release.json')
                if release_path.exists():
                    token = read(release_path)
                    if token['round_number'] == state['round_number']:
                        release(state, token['commit'])  # Revalidate disk/GitHub before starting.
                        release_path.unlink()
                        save(state, phase='released', next_expected=None)
                        continue
                time.sleep(10)
            save(state, phase='stopped')
            return 0
        except Exception as exc:
            save(state, phase='stopped' if STOP.exists() else 'needs_attention', error=f'{type(exc).__name__}: {exc}')
            return 2


def start(args):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    if STATE.exists():
        state = read(STATE)
        if live(state.get('worker')):
            return dict(already_running=True, phase=state['phase'], pid=state['worker']['pid'])
        if STOP.exists():
            raise ValueError('STOP is present. Review recovery before explicitly removing it.')
    else:
        config = read(ROOT/'tools/cssim_learning.json')
        state = dict(phase='created', rounds=[], round_number=0,
            session_id=datetime.now().strftime('learning-%Y%m%d-%H%M%S'),
            anchor_date=datetime.now(timezone(timedelta(hours=8))).date().isoformat(),
            reports_dir=str(desktop_reports()), cssim_root=str(args.cssim_root), server_pid=args.server_pid,
            **config)
        save(state)
    if not args.server_pid and not state.get('server_pid'):
        manager = str(Path(state['cssim_root'])/'Server/Server/Server.exe')
        matches = [p for p in WindowsProcesses().snapshot()
                   if p['name'].casefold() == 'server.exe'
                   and os.path.normcase(p.get('exe_path') or '') == os.path.normcase(manager)]
        if len(matches) == 1:
            save(state, server_pid=matches[0]['pid'])
    with STATE.with_name('worker.log').open('ab') as output:
        process = subprocess.Popen([pythonw(state), str(Path(__file__).resolve()), 'run'], cwd=ROOT,
            stdout=output, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    batches.write_json(STATE.with_name('launch.json'), dict(identity=WindowsProcesses().identity(process.pid),
        command=[pythonw(state), str(Path(__file__).resolve()), 'run'], started_at=batches.now()))
    return dict(started_pid=process.pid, reports_dir=state['reports_dir'])


def request_stop(state):
    STOP.parent.mkdir(parents=True, exist_ok=True)
    STOP.touch()
    campaign = state.get('active_campaign')
    if campaign:
        directory = ROOT/'logs/campaigns'/campaign
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'STOP').touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'run', 'status', 'stop', 'prepare-next', 'release'))
    parser.add_argument('--cssim-root', type=Path, default=Path('C:/CSSIM'))
    parser.add_argument('--server-pid', type=int)
    parser.add_argument('--commit')
    parser.add_argument('--use-candidate', action='store_true')
    args = parser.parse_args()
    if args.action == 'start':
        print(json.dumps(start(args), ensure_ascii=False)); return 0
    state = read(STATE)
    if args.action == 'run':
        return run(state)
    if args.action == 'status':
        print(json.dumps({k:state.get(k) for k in ('phase','round_number','active_campaign','updated_at','error')}, ensure_ascii=False))
    elif args.action == 'stop':
        request_stop(state)
    elif args.action == 'prepare-next':
        print(json.dumps(prepare_next(state, use_candidate=args.use_candidate), ensure_ascii=False))
    elif args.action == 'release':
        release(state, args.commit)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
