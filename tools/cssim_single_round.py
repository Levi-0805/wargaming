"""Collect one fresh ten-game cohort, then exit without training or release."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import batch_training as batches
from autotrain_runtime import SessionLock
from cssim_campaign import CampaignBlocked, CampaignStop, Settings, WindowsProcesses, normalized, same_process
from cssim_cumulative import CumulativeCollection

ROOT = Path(__file__).resolve().parents[1]


def desktop_reports():
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                       r'Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders') as key:
        return Path(os.path.expandvars(winreg.QueryValueEx(key, 'Desktop')[0])) / '迭代报告'


def bind_identity(processes, pid, executable):
    identity = processes.identity(pid)
    if (identity.get('created_ticks') is None or not identity.get('exe_path')
            or normalized(identity['exe_path']) != normalized(executable)):
        raise CampaignBlocked(f'PID {pid} lacks the expected full process identity: {executable}')
    return identity


class BoundBackend:
    """Recheck borrowed identities before the existing backend claims anything."""
    def __init__(self, backend, servers):
        self.backend, self.servers = backend, servers

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def launch_apps(self, context):
        for expected in self.servers:
            try:
                current = self.processes.identity(expected['pid'])
            except OSError as exc:
                raise CampaignBlocked('Cannot recheck explicit borrowed server identity') from exc
            if not same_process(expected, current):
                raise CampaignBlocked('Explicit borrowed server identity changed before launch')
        return self.backend.launch_apps(context)


class SingleRoundCollection(CumulativeCollection):
    def __init__(self, settings, servers, launch):
        self.control = settings.repo / 'logs/single-round' / settings.campaign_id
        self.launch = launch
        super().__init__(settings)
        self.backend = BoundBackend(self.backend, servers)

    def guard(self):
        super().guard()
        if (self.s.repo / 'logs/learning/STOP').exists():
            raise CampaignStop('Global learning STOP requested')

    def save(self, **values):
        super().save(**values)
        batches.write_json(self.control / 'status.json', dict(
            campaign_id=self.s.campaign_id, worker=self.launch['worker'],
            report_round=self.launch['report_round'], updated_at=batches.now(),
            phase=self.state['state'], completed_games=self.state['completed_games'],
            target_games=10, source_commit=self.state.get('source_commit'),
            error=self.state.get('error'), campaign_state=str(self.path)))


def validated_games(state):
    games = state.get('games', [])
    if state.get('completed_games') != len(games) or len(games) > 10:
        raise ValueError('Terminal-game ledger count differs from the ten-game budget')
    seen, models = set(), set()
    for game in games:
        key = (game['protocol'], game['episode'])
        if key in seen or not game.get('protocol_sha256'):
            raise ValueError('Duplicate or unbound terminal game')
        seen.add(key)
        if not state.get('source_commit') or game['source_commit'] != state['source_commit']:
            raise ValueError('Terminal games mix source versions')
        models.add(game['model_sha256'])
        for side in ('red', 'blue'):
            scores = game[side]
            values = [scores[k] for k in ('kill_score', 'capture_score', 'total')]
            if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                    or values[0] + values[1] != values[2]):
                raise ValueError('Terminal total must equal destruction score plus Reward')
    if len(models) > 1 or (games and not next(iter(models))):
        raise ValueError('Terminal games mix or omit model versions')
    return games


def write_exit_report(state, launch, control, reports, exit_code):
    """Report verified terminal games even if the collector's report push failed."""
    error = None
    try:
        games = validated_games(state)
    except (KeyError, TypeError, ValueError) as exc:
        games, error = [], f'{type(exc).__name__}: {exc}'
    complete = len(games) == 10 and error is None
    published = complete and state.get('state') == 'completed' and bool(state.get('report_commit'))
    if error or (exit_code == 0 and not complete):
        exit_code = 2
    folder = reports / f'第{launch["report_round"]:04d}轮'
    folder.mkdir(parents=True, exist_ok=True)
    report = folder / '复测状态.txt'
    lines = [f'独立复测：{launch["campaign_id"]}', f'更新时间：{batches.now()}',
             f'有效终局：{len(games)}/10；' + ('十局已完成。' if complete else '尚未完成十局。'),
             f'采集器状态：{state.get("state")}；退出码：{exit_code}',
             f'源码：{state.get("source_commit", "未绑定")}',
             f'模型SHA：{games[0]["model_sha256"] if games else "未核验"}',
             f'成绩GitHub备份：{"已核实" if published else "未核实"}；提交：{state.get("report_commit")}',
             '逐局终局摧毁分 + Reward = 总分：']
    lines += [f'{i}：{g["red"]["kill_score"]} + {g["red"]["capture_score"]} = {g["red"]["total"]}'
              for i, g in enumerate(games, 1)]
    if games:
        scores = [g['red']['total'] for g in games]
        lines += [f'均分：{sum(scores)/len(scores):.1f}；最低：{min(scores)}；最高：{max(scores)}；低于5000：{sum(s < 5000 for s in scores)}/{len(scores)}']
    lines += ['实际尝试退出记录：']
    lines += [f'{a["batch_id"]}：supervisor={a.get("supervisor_exit_code")}；runtime={(a.get("runtime_receipt") or {}).get("exit_code")}；错误={a.get("error")}'
              for a in state.get('attempts', [])]
    lines += [f'错误：{error or state.get("error") or "无"}',
              f'机器记录：{control / "status.json"}',
              '旧轮次成绩独立保留；本入口不会训练候选、发布下一版或启动下一轮。']
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')
    status = dict(campaign_id=launch['campaign_id'], worker=launch['worker'],
                  report_round=launch['report_round'], phase=state.get('state'),
                  updated_at=batches.now(), completed_games=len(games), target_games=10,
                  scores_complete=complete, report_published=published, exit_code=exit_code,
                  source_commit=state.get('source_commit'), error=error or state.get('error'),
                  report_path=str(report), campaign_state=launch['campaign_state'])
    batches.write_json(control / 'status.json', status)
    if complete:
        batches.write_json(control / 'report-ready.json', dict(
            id=launch['campaign_id'], ready_at=batches.now(), completed_games=10,
            source_commit=state['source_commit'], model_sha256=games[0]['model_sha256'],
            report_path=str(report), campaign_state=launch['campaign_state'],
            collection_report=launch['collection_report'], report_published=published,
            delivered_to_chat=False, worker=launch['worker']))
    return complete


def run_once(args, *, repo=ROOT):
    repo = Path(repo).resolve()
    if args.report_round <= 0:
        raise ValueError('Report round must be positive')
    settings = Settings(campaign_id=args.campaign_id, repo=repo, recipe=repo / 'tools/cssim_flow.json',
                        cssim_root=args.cssim_root, server_pid=args.server_pid,
                        additional_server_pids=tuple(args.additional_server_pid), max_batches=1)
    campaign = repo / 'logs/campaigns' / args.campaign_id
    control = repo / 'logs/single-round' / args.campaign_id
    if campaign.exists() or control.exists():
        raise ValueError('Use a fresh campaign ID; existing records must be preserved')
    if (repo / 'logs/learning/STOP').exists():
        raise CampaignStop('Global learning STOP is present')
    processes = WindowsProcesses()
    learner_path = repo / 'logs/learning/state.json'
    learner = batches.read_json(learner_path) if learner_path.exists() else {}
    worker = learner.get('worker')
    if worker:
        for row in processes.snapshot():
            if row['pid'] == worker['pid'] and (row.get('created_ticks') is None
                    or row['created_ticks'] == worker['created_ticks']):
                raise CampaignBlocked('Learning worker is still live or its identity cannot be excluded')
    servers = [bind_identity(processes, pid, settings.cssim_root / 'Server/Server/Server.exe')
               for pid in (settings.server_pid, *settings.additional_server_pids) if pid is not None]
    if args.adopt_client_pid is not None:
        settings.adopt_client = bind_identity(processes, args.adopt_client_pid,
            settings.cssim_root / 'Client/CSSIM/Binaries/Win64/CSSIMClient-Win64-Shipping.exe')
    launch = dict(campaign_id=args.campaign_id, report_round=args.report_round, started_at=batches.now(),
                  worker=processes.identity(os.getpid()), borrowed_servers=servers,
                  adopted_client=settings.adopt_client, target_games=10, max_rounds=1,
                  campaign_state=str(campaign / 'state.json'),
                  collection_report=str(repo / 'results/batches' / f'{args.campaign_id}-cumulative.json'),
                  stop_files=[str(repo / 'logs/learning/STOP'), str(campaign / 'STOP')])
    reports = desktop_reports()
    if (reports / f'第{args.report_round:04d}轮' / '复测状态.txt').exists():
        raise ValueError('An independent report already uses this report round')
    # Prevent a learner from acquiring its loop while this independent cohort runs.
    with SessionLock(repo / 'logs/learning/learning.lock'):
        batches.write_json(control / 'launch.json', launch)
        collector = SingleRoundCollection(settings, servers, launch)
        code = 2
        try:
            code = collector.run()  # Exactly one cohort; never call learner release or training.
        except Exception as exc:
            collector.save(state='needs_attention', error=f'{type(exc).__name__}: {exc}')
        finally:
            complete = write_exit_report(collector.state, launch, control, reports, code)
            if code == 0 and not complete:
                code = 2
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-id', required=True)
    parser.add_argument('--report-round', required=True, type=int)
    parser.add_argument('--server-pid', type=int)
    parser.add_argument('--additional-server-pid', type=int, action='append', default=[])
    parser.add_argument('--adopt-client-pid', type=int)
    parser.add_argument('--cssim-root', type=Path, default=Path('C:/CSSIM'))
    args = parser.parse_args()
    try:
        return run_once(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps(dict(error=f'{type(exc).__name__}: {exc}'), ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
