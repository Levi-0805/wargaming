"""Collect ten completed games for one published version across client restarts.

Each retry receives only the remaining episode budget. Failed/incomplete games
are never counted; failed process exits are retained, not rewritten as success.
After the version reaches ten games this controller exits for review/iteration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import batch_training as batches
from autotrain_runtime import SessionLock
from cssim_campaign import CampaignBlocked, CampaignStop, RealBackend, Settings


def append_completed(games, manifest, progress, *, target=10):
    """Use terminal score pairs AND done records, deduplicated by source/round."""
    parsed = progress['score_summary']
    if parsed['malformed_json_lines']:
        raise ValueError('Malformed protocol cannot contribute cumulative games')
    done = set(progress['status']['done_episodes'])
    known = {(g['protocol'], g['episode']) for g in games}
    additions = []
    for row in parsed['rounds']:
        key = (progress['status']['protocol'], row['episode'])
        if not row['complete'] or row['episode'] not in done or key in known:
            continue
        additions.append(dict(ordinal=len(games)+len(additions)+1, episode=row['episode'],
            batch_id=manifest['batch_id'], protocol=key[0], protocol_sha256=parsed['source_sha256'],
            source_commit=manifest['source_commit'], model_sha256=manifest['model_sha256'],
            red={k: row['scores']['red'][k] for k in ('kill_score', 'capture_score', 'total')},
            blue={k: row['scores']['blue'][k] for k in ('kill_score', 'capture_score', 'total')}))
    if len(games)+len(additions) > target:
        raise ValueError('Runtime exceeded its remaining episode budget')
    if games and any(g['source_commit'] != games[0]['source_commit'] or
                     g['model_sha256'] != games[0]['model_sha256'] for g in additions):
        raise ValueError('Cannot combine different code/model versions')
    games.extend(additions)
    return len(additions)


class CumulativeCollection:
    def __init__(self, settings):
        self.s, self.backend = settings, RealBackend(settings)
        self.directory = settings.repo/'logs/campaigns'/settings.campaign_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stopfile = self.directory/'STOP'
        self.path = self.directory/'state.json'
        if self.path.exists():
            raise ValueError('Use a new campaign ID; never overwrite previous records')
        self.state = dict(campaign_id=settings.campaign_id, state='created', target_games=10,
                          completed_games=0, games=[], attempts=[], no_progress_failures=0,
                          mode='cumulative_one_code_version', successful_reports=[])
        self.save()

    def save(self, **values):
        self.state.update(values, updated_at=batches.now())
        batches.write_json(self.path, self.state)

    def guard(self):
        if self.stopfile.exists():
            raise CampaignStop('STOP requested')
        if self.backend.free_bytes(self.directory) < self.s.minimum_free_gb*1024**3:
            raise CampaignStop('Low disk space')

    def attempt(self, number):
        context = dict(batch_id=f'{self.s.campaign_id}-{number:04d}',
            directory=self.directory/f'attempt-{number:04d}', owned=[], baseline_pids=set(),
            stopfile=self.stopfile, episode_budget=10-len(self.state['games']))
        context['directory'].mkdir()
        record = dict(batch_id=context['batch_id'], started_at=batches.now(),
                      episode_budget=context['episode_budget'])
        error = None
        try:
            self.guard()
            commit = self.backend.preflight()
            context['source_commit'] = commit
            if self.state.get('source_commit', commit) != commit:
                raise CampaignBlocked('Published code changed before ten cumulative games')
            self.save(source_commit=commit, state='configuring_ui', current_attempt=context['batch_id'])
            self.backend.launch_apps(context)
            self.backend.wait_client(context, self.guard)
            self.backend.flow(context, 'pre_python', self.guard)
            self.backend.check_platform(context, self.guard)
            self.backend.resolve_auto_python(context)
            self.backend.bind(context)
            if context['manifest']['source_commit'] != commit:
                raise CampaignBlocked('Source changed during setup')
            self.backend.start_runtime(context)
            self.backend.wait_ready(context, self.guard)
            self.backend.flow(context, 'post_python', self.guard)
            self.save(state='collecting')
            process = context['supervisor']['process']
            while process.poll() is None:
                self.guard()
                self.backend.refresh(context)
                status_path = Path(context['manifest']['manifest']).with_name('supervisor-status.json')
                if status_path.exists():
                    self.save(runtime=batches.read_json(status_path))
                time.sleep(5)
            record['supervisor_exit_code'] = process.returncode
        except BaseException as exc:
            error = exc
            record['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            try:
                self.backend.cleanup(context)
            except BaseException as exc:
                error = CampaignBlocked(str(exc))
                record['cleanup_error'] = str(exc)
        added = 0
        if context.get('manifest'):
            manifest = context['manifest']
            current = batches.status(manifest['manifest'])
            progress_path = Path(manifest['manifest']).with_name('progress.json')
            if progress_path.exists():
                progress = batches.read_json(progress_path)
                if progress.get('score_summary'):
                    added = append_completed(self.state['games'], manifest, progress)
            receipt_path = Path(manifest['runtime_exit_path'])
            record.update(manifest_path=manifest['manifest'], status=current,
                          runtime_receipt=batches.read_json(receipt_path) if receipt_path.exists() else None)
        record.update(completed_games_added=added, ended_at=batches.now())
        self.state['attempts'].append(record)
        self.state['no_progress_failures'] = 0 if added else self.state['no_progress_failures']+1
        batches.write_json(context['directory']/'attempt.json', record)
        self.save(completed_games=len(self.state['games']))
        if isinstance(error, (CampaignStop, CampaignBlocked, KeyboardInterrupt, SystemExit)):
            raise error

    def finish(self):
        games = self.state['games']
        scores = [g['red']['total'] for g in games]
        report = dict(classification='ten_cumulative_games_complete',
            campaign_id=self.s.campaign_id, source_commit=self.state['source_commit'],
            model_sha256=games[0]['model_sha256'], completed_games=len(games),
            games=games, attempts=self.state['attempts'], finished_at=batches.now(),
            summary=dict(mean=statistics.mean(scores), minimum=min(scores), maximum=max(scores),
                         under_4000=sum(s<4000 for s in scores)),
            limitation='Completed games may span failed processes; actual receipts remain unchanged.',
            next_step='Summarize and review modifications before any next code version.')
        output = self.s.repo/'results/batches'/f'{self.s.campaign_id}-cumulative.json'
        batches.write_json(output, report)
        result = dict(result_path=str(output), batch_id=self.s.campaign_id)
        commit = self.backend.commit_report(result)
        self.save(state='completed', successful_reports=[str(output)], report_commit=commit,
                  summary=report['summary'])

    def run(self):
        try:
            with SessionLock(self.s.repo/'logs/autotrain/campaign.lock'):
                number = 0
                while len(self.state['games']) < 10:
                    self.guard()
                    number += 1
                    self.attempt(number)
                    if len(self.state['games']) >= 10:
                        break
                    if self.state['no_progress_failures'] >= 3:
                        raise CampaignBlocked('Three attempts without any new completed game')
                    self.save(state='retry_wait')
                    for _ in range(15):
                        self.guard(); time.sleep(1)
                self.finish()
            return 0
        except (CampaignStop, KeyboardInterrupt) as exc:
            self.save(state='stopped', error=str(exc)); return 2
        except Exception as exc:
            self.save(state='needs_attention', error=f'{type(exc).__name__}: {exc}'); return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-id', required=True)
    parser.add_argument('--server-pid', type=int)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--cssim-root', type=Path, default=Path('C:/CSSIM'))
    parser.add_argument('--recipe', type=Path, default=Path('tools/cssim_flow.json'))
    args = parser.parse_args()
    return CumulativeCollection(Settings(**vars(args), max_batches=1)).run()


if __name__ == '__main__':
    raise SystemExit(main())
