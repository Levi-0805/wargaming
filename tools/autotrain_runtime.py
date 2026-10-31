"""Own one CSSIM Python process and persist its real exit for batch auditing.

The desktop agent handles the observed CSSIM UI. This supervisor never clicks
blindly, changes models mid-match, or kills unrelated Python/app processes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from batch_training import read_json, sha, status, write_json


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def run_metadata(log_root, batch_id, pid):
    matches = []
    for path in Path(log_root).glob('*/run.json'):
        try:
            metadata = read_json(path)
        except (OSError, ValueError):
            continue
        if metadata.get('batch_id') == batch_id and metadata.get('process_id') == pid:
            matches.append(path.parent)
    if len(matches) > 1:
        raise ValueError('Multiple runtime directories for one owned process')
    return matches[0] if matches else None


def stop_reason(*, stop_requested, free_bytes, minimum_free_bytes, elapsed,
                total_timeout, idle, idle_timeout):
    if stop_requested:
        return 'requested_stop'
    if free_bytes < minimum_free_bytes:
        return 'low_disk_space'
    if elapsed >= total_timeout:
        return 'batch_timeout'
    if idle >= idle_timeout:
        return 'progress_timeout'
    return None


class SessionLock:
    """An OS lock automatically released even if the supervisor crashes."""
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+b')
        self.stream.seek(0)
        if not self.stream.read(1):
            self.stream.write(b'0'); self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError('Another managed CSSIM runtime is already running')
        return self

    def __exit__(self, *_):
        self.stream.close()


def run(manifest_path, *, timeout_minutes=360, idle_minutes=10, minimum_free_gb=10):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    batch = manifest_path.parent
    receipt_path = Path(manifest['runtime_exit_path'])
    if receipt_path.exists() or (batch / 'runtime-process.json').exists():
        raise ValueError('This attempt was already launched; preserve it and prepare a new batch ID')
    if min(timeout_minutes, idle_minutes, minimum_free_gb) <= 0:
        raise ValueError('Timeout and disk limits must be positive')
    minimum_free = int(minimum_free_gb * 1024**3)
    if shutil.disk_usage(batch).free < minimum_free:
        raise RuntimeError('Insufficient free space for detailed training logs')
    status(manifest_path)  # Fail before launch on source/config/deployment drift.
    code = Path(manifest['cssim_root']) / 'Client/Data/AlgData/PythonCode'
    python = code.parent / 'PythonEnv/python.exe'
    command = [str(python), str(code / 'main.py'), '-c', manifest['config_path'],
               '--log-root', manifest['log_root'], '--log-level', 'DEBUG',
               '--exit-after-run',
               '--episode-limit', str(manifest['expected_episodes']),
               '--batch-id', manifest['batch_id'], '--source-commit', manifest['source_commit']]
    child = None
    directory = None
    reason, error = None, None
    lock = Path(manifest['repo']) / 'logs/autotrain/runtime.lock'
    with SessionLock(lock), (batch / 'runtime-console.log').open('wb') as console:
        started = utcnow()
        beginning = last_progress = time.monotonic()
        previous_size = 0
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        child = subprocess.Popen(command, cwd=code, stdout=console, stderr=subprocess.STDOUT,
                                 env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            write_json(batch / 'runtime-process.json', dict(batch_id=manifest['batch_id'], pid=child.pid,
                       started_at=started, command=command, supervisor_pid=os.getpid()))
            print(json.dumps({'state': 'started', 'pid': child.pid, 'batch_id': manifest['batch_id']}), flush=True)
            while child.poll() is None:
                directory = run_metadata(manifest['log_root'], manifest['batch_id'], child.pid)
                simulation = directory / 'simulation.jsonl' if directory else None
                size = simulation.stat().st_size if simulation and simulation.exists() else 0
                if size != previous_size:
                    last_progress, previous_size = time.monotonic(), size
                moment = time.monotonic()
                reason = stop_reason(stop_requested=(batch / 'STOP').exists(),
                                     free_bytes=shutil.disk_usage(batch).free, minimum_free_bytes=minimum_free,
                                     elapsed=moment-beginning, total_timeout=timeout_minutes*60,
                                     idle=moment-last_progress, idle_timeout=idle_minutes*60)
                if sha(manifest['config_path']) != manifest['config_sha256']:
                    reason = 'runtime_config_changed'
                progress = status(manifest_path)
                write_json(batch / 'supervisor-status.json', dict(updated_at=utcnow(), pid=child.pid,
                           elapsed_seconds=round(moment-beginning), stop_reason=reason, batch_status=progress))
                if reason:
                    break
                time.sleep(5)
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            reason = 'supervisor_error'
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=15)
            try:
                directory = run_metadata(manifest['log_root'], manifest['batch_id'], child.pid)
            except (OSError, ValueError) as exc:
                directory = None
                reason = reason or 'metadata_error'
                error = error or f'{type(exc).__name__}: {exc}'
            receipt = dict(batch_id=manifest['batch_id'], source_commit=manifest['source_commit'],
                           config_sha256=manifest['config_sha256'], pid=child.pid,
                           started_at=started, ended_at=utcnow(), exit_code=child.returncode,
                           log_directory=str(directory) if directory else None, stop_reason=reason, error=error)
            write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0 if child.returncode == 0 and reason is None else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--timeout-minutes', type=float, default=360)
    parser.add_argument('--idle-minutes', type=float, default=10)
    parser.add_argument('--minimum-free-gb', type=float, default=10)
    args = parser.parse_args()
    return run(args.manifest, timeout_minutes=args.timeout_minutes,
               idle_minutes=args.idle_minutes, minimum_free_gb=args.minimum_free_gb)


if __name__ == '__main__':
    raise SystemExit(main())
