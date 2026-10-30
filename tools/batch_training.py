"""Version-bound CSSIM batch bookkeeping; never launches or trains a policy.

The process supervisor writes runtime-exit.json only after main.py exits. A
quiet log or one terminal score is never treated as completion of a batch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'PythonCode'))
import commentjson
from parse_match_score import parse_output

RED = 'TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RLAgentAlgorithm'
BLUE = 'TeamAlg.BlueModel.Algorithm.RLAgentAlgorithm'
MODEL = 'TeamAlg/FsLutk7wD2RkNgbs/Models/target_iql_v1.json'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         prefix=path.name + '.', suffix='.tmp',
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        # Windows readers/other atomic replacements may briefly hold a handle
        # that denies rename. Keep the old complete JSON until replacement.
        deadline = time.monotonic() + 3
        while True:
            try:
                temporary.replace(path)
                break
            except PermissionError as error:
                if os.name != 'nt' or getattr(error, 'winerror', None) not in (5, 32, 33) or time.monotonic() >= deadline:
                    raise
                time.sleep(.025)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class StatusLock:
    """Serialize status read/modify/write across supervisor and CLI processes."""
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+b')
        self.stream.seek(0)
        if not self.stream.read(1):
            self.stream.write(b'0')
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                while True:
                    try:
                        msvcrt.locking(self.stream.fileno(), msvcrt.LK_LOCK, 1)
                        break
                    except OSError as error:
                        if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                            raise
                        time.sleep(.05)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX)
        except BaseException:
            self.stream.close()
            raise
        return self

    def __exit__(self, *_):
        # Closing releases the OS lock, including when the owner crashes.
        self.stream.close()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def git(repo, *arguments):
    return subprocess.check_output(['git', '-C', str(repo), *arguments]).decode('utf-8').strip()


def section(raw, name):
    matches = [v for k, v in raw.items() if k.split('->')[-1] == name]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise ValueError(f'Expected exactly one {name} section')
    return matches[0]


def validate_config(path, episodes=10, pythoncode_root=None):
    raw = commentjson.loads(Path(path).read_text(encoding='utf-8-sig'))
    global_cfg, scenario = section(raw, 'GlobalConfig'), section(raw, 'ScenarioConfig')
    expected_global = dict(num_worker_threads=1, device='cpu', max_episode=episodes, seed=10098)
    expected_scenario = dict(MaxStepsPerEpisode=500, StepGameTime=.5, TimeDilation=1)
    for values, expected in ((global_cfg, expected_global), (scenario, expected_scenario)):
        for key, value in expected.items():
            if values.get(key) != value or isinstance(values.get(key), bool):
                raise ValueError(f'Unexpected {key}: {values.get(key)!r}; require {value!r}')
    expected_names = [RED + '->ReinforceAgentAlgorithm', BLUE + '->ReinforceAgentAlgorithm']
    if [x.replace('.py->', '->') for x in scenario.get('Algorithm_Names', [])] != expected_names:
        raise ValueError('Unexpected red/blue algorithm entry')
    for side, module in (('Red', RED), ('Blue', BLUE)):
        if global_cfg.get('AlgorithmConfig_' + side, '').replace('.py->', '->') != module + '->AlgorithmConfig':
            raise ValueError(f'Unexpected {side} configuration pointer')
    overrides = [v for k, v in raw.items() if k.replace('.py->', '->') == RED + '->AlgorithmConfig']
    if len(overrides) != 1 or overrides[0].get('mode') != 'hybrid_iql' or overrides[0].get('iql_shadow') is not False:
        raise ValueError('Red must actually enable hybrid_iql, shadow=false')
    configured_model = overrides[0].get('iql_model', '').replace('\\', '/')
    if configured_model != 'Models/target_iql_v1.json':
        if (pythoncode_root is None or not Path(configured_model).is_absolute()
                or Path(configured_model).resolve() != (Path(pythoncode_root) / MODEL).resolve()):
            raise ValueError('Unexpected IQL model path')
    return raw


def verify_sources(repo, cssim, *, expected=None, clean_all=False):
    head = git(repo, 'rev-parse', 'HEAD')
    if expected and head != expected:
        raise ValueError('Repository HEAD changed during batch')
    paths = [] if clean_all else ['--', 'PythonCode', 'tools']
    if git(repo, 'status', '--porcelain', *paths):
        raise ValueError('Repository source is not clean; publish before preparing a batch')
    paths = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z', '--', 'PythonCode']).decode().split('\0')
    managed = [p for p in paths if p and not p.startswith('PythonCode/TaskConfig/')]
    if not managed:
        raise ValueError('No tracked PythonCode files')
    for name in managed:
        relative = Path(name).relative_to('PythonCode')
        expected_hash = sha(repo / name)
        for side in ('Client', 'Server'):
            target = cssim / side / 'Data/AlgData/PythonCode' / relative
            if not target.is_file() or sha(target) != expected_hash:
                raise ValueError(f'Deployed source differs from canonical Git worktree: {target}')
    return head


def protocols(root):
    return sorted(str(p.resolve()) for p in Path(root).glob('*/protocol.jsonl'))


def verify_remote(repo, commit):
    """Bind one fresh origin/main observation at launch, never during polling."""
    if git(repo, 'branch', '--show-current') != 'main':
        raise ValueError('Batch collection requires the published main branch')
    lines = git(repo, 'ls-remote', '--exit-code', 'origin', 'refs/heads/main').splitlines()
    rows = [line.split() for line in lines]
    if (len(rows) != 1 or len(rows[0]) != 2 or rows[0][1] != 'refs/heads/main'
            or rows[0][0] != commit):
        raise ValueError('GitHub origin/main does not match local HEAD; publish before preparing a batch')
    return dict(remote='origin', ref='refs/heads/main', commit=commit, verified_at=now())


def prepare(repo, cssim, config, batch_id, episodes=10):
    repo, cssim, config = Path(repo).resolve(), Path(cssim).resolve(), Path(config).resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', batch_id):
        raise ValueError('Batch ID must be 1..80 letters, digits, underscore or hyphen')
    if episodes != 10:
        raise ValueError('This collection workflow requires exactly 10 episodes')
    batch = repo / 'logs/autotrain' / batch_id
    if batch.exists():
        raise ValueError('Batch ID already exists; resume it or use a new ID')
    raw = validate_config(config, episodes, cssim / 'Client/Data/AlgData/PythonCode')
    commit = verify_sources(repo, cssim, clean_all=True)
    remote_verification = verify_remote(repo, commit)
    # Pass this path to main.py --log-root. Dedicated output prevents another
    # manually started match from being mistaken for this batch.
    log_root = batch / 'logs'
    manifest = dict(version=1, batch_id=batch_id, created_at=now(), repo=str(repo), cssim_root=str(cssim),
                    source_commit=commit, remote_verification=remote_verification,
                    expected_episodes=episodes, config_path=str(config),
                    config_sha256=sha(config), runtime_config=raw, model_sha256=sha(repo / 'PythonCode' / MODEL),
                    log_root=str(log_root), known_protocols=protocols(log_root),
                    known_log_directories=sorted(str(p.resolve()) for p in log_root.glob('*') if p.is_dir()),
                    runtime_exit_path=str(batch / 'runtime-exit.json'))
    write_json(batch / 'manifest.json', manifest)
    return dict(manifest=str(batch / 'manifest.json'), **manifest)


def incremental_simulation(path, cache):
    """Consume only complete appended lines, retaining a possibly partial tail."""
    identity = [path.stat().st_dev, path.stat().st_ino]
    if cache.get('identity') not in (None, identity):
        raise ValueError('Bound simulation log was replaced')
    cache['identity'] = identity
    offset = cache.get('offset', 0)
    limit = path.stat().st_size
    if limit < offset:
        raise ValueError('Bound simulation log was truncated')
    cache.setdefault('done', [])
    cache.setdefault('resets', [])
    cache.setdefault('run_ends', [])
    with path.open('rb') as stream:
        stream.seek(offset)
        while stream.tell() < limit:
            line = stream.readline(limit - stream.tell())
            if not line.endswith(b'\n'):
                break
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('Malformed simulation record')
            if row.get('kind') == 'reset':
                cache['resets'].append({k: row.get(k) for k in ('episode', 'training', 'policies')})
            if row.get('done') is True:
                cache['done'].append(row.get('episode'))
            if row.get('kind') == 'run_end':
                cache['run_ends'].append({k: row.get(k) for k in ('status', 'error_type')})
            cache['last_kind'] = row.get('kind')
            if 'episode' in row:
                cache['last_episode'] = row['episode']
            if 'step' in row:
                cache['last_step'] = row['step']
            cache['offset'] = stream.tell()
    cache['partial_tail'] = cache.get('offset', 0) != limit
    return cache


def first_game_config(path):
    limit = path.stat().st_size
    with path.open('rb') as stream:
        while stream.tell() < limit:
            line = stream.readline(limit - stream.tell())
            if not line.endswith(b'\n'):
                return None
            row = json.loads(line)
            p = row.get('payload')
            if row.get('direction') == 'ue_to_python' and isinstance(p, dict) and p.get('dataArr'):
                return p.get('dataGlobal', {}).get('rSVD1')
    return None


def validate_resets(resets, model_hash):
    for row in resets:
        policies = row.get('policies')
        if row.get('training') is not True or not isinstance(policies, list) or len(policies) != 2:
            raise ValueError('Reset did not prove training mode and actual policies')
        red, blue = policies
        if not isinstance(red, dict) or not isinstance(blue, dict):
            raise ValueError('Malformed reset policy metadata')
        if red.get('mode') != 'hybrid_iql' or red.get('shadow') is not False or red.get('model_sha256') != model_hash:
            raise ValueError('Actually loaded red model differs from batch manifest')
        if not str(blue.get('class', '')).startswith('TeamAlg.BlueModel.'):
            raise ValueError('Actually loaded blue algorithm differs from batch manifest')


def validate_receipt(receipt, manifest, directory):
    for key in ('batch_id', 'source_commit', 'config_sha256'):
        if receipt.get(key) != manifest[key]:
            raise ValueError(f'Runtime exit receipt {key} mismatch')
    if directory is None:
        if receipt.get('log_directory') not in (None, ''):
            raise ValueError('Runtime exit receipt names a missing log directory')
    elif Path(receipt.get('log_directory') or '').resolve() != directory.resolve():
        raise ValueError('Runtime exit receipt log directory mismatch')
    if type(receipt.get('pid')) is not int or receipt['pid'] <= 0:
        raise ValueError('Runtime exit receipt lacks process identity')
    started = datetime.fromisoformat(receipt['started_at'])
    ended = datetime.fromisoformat(receipt['ended_at'])
    created = datetime.fromisoformat(manifest['created_at'])
    if started.tzinfo is None or ended.tzinfo is None or started < created or ended < started:
        raise ValueError('Runtime exit receipt chronology mismatch')
    if type(receipt.get('exit_code')) is not int:
        raise ValueError('Runtime exit receipt lacks exit code')


def validate_runtime_identity(identity, manifest):
    for key in ('batch_id', 'source_commit', 'config_sha256'):
        if identity.get(key) != manifest[key]:
            raise ValueError(f'Runtime run.json {key} mismatch')
    if type(identity.get('process_id')) is not int or identity['process_id'] <= 0:
        raise ValueError('Runtime run.json lacks process identity')


def status(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    with StatusLock(manifest_path.with_name('status.lock')):
        return _status_locked(manifest_path)


def _status_locked(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    repo, cssim = Path(manifest['repo']), Path(manifest['cssim_root'])
    verify_sources(repo, cssim, expected=manifest['source_commit'])
    if sha(manifest['config_path']) != manifest['config_sha256']:
        raise ValueError('Runtime config changed during batch')
    progress_path = manifest_path.with_name('progress.json')
    progress = read_json(progress_path) if progress_path.exists() else {}
    fresh = [p for p in protocols(manifest['log_root']) if p not in manifest['known_protocols']
             and str(Path(p).parent) not in manifest['known_log_directories']]
    if len(fresh) > 1:
        raise ValueError('Multiple new protocol logs; refuse ambiguous batch binding')
    if progress.get('protocol') and fresh != [progress['protocol']]:
        raise ValueError('Bound protocol disappeared or a different runtime started')
    summary = dict(batch_id=manifest['batch_id'], source_commit=manifest['source_commit'],
                   expected_episodes=manifest['expected_episodes'], state='awaiting_runtime',
                   complete_episodes=0, done_episodes=[], ready_to_finish=False)
    if not fresh:
        receipt_path = Path(manifest['runtime_exit_path'])
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            validate_receipt(receipt, manifest, None)
            summary.update(state='partial', runtime_exit_code=receipt['exit_code'],
                           reason='Runtime exited before creating a protocol log')
        return summary
    protocol = Path(fresh[0])
    progress['protocol'] = str(protocol)
    identity = [protocol.stat().st_dev, protocol.stat().st_ino]
    if progress.get('protocol_identity') not in (None, identity):
        raise ValueError('Bound protocol file was replaced')
    progress['protocol_identity'] = identity
    if protocol.stat().st_size < progress.get('protocol_bytes', 0):
        raise ValueError('Bound protocol file was truncated')
    progress['protocol_bytes'] = protocol.stat().st_size
    if progress.get('ue_game_config') is None:
        progress['ue_game_config'] = first_game_config(protocol)
    if progress.get('ue_game_config') not in (None, '10;1'):
        raise ValueError(f'UE overrode batch settings: {progress["ue_game_config"]!r}, require 10;1')
    simulation = protocol.with_name('simulation.jsonl')
    sim = incremental_simulation(simulation, progress.setdefault('simulation', {})) if simulation.exists() else {}
    validate_resets(sim.get('resets', []), manifest['model_sha256'])
    expected = list(range(1, manifest['expected_episodes'] + 1))
    dones = sim.get('done', [])
    resets = [r['episode'] for r in sim.get('resets', [])]
    for values in (dones, resets):
        if len(values) != len(set(values)) or any(v not in expected for v in values):
            raise ValueError('Duplicate or out-of-range episode records')
    receipt_path = Path(manifest['runtime_exit_path'])
    receipt = read_json(receipt_path) if receipt_path.exists() else None
    identity_path = protocol.with_name('run.json')
    runtime_identity = read_json(identity_path) if identity_path.exists() else None
    if runtime_identity:
        validate_runtime_identity(runtime_identity, manifest)
    if receipt:
        validate_receipt(receipt, manifest, protocol.parent)
        if not runtime_identity or receipt['pid'] != runtime_identity['process_id']:
            raise ValueError('Runtime exit receipt does not match actual run.json process')
    # Score parsing is cached until a new terminal record or process exit. It
    # always uses the canonical parser; no timestamp-based score association.
    key = [len(dones), protocol.stat().st_size if receipt else None]
    if progress.get('score_key') != key:
        parsed = parse_output(protocol, manifest['source_commit'])
        progress['score_summary'] = parsed
        progress['score_key'] = key
    parsed = progress['score_summary']
    complete = [r for r in parsed['rounds'] if r['complete']]
    run_ends = sim.get('run_ends', [])
    normal_end = (len(run_ends) == 1 and run_ends[0]['status'] == 'returned'
                  and not run_ends[0]['error_type'] and sim.get('last_kind') == 'run_end')
    ready = bool(runtime_identity and receipt and receipt['exit_code'] == 0
                 and not receipt.get('stop_reason') and not receipt.get('error') and normal_end
                 and dones == expected and resets == expected
                 and not sim.get('partial_tail') and progress.get('ue_game_config') == '10;1'
                 and len(parsed['rounds']) == len(expected) == len(complete)
                 and not parsed['malformed_json_lines'])
    summary.update(state=('complete' if ready else 'partial' if receipt else 'running'),
                   protocol=str(protocol), ue_game_config=progress.get('ue_game_config'),
                   complete_episodes=len(complete), done_episodes=dones,
                   last_episode=sim.get('last_episode'), last_step=sim.get('last_step'),
                   runtime_exit_code=receipt.get('exit_code') if receipt else None,
                   runtime_stop_reason=receipt.get('stop_reason') if receipt else None,
                   runtime_error=receipt.get('error') if receipt else None,
                   normal_run_end=normal_end,
                   ready_to_finish=ready,
                   scores=[dict(episode=r['episode'], red=r['scores']['red']['total'],
                                blue=r['scores']['blue']['total']) for r in complete])
    progress['status'] = summary
    progress['runtime_receipt'] = receipt
    progress['runtime_identity'] = runtime_identity
    write_json(progress_path, progress)
    return summary


def finish(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    current = status(manifest_path)
    if not current['ready_to_finish']:
        raise ValueError(f'Batch is not complete: {current["state"]}; {current["complete_episodes"]}/10 scores')
    manifest = read_json(manifest_path)
    progress = read_json(manifest_path.with_name('progress.json'))
    result = dict(batch_id=manifest['batch_id'], finished_at=now(), source_commit=manifest['source_commit'],
                  remote_verification=manifest['remote_verification'],
                  config_sha256=manifest['config_sha256'], runtime_config=manifest['runtime_config'],
                  model_sha256=manifest['model_sha256'], runtime_receipt=progress['runtime_receipt'],
                  runtime_identity=progress['runtime_identity'],
                  status=current, match=progress['score_summary'])
    output = Path(manifest['repo']) / 'results/batches' / (manifest['batch_id'] + '.json')
    if output.exists():
        old = read_json(output)
        if old['source_commit'] != result['source_commit'] or old['match'] != result['match']:
            raise ValueError('Existing finished batch has different evidence')
    else:
        write_json(output, result)
    return dict(result_path=str(output), **current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--repo', type=Path, default=ROOT)
    p.add_argument('--cssim-root', type=Path, default=Path('C:/CSSIM'))
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--batch-id', required=True)
    for name in ('status', 'finish'):
        sub.add_parser(name).add_argument('manifest', type=Path)
    args = parser.parse_args()
    try:
        result = (prepare(args.repo, args.cssim_root, args.config, args.batch_id) if args.command == 'prepare'
                  else status(args.manifest) if args.command == 'status' else finish(args.manifest))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(json.dumps(dict(state='error', error=str(error)), ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
