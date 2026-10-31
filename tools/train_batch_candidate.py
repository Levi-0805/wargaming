"""Train an unpublished IQL candidate from verified finished batches and history.

Source arguments are ordered explicitly: historical sources first, then batch
reports. The existing extractor reserves the last two valid source sessions for
validation and test. No deployment, promotion or new match score is produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from batch_training import sha, write_json
from parse_match_score import parse_output
from train_target_iql import build_dataset, train

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'PythonCode/TeamAlg/FsLutk7wD2RkNgbs/Models/target_iql_v1.json'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def finished_batch(path):
    """Recheck archived completion evidence and the current protocol bytes."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    report = json.loads(raw.decode('utf-8-sig'))
    status, match = report['status'], report['match']
    expected = list(range(1, 11))
    if (not report.get('finished_at') or status.get('state') != 'complete'
            or status.get('ready_to_finish') is not True or status.get('expected_episodes') != 10
            or status.get('complete_episodes') != 10 or status.get('done_episodes') != expected
            or status.get('ue_game_config') != '10;1'):
        raise ValueError('Batch has not finished all ten training episodes')
    commit = report.get('source_commit', '')
    if not isinstance(commit, str) or re.fullmatch('[0-9a-f]{40}', commit) is None:
        raise ValueError('Finished batch lacks a full source commit')
    if match.get('source_commit') != commit or status.get('source_commit') != commit:
        raise ValueError('Batch source commit evidence disagrees')
    receipt, identity = report['runtime_receipt'], report['runtime_identity']
    if receipt.get('exit_code') != 0 or type(receipt.get('exit_code')) is not int:
        raise ValueError('Runtime did not exit successfully')
    if receipt.get('stop_reason') or receipt.get('error'):
        raise ValueError('Runtime was stopped or reported an error')
    for key in ('batch_id', 'source_commit', 'config_sha256'):
        if receipt.get(key) != report.get(key) or identity.get(key) != report.get(key):
            raise ValueError(f'Runtime identity {key} disagrees with finished batch')
    if type(receipt.get('pid')) is not int or receipt['pid'] <= 0 or receipt['pid'] != identity.get('process_id'):
        raise ValueError('Runtime process identity disagrees')
    source = Path(match['output_file']).resolve()
    if source != Path(status['protocol']).resolve():
        raise ValueError('Batch protocol path evidence disagrees')
    if not receipt.get('log_directory') or Path(receipt['log_directory']).resolve() != source.parent:
        raise ValueError('Runtime receipt log directory disagrees with protocol')
    actual = parse_output(source, commit)
    if (actual['source_sha256'] != match.get('source_sha256') or actual['source_bytes'] != match.get('source_bytes')
            or actual['malformed_json_lines'] or not actual['complete']
            or len(actual['rounds']) != 10 or not all(r['complete'] for r in actual['rounds'])
            or actual['rounds'] != match.get('rounds')):
        raise ValueError('Finished protocol changed or does not contain the recorded ten complete games')
    if path.read_bytes() != raw:
        raise ValueError('Finished report changed during verification')
    return dict(path=str(source), sha256=actual['source_sha256'], bytes=actual['source_bytes'],
                source_commit=commit, batch_reports=[str(path)],
                batch_report_sha256={str(path): hashlib.sha256(raw).hexdigest()},
                rounds=[dict(episode=r['episode'], complete=True, score=r['scores']['red']['total'])
                        for r in actual['rounds']])


def historical_source(path):
    path = Path(path).resolve()
    actual = parse_output(path)
    if actual['malformed_json_lines'] or not any(r['complete'] for r in actual['rounds']):
        raise ValueError('Historical source lacks unambiguous complete rounds or has malformed JSON')
    identity_path = path.with_name('run.json')
    identity = read_json(identity_path) if identity_path.exists() else {}
    commit = identity.get('source_commit')
    if commit is not None and (not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit)):
        raise ValueError('Invalid recorded runtime source commit')
    return dict(path=str(path), sha256=actual['source_sha256'], bytes=actual['source_bytes'],
                source_commit=commit, batch_reports=[], batch_report_sha256={},
                source_commit_evidence='runtime_run_json' if commit else 'unbound historical source',
                runtime_identity_path=str(identity_path) if identity else None,
                runtime_identity_sha256=sha(identity_path) if identity else None,
                rounds=[dict(episode=r['episode'], complete=r['complete'],
                             score=r['scores']['red']['total'] if r['complete'] else None)
                        for r in actual['rounds']])


def completed_report_sources(path):
    """Cumulative ten games may include failed processes, never failed games."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    report = json.loads(raw.decode('utf-8-sig'))
    if report.get('classification') != 'ten_cumulative_games_complete':
        return [finished_batch(path)]
    games, commit = report['games'], report['source_commit']
    if (report.get('completed_games') != 10 or len(games) != 10
            or not re.fullmatch('[0-9a-f]{40}', commit)
            or not re.fullmatch('[0-9a-f]{64}', report['model_sha256'])):
        raise ValueError('Invalid cumulative ten-game version binding')
    groups, known = {}, set()
    attempts = {a['batch_id']: a for a in report['attempts']}
    for ordinal, game in enumerate(games, 1):
        key = (game['protocol'], game['episode'])
        if (key in known or game['ordinal'] != ordinal or game['source_commit'] != commit
                or game['model_sha256'] != report['model_sha256']):
            raise ValueError('Mixed or duplicate cumulative games')
        known.add(key)
        attempt = attempts[game['batch_id']]
        manifest = read_json(attempt['manifest_path'])
        if (manifest['source_commit'] != commit or manifest['model_sha256'] != report['model_sha256']
                or manifest['batch_id'] != game['batch_id']
                or game['episode'] not in attempt['status']['done_episodes']
                or Path(attempt['status']['protocol']).resolve() != Path(game['protocol']).resolve()):
            raise ValueError('Cumulative game lacks matching manifest/done evidence')
        groups.setdefault(game['protocol'], []).append(game)
    result = []
    for protocol, rows in groups.items():
        source = Path(protocol).resolve()
        actual = parse_output(source, commit)
        complete = {r['episode']: r for r in actual['rounds'] if r['complete']}
        if actual['malformed_json_lines'] or set(complete) != {g['episode'] for g in rows}:
            raise ValueError('Protocol complete games differ from cumulative selection')
        for game in rows:
            if game['protocol_sha256'] != actual['source_sha256']:
                raise ValueError('Cumulative protocol SHA changed')
            for side in ('red', 'blue'):
                recorded = complete[game['episode']]['scores'][side]
                if any(game[side][k] != recorded[k] for k in ('kill_score', 'capture_score', 'total')):
                    raise ValueError('Cumulative terminal score changed')
        result.append(dict(path=str(source), sha256=actual['source_sha256'], bytes=actual['source_bytes'],
            source_commit=commit, batch_reports=[str(path)],
            batch_report_sha256={str(path): hashlib.sha256(raw).hexdigest()},
            rounds=[dict(episode=r['episode'], complete=True, score=r['red']['total']) for r in rows]))
    if path.read_bytes() != raw:
        raise ValueError('Cumulative report changed during verification')
    return result


def select_sources(batch_reports, sources=()):
    accepted, rejected, by_hash = [], [], {}
    finished_count = 0
    for kind, items, loader in (('historical', sources, lambda p: [historical_source(p)]),
                                ('batch', batch_reports, completed_report_sources)):
        for item in items:
            try:
                records = loader(item)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                rejected.append(dict(kind=kind, path=str(Path(item).resolve()), reason=str(exc)))
                continue
            for record in records:
                existing = by_hash.get(record['sha256'])
                if existing:
                    if existing['source_commit'] and record['source_commit'] and existing['source_commit'] != record['source_commit']:
                        raise ValueError('Identical protocol bytes have conflicting bound source commits')
                    if record['source_commit']:
                        existing['source_commit'] = record['source_commit']
                        existing['batch_reports'] = sorted(set(existing['batch_reports'] + record['batch_reports']))
                        existing['batch_report_sha256'].update(record['batch_report_sha256'])
                    rejected.append(dict(kind=kind, path=str(Path(item).resolve()), reason='duplicate protocol SHA256'))
                else:
                    by_hash[record['sha256']] = record
                    accepted.append(record)
            finished_count += kind == 'batch'
    if not finished_count:
        raise ValueError('At least one verified finished ten-game batch is required; rejected=' + json.dumps(rejected))
    if len(accepted) < 3:
        raise ValueError('Need at least three distinct valid protocol sources; rejected=' + json.dumps(rejected))
    return dict(sources=accepted, rejected=rejected,
                ordering='Explicit historical source order, followed by explicit finished batch order; duplicates keep first position')


def bind_manifest(dataset, selection):
    path = dataset / 'manifest.json'
    manifest = read_json(path)
    bindings = {r['sha256']: r for r in selection['sources']}
    for source in manifest['sources']:
        evidence = bindings.get(source['sha256'])
        if (evidence is None or Path(source['path']).resolve() != Path(evidence['path']).resolve()
                or source['bytes'] != evidence['bytes']):
            raise ValueError('Extractor source snapshot differs from verified selection')
        source['source_commit'] = evidence['source_commit']
        source['source_commit_evidence'] = evidence.get('source_commit_evidence',
            'verified_finished_batch' if evidence['source_commit'] else 'unbound historical source')
        if evidence.get('runtime_identity_path'):
            source['runtime_identity_path'] = evidence['runtime_identity_path']
            source['runtime_identity_sha256'] = evidence['runtime_identity_sha256']
        source['batch_reports'] = evidence['batch_reports']
        source['batch_report_sha256'] = evidence['batch_report_sha256']
    if len(manifest['sources']) != len(bindings):
        raise ValueError('Extractor did not retain all verified source sessions')
    if sha(dataset / 'decisions.npz') != manifest['dataset_sha256']:
        raise ValueError('Dataset checksum mismatch before provenance binding')
    write_json(path, manifest)
    return manifest


def train_candidate(batch_reports, sources, candidate_id, *, repo=ROOT, steps=3000, seed=10098):
    repo = Path(repo).resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', candidate_id):
        raise ValueError('Candidate ID must be 1..80 letters, digits, underscore or hyphen')
    output = repo / 'logs/autotrain-candidates' / candidate_id
    if output.exists():
        raise ValueError('Candidate directory already exists; choose a new ID')
    baseline = repo / BASELINE
    baseline_sha = sha(baseline)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    selection = select_sources(batch_reports, sources)
    output.mkdir(parents=True)
    write_json(output / 'selection.json', selection)
    dataset = output / 'dataset'
    build_dataset([Path(r['path']) for r in selection['sources']], dataset)
    manifest = bind_manifest(dataset, selection)
    candidate = output / 'candidate.json'
    report = train(dataset, candidate, steps=steps, seed=seed)
    training_report = dataset / 'training-report.json'
    actual_report = read_json(training_report)
    if report != actual_report or actual_report['model_sha256'] != sha(candidate):
        raise ValueError('Training report does not match candidate artifact')
    if sha(baseline) != baseline_sha:
        raise ValueError('Baseline changed during training; candidate comparison context is ambiguous')
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != commit:
        raise ValueError('Repository HEAD changed during candidate training')
    review = dict(candidate_id=candidate_id, state='trained_unpublished', training_base_commit=commit,
                  wrapper_sha256=sha(Path(__file__)),
                  baseline_model=str(baseline), baseline_model_sha256=baseline_sha,
                  candidate_model=str(candidate), candidate_model_sha256=sha(candidate),
                  training_report=str(training_report), training_report_sha256=sha(training_report),
                  dataset_sha256=manifest['dataset_sha256'], manifest_sha256=sha(dataset / 'manifest.json'),
                  selection=selection, split_sessions=manifest['split_sessions'],
                  offline_metrics=actual_report['offline_metrics'], promoted=False,
                  new_match_outcomes=False,
                  next_steps=['Review held-out metrics and run integration replay.',
                              'Commit, push and verify any approved candidate before deployment.',
                              'Collect fresh controlled matches to compare scores; offline agreement is not improvement.'])
    write_json(output / 'review.json', review)
    return dict(review_path=str(output / 'review.json'), **review)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', type=Path, nargs='+', required=True)
    parser.add_argument('--sources', type=Path, nargs='*', default=[])
    parser.add_argument('--candidate-id', required=True)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=10098)
    args = parser.parse_args()
    try:
        result = train_candidate(args.batches, args.sources, args.candidate_id, steps=args.steps, seed=args.seed)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f'Candidate training refused: {exc}\n')
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
