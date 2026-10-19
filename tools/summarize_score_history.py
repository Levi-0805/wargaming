"""Inventory completed matches; score shortfalls are not causal diagnoses.

Cached results are reused only when source size and SHA256 both still match.
The minimum score gate describes recorded results, not unseen future games.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

from parse_match_score import parse_output


def source_digest(path, limit):
    digest = hashlib.sha256()
    consumed = 0
    with path.open('rb') as stream:
        while consumed < limit:
            block = stream.read(min(4 * 1024 * 1024, limit - consumed))
            if not block:
                raise RuntimeError(f'Source shrank during inspection: {path}')
            consumed += len(block)
            digest.update(block)
    return digest.hexdigest()


def normalized_round(row):
    red = row.get('scores', {}).get('red') or row.get('red')
    return {'episode': row['episode'], 'complete': row.get('complete', False),
            'frames': row.get('frames'), 'last_step': row.get('last_step'),
            'red': {k: red.get(k) for k in ('kill_score', 'capture_score', 'total')} if red else None,
            'capture': row.get('capture', row.get('final_capture', []))}


def summarize(log_root, cache=None, threshold=4000, minimum_rounds=20):
    cached = {str(Path(r['source'])).casefold(): r for r in (cache or {}).get('runs', [])}
    runs, inventory = [], []
    for source in sorted(log_root.glob('*/protocol.jsonl')):
        limit = source.stat().st_size
        previous = cached.get(str(source).casefold())
        digest = source_digest(source, limit)
        valid = (previous is not None and previous.get('source_bytes') == limit
                 and previous.get('source_sha256') == digest
                 and all(isinstance(r.get('complete'), bool) and 'red' in r for r in previous['rounds']))
        parsed = previous if valid else parse_output(source)
        # The parser takes its own bounded snapshot. Record that exact snapshot.
        rows = [normalized_round(r) for r in parsed['rounds']]
        complete = [r for r in rows if r['complete'] and r['red'] and r['red']['total'] is not None]
        scores = [r['red']['total'] for r in complete]
        run = {'run': source.parent.name, 'source': str(source),
               'source_bytes': parsed['source_bytes'], 'source_sha256': parsed['source_sha256'],
               'source_commit': parsed.get('source_commit'),
               'source_commit_evidence': parsed.get('source_commit_evidence', 'unbound historical source'),
               'malformed_json_lines': parsed.get('malformed_json_lines', []),
               'rounds': rows, 'complete_rounds': len(complete), 'incomplete_rounds': len(rows) - len(complete),
               'mean': statistics.mean(scores) if scores else None,
               'median': statistics.median(scores) if scores else None,
               'min': min(scores) if scores else None, 'max': max(scores) if scores else None,
               'below_4000': sum(s < threshold for s in scores),
               'at_least_4000': sum(s >= threshold for s in scores),
               'mean_destruction': statistics.mean(r['red']['kill_score'] for r in complete) if complete else None,
               'mean_capture': statistics.mean(r['red']['capture_score'] for r in complete) if complete else None,
               'capture_reward_distribution': dict(Counter(r['red']['capture_score'] for r in complete)),
               'recorded_score_gate': ('insufficient_rounds' if len(scores) < minimum_rounds
                   else 'failed' if min(scores) < threshold else 'passed_observed_sample_only')}
        runs.append(run)
        for row in complete:
            red = row['red']
            reason = ('threshold_met' if red['total'] >= threshold else
                      'no_terminal_capture_reward' if red['capture_score'] == 0 else
                      'partial_capture_and_insufficient_destruction' if red['capture_score'] < 3200 else
                      'full_capture_but_insufficient_destruction')
            inventory.append({'run': run['run'], 'episode': row['episode'], **red,
                'gap_to_4000': max(0, threshold - red['total']), 'outcome_category': reason,
                'last_step': row['last_step'], 'source_sha256': run['source_sha256'],
                'source_commit': run['source_commit'],
                'capture': json.dumps(row['capture'], ensure_ascii=False)})
        print(source.parent.name, 'cached' if valid else 'parsed', len(complete), run['mean'], flush=True)
    low = [r for r in inventory if r['total'] < threshold]
    return {'score_definition': 'terminal Score[*].SumWorth + Reward; complete side-attributed settlements only',
        'scope': str(log_root), 'threshold': threshold, 'minimum_rounds_for_score_gate': minimum_rounds,
        'limitations': ['Outcome categories explain the numerical shortfall, not the causal policy error.',
                       'No historical source version is inferred from timestamps.',
                       'An observed score pass cannot guarantee future scores or identify defense coverage.'],
        'summary': {'run_directories': len(runs), 'complete_rounds': len(inventory),
            'incomplete_rounds': sum(r['incomplete_rounds'] for r in runs),
            'below_4000': len(low), 'at_least_4000': len(inventory) - len(low),
            'low_score_categories': dict(Counter(r['outcome_category'] for r in low)),
            'low_score_capture_rewards': dict(Counter(r['capture_score'] for r in low)),
            'runs_with_source_commit': sum(bool(r['source_commit']) for r in runs)},
        'runs': runs}, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log_root', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cache = json.loads(args.cache.read_text(encoding='utf-8-sig')) if args.cache else None
    result, rows = summarize(args.log_root, cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if rows:
        with args.out.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
