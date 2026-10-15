"""Compare rule decisions on recorded public views; never simulate a score.

Reads a bounded snapshot of protocol.jsonl without connecting to CSSIM. The
legacy combination must match the run being audited before attributing a
change to these switches. Other versions are useful as held-out observations,
but their recorded actions need not match this baseline.
"""
import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'PythonCode'))

from cssim.algorithms import AlgorithmContext
from cssim.config import RuntimeConfig
from cssim.environment.env import CssimEnvironment
from cssim.protocol import ActionSetFactory
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RuleAgentAlgorithm import ReinforceAgentAlgorithm

SWITCHES = ('enable_preservation_pause', 'enable_shared_contact_maneuver',
            'enable_forced_entry_under_contact')
VARIANTS = {'legacy': dict.fromkeys(SWITCHES, True)}
for switch in SWITCHES:
    VARIANTS['without_' + switch.removeprefix('enable_')] = {
        **VARIANTS['legacy'], switch: False,
    }
VARIANTS['default'] = dict.fromkeys(SWITCHES, False)


def instrument(rule, counters):
    """Observe actual invoked branches; do not call a decision function twice."""
    preserve, entry, engage = rule._readiness.choose, rule._hq_entry.choose, rule._engage

    def preservation(*args, **kwargs):
        result = preserve(*args, **kwargs)
        if result is not None:
            counters['preservation_actions'] += 1
        return result

    def entrance(state, unit, *args, **kwargs):
        result = entry(state, unit, *args, **kwargs)
        if result is not None:
            counters['entrance_actions'] += 1
            local = state.perceived_opponents(unit)
            if rule._has_ammo(unit) and any(e.alive and e.entity_type in rule._urban.GROUND for e in local):
                counters['entrance_over_local_ground_contact'] += 1
        return result

    def engagement(state, unit, contacts, local):
        result = engage(state, unit, contacts, local)
        if not any(e.uid in local for e in contacts):
            counters['shared_only_engagement_actions'] += 1
        return result

    rule._readiness.choose = preservation
    rule._hq_entry.choose = entrance
    rule._engage = engagement


def compare(path):
    limit, read = path.stat().st_size, 0
    digest = hashlib.sha256()
    episode, previous = 0, None
    env = None
    rules, pending, rounds = {}, {}, []
    totals = Counter()
    mismatches = []
    with path.open('rb') as source:
        while read < limit:
            line = source.readline(limit - read)
            read += len(line)
            digest.update(line)
            if not line:
                break
            try:
                row = json.loads(line)
            except ValueError:
                if read == limit and not line.endswith(b'\n'):
                    break  # A concurrently written final line is not a frame.
                raise
            packet = row.get('payload', {})
            if not isinstance(packet, dict):
                continue
            if row.get('direction') == 'ue_to_python':
                if packet.get('propertyArr'):
                    env = CssimEnvironment(RuntimeConfig(ROOT, {}, '127.0.0.1', 21051,
                        ('red', 'blue'), 500, 5, 4, 60, .5, 0, 'cpu', 'WARNING'))
                    env._build_roster(packet)
                if packet.get('dataArr'):
                    step = packet['dataGlobal']['timeCnt']
                    if previous is None or step < previous:
                        episode += 1
                        rules = {}
                        current = {'episode': episode, 'frames': 0, 'last_step': step,
                                   'variants': {name: Counter() for name in VARIANTS},
                                   'recorded_score_packets_red_then_blue': []}
                        rounds.append(current)
                        print(path.parent.name, 'episode', episode, flush=True)
                    previous = step
                    if env is None:
                        raise ValueError('Frame received without a roster')
                    env.episode = episode
                    snapshot = env._parse_state(packet)
                    state = env.build_team_state(snapshot, 0)  # Never pass hidden blue state to policies.
                    if not rules:
                        context = AlgorithmContext(0, state.agents, state.visible_opponents,
                            ActionSetFactory.standard(), 'cpu', {}, state.commanders)
                        for name, options in VARIANTS.items():
                            rules[name] = ReinforceAgentAlgorithm(context, **options)
                            instrument(rules[name], current['variants'][name])
                    generated = {}
                    for name, rule in rules.items():
                        counts = current['variants'][name]
                        actions = rule.act(state)
                        encoded = {}
                        for unit, action in zip(state.agents, actions):
                            encoded[str(unit.uid)] = action.to_ue_string(unit.uid, snapshot.entities)
                            counts['serialized_actions'] += 1
                            if unit.alive and unit.entity_type in rule._urban.GROUND:
                                counts['ground_' + action.command] += 1
                                if action.command == 'NormalAttacking':
                                    local = {e.uid for e in state.perceived_opponents(unit) if e.alive}
                                    counts['local_attack_requests' if action.target_uid in local else 'shared_attack_requests'] += 1
                        generated[name] = encoded
                    old = generated['legacy']
                    for name, encoded in generated.items():
                        current['variants'][name]['changed_commands_from_legacy'] += sum(
                            value != old[uid] for uid, value in encoded.items())
                        for unit in state.agents:
                            if unit.entity_type == rules[name]._UAV:
                                assert encoded[str(unit.uid)] == old[str(unit.uid)], 'Unexpected UAV change'
                    default = current['variants']['default']
                    assert not any(default[key] for key in ('preservation_actions',
                        'entrance_over_local_ground_contact', 'shared_only_engagement_actions', 'shared_attack_requests'))
                    pending = old
                    current['frames'] += 1
                    current['last_step'] = step
                if 'Score' in packet and rounds:
                    kills = sum(item.get('SumWorth', 0) for item in packet['Score'].values())
                    capture = packet.get('Reward', 0)
                    current['recorded_score_packets_red_then_blue'].append(
                        {'kill_score': kills, 'capture_score': capture, 'total': kills + capture})
            elif row.get('direction') == 'python_to_ue' and row.get('channel') == 'frame' and pending:
                recorded = {a.rsplit(';', 1)[-1]: a for a in packet.get('StringActions', [])}
                for uid, action in pending.items():
                    if uid not in recorded:
                        totals['legacy_missing_recorded_commands'] += 1
                    elif recorded[uid] == action:
                        totals['legacy_matches_recorded'] += 1
                    else:
                        totals['legacy_mismatches_recorded'] += 1
                        if len(mismatches) < 5:
                            mismatches.append({'episode': episode, 'step': previous, 'uid': uid,
                                               'recorded': recorded[uid], 'generated': action})
                pending = {}
    return {'source': str(path), 'bytes_read': read, 'snapshot_sha256': digest.hexdigest(),
            'baseline_validation': dict(totals), 'mismatch_examples': mismatches, 'rounds': rounds}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('protocol', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-recorded-match', action='store_true')
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    output = {'scope': 'Decision replay on fixed recorded public observations. No UE physics, new match, score or survival prediction.',
              'variants': VARIANTS, 'runs': [compare(path) for path in args.protocol]}
    output['source_hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (ROOT / 'PythonCode/TeamAlg/FsLutk7wD2RkNgbs/Algorithm').glob('*.py')}
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.require_recorded_match:
        for run in output['runs']:
            counts = run['baseline_validation']
            if (not counts.get('legacy_matches_recorded') or counts.get('legacy_mismatches_recorded')
                    or counts.get('legacy_missing_recorded_commands')):
                raise SystemExit('Recorded commands do not match the legacy baseline; see output diagnostics.')
