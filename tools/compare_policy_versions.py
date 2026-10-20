"""Compare two local algorithm versions on recorded public observations.

This does not run UE or simulate new scores. The baseline directory must be
an inspected algorithm package exported from the stated Git revision.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import importlib.util
import json
import logging
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'PythonCode'))
from cssim.algorithms import AlgorithmContext
from cssim.config import RuntimeConfig
from cssim.environment.env import CssimEnvironment
from cssim.protocol import ActionSetFactory
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RuleAgentAlgorithm import ReinforceAgentAlgorithm

PACKAGE = 'PythonCode/TeamAlg/FsLutk7wD2RkNgbs/Algorithm'


def hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob('*.py'))}


def load_baseline(directory, revision):
    revision = subprocess.check_output(
        ['git', 'rev-parse', '--verify', revision + '^{commit}'], cwd=ROOT, text=True).strip()
    tracked = subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', revision, '--', PACKAGE], cwd=ROOT, text=True)
    expected = {}
    for name in tracked.splitlines():
        relative = Path(name).relative_to(PACKAGE)
        if len(relative.parts) != 1 or relative.suffix != '.py':
            continue
        raw = subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=ROOT)
        expected[relative.name] = hashlib.sha256(raw).hexdigest()
    if not expected or hashes(directory) != expected:
        raise ValueError('Baseline files do not exactly match the stated Git revision.')
    name = '_cssim_comparison_baseline'
    spec = importlib.util.spec_from_file_location(
        name, directory / '__init__.py', submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    cls = importlib.import_module(name + '.RuleAgentAlgorithm').ReinforceAgentAlgorithm
    return cls, revision, expected


def new_stats():
    return dict(objectives=[], unsafe_departures=[], pre_hq_diversions=0,
                invalid_vehicle_fire=0, invalid_uav_fire=0, reserve_waits={},
                column_holds=0, entry_holds=0)


def record_policy(stats, rule, state, actions):
    route = rule._route
    obj = route.objective(state)
    uid = obj.uid if obj else None
    if not stats['objectives'] or stats['objectives'][-1]['uid'] != uid:
        if stats['objectives']:
            old = state.key_object_by_uid(stats['objectives'][-1]['uid'])
            count = route.enemy_count(state, old) if old else None
            if count is None or count > 0:
                stats['unsafe_departures'].append(
                    dict(step=state.step, left_uid=old.uid if old else None, enemies=count))
        stats['objectives'].append(dict(step=state.step, uid=uid,
                                       name=obj.raw.get('rSVD1') if obj else None))
    stats['pre_hq_diversions'] += bool(obj and not route.hq_secured and not route.is_hq(obj))
    stats['column_holds'] += len(rule._column.held)
    stats['entry_holds'] += len(rule._entry_waves.held)
    for held in getattr(rule._formation, 'reserve_held', ()):
        key = f'{held}:{uid}'
        wait = stats['reserve_waits'].setdefault(key, dict(first=state.step, last=state.step, count=0))
        wait['last'] = state.step
        wait['count'] += 1
    for unit, action in zip(state.agents, actions):
        if not unit.alive:
            assert action.command == 'Idle'
        if unit.entity_type in rule.VEHICLES and action.command == 'NormalAttacking':
            eligible = {e.uid for e in rule._vehicle_fire.eligible(state, unit)}
            stats['invalid_vehicle_fire'] += action.target_uid not in eligible
        if unit.entity_type == rule._UAV and action.command in {'NormalAttacking', 'SelfDestruct'}:
            mission = rule._uav._missions.get(unit.uid)
            target = mission[0] if action.command == 'SelfDestruct' and mission else action.target_uid
            stats['invalid_uav_fire'] += target not in unit.perceived_uids()
    for command in rule.take_selected_special_commands():
        command.parameters()


def compare(source, baseline_dir, revision, selected):
    baseline, revision, baseline_hashes = load_baseline(baseline_dir, revision)
    candidate_hashes = hashes(ROOT / PACKAGE)
    config = RuntimeConfig(ROOT, {}, '127.0.0.1', 21051, ('red', 'blue'),
                           500, 1, 1., 60., .5, 0, 'cpu', 'WARNING')
    environments = {key: CssimEnvironment(config) for key in ('baseline', 'candidate')}
    classes = dict(baseline=baseline, candidate=ReinforceAgentAlgorithm)
    rounds, current, previous, episode, rules, pending = [], None, None, 0, {}, {}
    limit, consumed, digest = source.stat().st_size, 0, hashlib.sha256()
    with source.open('rb') as stream:
        while consumed < limit:
            line = stream.readline(limit - consumed)
            if not line:
                break
            consumed += len(line)
            digest.update(line)
            row = json.loads(line)
            packet = row.get('payload')
            if not isinstance(packet, dict):
                continue
            if row.get('direction') == 'ue_to_python':
                if packet.get('propertyArr'):
                    for env in environments.values():
                        env._build_roster(packet)
                if not packet.get('dataArr'):
                    continue
                step = packet['dataGlobal']['timeCnt']
                if previous is None or step < previous:
                    episode += 1
                    for env in environments.values():
                        env.episode = episode
                    current, rules, pending = None, {}, {}
                    if not selected or episode in selected:
                        current = dict(episode=episode, frames=0, baseline_recorded_matches=0,
                                       baseline_recorded_mismatches=0, changed_commands=0,
                                       changes_by_kind=Counter(), changes_by_command=Counter(),
                                       modes={key: new_stats() for key in environments})
                        rounds.append(current)
                        print(f'compare episode {episode}', flush=True)
                previous = step
                if current is None:
                    continue
                wires, command_names = {}, {}
                for key, env in environments.items():
                    # Each policy gets its own parsed entities and mutable raw fields.
                    snapshot = env._parse_state(packet)
                    state = env.build_team_state(snapshot, 0)
                    if key not in rules:
                        rules[key] = classes[key](AlgorithmContext(
                            0, state.agents, state.visible_opponents,
                            ActionSetFactory.standard(), 'cpu', {}, state.commanders))
                    rule = rules[key]
                    actions = rule.act(state)
                    assert len(actions) == len(state.agents)
                    wires[key] = {u.uid: a.to_ue_string(u.uid, snapshot.entities)
                                  for u, a in zip(state.agents, actions)}
                    command_names[key] = {u.uid: a.command for u, a in zip(state.agents, actions)}
                    record_policy(current['modes'][key], rule, state, actions)
                pending = wires['baseline']
                for unit in state.agents:
                    if wires['baseline'][unit.uid] != wires['candidate'][unit.uid]:
                        current['changed_commands'] += 1
                        current['changes_by_kind'][unit.entity_type] += 1
                        pair = command_names['baseline'][unit.uid] + ' -> ' + command_names['candidate'][unit.uid]
                        current['changes_by_command'][pair] += 1
                current['frames'] += 1
            elif row.get('direction') == 'python_to_ue' and current is not None:
                for action in packet.get('StringActions', ()) or ():
                    uid = int(action.split(';')[-1])
                    if uid in pending:
                        key = 'baseline_recorded_matches' if pending[uid] == action else 'baseline_recorded_mismatches'
                        current[key] += 1
                pending = {}
    if candidate_hashes != hashes(ROOT / PACKAGE) or baseline_hashes != hashes(baseline_dir):
        raise RuntimeError('Algorithm source changed during comparison.')
    return dict(source=str(source), source_bytes=consumed, source_sha256=digest.hexdigest(),
                baseline_commit=revision, baseline_files=baseline_hashes,
                candidate_files=candidate_hashes, episodes=rounds,
                limitation='Historical observations only; changed actions do not predict new movement, survival or scores.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--baseline-dir', type=Path, required=True)
    p.add_argument('--baseline-commit', required=True)
    p.add_argument('--episodes', nargs='*', type=int, default=[])
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    logging.disable(logging.CRITICAL)
    result = compare(args.source, args.baseline_dir.resolve(), args.baseline_commit, set(args.episodes))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    failures = 0
    for row in result['episodes']:
        failures += row['baseline_recorded_mismatches']
        candidate = row['modes']['candidate']
        failures += sum(candidate[k] for k in
                        ('invalid_vehicle_fire', 'invalid_uav_fire', 'pre_hq_diversions'))
        failures += any(v['last'] - v['first'] >= 28 for v in candidate['reserve_waits'].values())
    print(f"episodes={len(result['episodes'])} frames={sum(r['frames'] for r in result['episodes'])} failures={failures}")
    return 1 if failures or not result['episodes'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
