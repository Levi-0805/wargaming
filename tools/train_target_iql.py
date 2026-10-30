"""Build actual logged target decisions and train an offline IQL sub-policy.

No policy replay labels or synthetic outcomes. Intermediate navigation remains
the behavior controller: this is a semi-Markov target-selection problem, not
an end-to-end replacement. Every trajectory receives its actual terminal team
score once, discounted over elapsed simulator steps. No online rollout occurs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'PythonCode'))
from cssim.config import RuntimeConfig
from cssim.environment.env import CssimEnvironment
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.TargetFeatures import (
    VERSION, STATE_DIM, TARGET_DIM, KINDS, state_features, target_features, local_targets, ammo)
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.VehicleFireControl import VehicleFireControl
from parse_match_score import parse_output

MAX_TARGETS = 16
FEATURE_SOURCE = ROOT / 'PythonCode/TeamAlg/FsLutk7wD2RkNgbs/Algorithm/TargetFeatures.py'


def config():
    return RuntimeConfig(ROOT, {}, '127.0.0.1', 21051, ('red', 'blue'),
                         500, 1, 1., 60., .5, 0, 'cpu', 'WARNING')


def aligned_actions(packet, expected_step):
    """Do not attach a stale, duplicated or malformed action to a new state."""
    if packet.get('DataCmd') != 'step' or packet.get('TimeStep') != expected_step:
        return False
    wires = packet.get('StringActions')
    if not isinstance(wires, list) or not wires:
        return False
    try:
        ids = [int(wire.rsplit(';', 1)[1]) for wire in wires if isinstance(wire, str)]
    except (ValueError, IndexError):
        return False
    return len(ids) == len(wires) == len(set(ids))


def episode_transitions(trajectories, terminal_step, score, gamma):
    """Return actual-decision SMDP samples; no reward on every frame."""
    for records in trajectories.values():
        for index, row in enumerate(records):
            final = index + 1 == len(records)
            end = terminal_step if final else records[index+1]['step']
            elapsed = max(0 if final else 1, end - row['step'])
            discount = gamma ** elapsed
            yield dict(s=row['s'], a=row['a'], mask=row['mask'], label=row['label'],
                       kind=row['kind'], target_kind=row['target_kind'],
                       next_s=np.zeros(STATE_DIM, np.float32) if final else records[index+1]['s'],
                       reward=score / 6000 * discount if final else 0.,
                       discount=0. if final else discount, terminal=final)


def build_dataset(sources, output, gamma=.997, stride=4):
    if not 0 < gamma <= 1 or stride < 1:
        raise ValueError('Invalid discount or sampling stride')
    all_rows, groups, provenance, rejected, seen_sources, seen_episodes = [], [], [], [], set(), set()
    for source in sources:
        audit = parse_output(source)
        if audit['source_sha256'] in seen_sources:
            rejected.append({'source': str(source), 'reason': 'duplicate source SHA256'})
            continue
        seen_sources.add(audit['source_sha256'])
        if audit['malformed_json_lines']:
            rejected.append({'source': str(source), 'reason': 'malformed JSON'})
            continue
        complete = {r['episode']: r for r in audit['rounds'] if r['complete']}
        env, episode, seen_frame, previous_step, current, view = CssimEnvironment(config()), 0, False, None, {}, None
        decisions, sent_frames, ambiguous = {}, set(), set()
        counts = {'attacks': 0, 'accepted': 0, 'not_local_or_ineligible': 0, 'sampled_out': 0}
        digest = hashlib.sha256(); consumed = 0
        print(f'build {source.parent.name}: complete={len(complete)}', flush=True)
        with source.open('rb') as stream:
            while consumed < audit['source_bytes']:
                line = stream.readline(audit['source_bytes'] - consumed)
                if not line:
                    break
                consumed += len(line); digest.update(line)
                row = json.loads(line); packet = row.get('payload', {})
                if not isinstance(packet, dict):
                    continue
                if row.get('direction') == 'ue_to_python':
                    if packet.get('propertyArr'):
                        if episode == 0 or seen_frame:
                            episode += 1
                        previous_step, seen_frame, view = None, False, None
                        env._build_roster(packet)
                    if not packet.get('dataArr'):
                        continue
                    step = packet['dataGlobal']['timeCnt']
                    if episode == 0 or previous_step is not None and step < previous_step:
                        episode += 1
                    previous_step, seen_frame = step, True
                    env.episode = episode
                    if episode not in complete:
                        view = None
                        continue
                    snap = env._parse_state(packet)
                    view = env.build_team_state(snap, 0)
                    current = decisions.setdefault(episode, {})
                elif row.get('direction') == 'python_to_ue' and view is not None:
                    wires = packet.get('StringActions', ()) or ()
                    if not wires:
                        continue
                    key = (episode, view.step)
                    if key in sent_frames or not aligned_actions(packet, view.step):
                        ambiguous.add(episode)
                        continue
                    sent_frames.add(key)
                    agents = {u.uid: u for u in view.agents}
                    for wire in wires:
                        parts = wire.split(';')
                        if len(parts) != 3 or parts[0] != 'ActionSet2::NormalAttacking':
                            continue
                        target_uid, uid = int(parts[1]), int(parts[2])
                        unit = agents.get(uid)
                        if unit is None or unit.entity_type not in KINDS or not unit.alive:
                            continue
                        counts['attacks'] += 1
                        targets = (list(VehicleFireControl().eligible(view, unit))
                                   if unit.entity_type in VehicleFireControl.VEHICLES else local_targets(view, unit))
                        targets = [t for t in targets if t.entity_type in KINDS]
                        target = next((t for t in targets if t.uid == target_uid), None)
                        if target is None or ammo(unit) == 0 or unit.communication_ok() is False:
                            counts['not_local_or_ineligible'] += 1
                            continue
                        history = current.setdefault(uid, [])
                        if history and view.step - history[-1]['step'] < stride and history[-1]['target_uid'] == target_uid:
                            counts['sampled_out'] += 1
                            continue
                        targets = sorted(targets, key=lambda t: (np.linalg.norm(t.position-unit.position), t.uid))[:MAX_TARGETS]
                        if target_uid not in {t.uid for t in targets}:
                            targets[-1] = target
                        options = np.zeros((MAX_TARGETS, TARGET_DIM), np.float32)
                        mask = np.zeros(MAX_TARGETS, bool)
                        for i, t in enumerate(targets):
                            options[i] = target_features(view, unit, t); mask[i] = True
                        history.append(dict(step=view.step, s=state_features(view, unit), a=options,
                                            mask=mask, label=next(i for i, t in enumerate(targets) if t.uid == target_uid),
                                            target_uid=target_uid, kind=KINDS.index(unit.entity_type),
                                            target_kind=KINDS.index(target.entity_type)))
                        counts['accepted'] += 1
        if consumed != audit['source_bytes'] or digest.hexdigest() != audit['source_sha256']:
            raise ValueError(f'Source snapshot changed during extraction: {source}')
        session = len(provenance)
        provenance.append({'path': str(source), 'sha256': audit['source_sha256'],
                           'bytes': consumed, 'source_commit': None, 'counts': counts,
                           'rounds': [{'episode': r['episode'], 'complete': r['complete'],
                                       'score': r['scores']['red']['total'] if r['scores']['red'] else None}
                                      for r in audit['rounds']]})
        for ep, trajectories in decisions.items():
            if ep in ambiguous:
                rejected.append({'source': str(source), 'episode': ep, 'reason': 'misaligned, malformed or duplicate action packet'})
                continue
            actual = complete[ep]
            rows = list(episode_transitions(trajectories, actual['last_step'], actual['scores']['red']['total'], gamma))
            fingerprint = hashlib.sha256()
            for r in rows:
                fingerprint.update(r['s'].tobytes()); fingerprint.update(r['a'].tobytes())
                fingerprint.update(str((r['label'], r['reward'], r['discount'])).encode())
            if not rows or fingerprint.hexdigest() in seen_episodes:
                rejected.append({'source': str(source), 'episode': ep, 'reason': 'empty or duplicate decision trajectory'})
                continue
            seen_episodes.add(fingerprint.hexdigest())
            group = len(groups)
            groups.append({'session': session, 'episode': ep, 'score': actual['scores']['red']['total'],
                           'decisions': len(rows), 'fingerprint': fingerprint.hexdigest()})
            all_rows.extend(dict(r, group=group, session=session) for r in rows)
    sessions = sorted({r['session'] for r in all_rows})
    if len(sessions) < 3:
        raise ValueError('Need at least three distinct nonempty sessions for train/validation/test')
    split = np.asarray([2 if r['session'] == sessions[-1] else 1 if r['session'] == sessions[-2] else 0
                        for r in all_rows], np.int8)
    output.mkdir(parents=True, exist_ok=True)
    keys = ('s', 'a', 'mask', 'label', 'kind', 'target_kind', 'next_s', 'reward', 'discount', 'terminal', 'group')
    arrays = {k: np.asarray([r[k] for r in all_rows]) for k in keys}
    np.savez_compressed(output / 'decisions.npz', **arrays, split=split)
    manifest = dict(feature_version=VERSION, gamma=gamma, stride=stride, sources=provenance, groups=groups,
                    rejected=rejected, split_sessions={'train': sessions[:-2], 'validation': sessions[-2], 'test': sessions[-1]},
                    split_counts={str(i): int((split == i).sum()) for i in range(3)},
                    dataset_sha256=hashlib.sha256((output / 'decisions.npz').read_bytes()).hexdigest(),
                    extraction_base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    features_sha256=hashlib.sha256((ROOT / 'PythonCode/TeamAlg/FsLutk7wD2RkNgbs/Algorithm/TargetFeatures.py').read_bytes()).hexdigest(),
                    reward='One terminal team score / 6000 per unit decision trajectory, discounted by elapsed steps; no synthetic rewards',
                    limitation='Target-selection SMDP under recorded navigation and teammates; historical source commits unbound.')
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return manifest


def train(dataset, output, steps=3000, seed=10098):
    import torch
    from torch import nn
    torch.set_num_threads(2); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('feature_version') != VERSION or manifest.get('features_sha256') !=
            hashlib.sha256(FEATURE_SOURCE.read_bytes()).hexdigest()):
        raise ValueError('Dataset feature schema/source mismatch; rebuild the dataset')
    if hashlib.sha256((dataset / 'decisions.npz').read_bytes()).hexdigest() != manifest['dataset_sha256']:
        raise ValueError('Dataset checksum mismatch')
    with np.load(dataset / 'decisions.npz', allow_pickle=False) as source:
        data = {k: source[k].copy() for k in source.files}
    train_ids = np.flatnonzero(data['split'] == 0)
    if len(train_ids) < 256 or steps < 1:
        raise ValueError('Insufficient training data or invalid update count')
    def network(width):
        return nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    q1, q2, value, actor = network(STATE_DIM+TARGET_DIM), network(STATE_DIM+TARGET_DIM), network(STATE_DIM), network(STATE_DIM+TARGET_DIM)
    tq1, tq2 = copy.deepcopy(q1), copy.deepcopy(q2)
    opts = [torch.optim.Adam(m.parameters(), lr=3e-4) for m in (value, q1, q2, actor)]
    def measure(model, split_id):
        ids = np.flatnonzero((data['split'] == split_id) & (data['mask'].sum(axis=1) > 1))
        total_nll, correct = 0., 0
        with torch.no_grad():
            for start in range(0, len(ids), 512):
                rows = ids[start:start+512]
                s = torch.as_tensor(data['s'][rows], dtype=torch.float32)
                a = torch.as_tensor(data['a'][rows], dtype=torch.float32)
                logits = model(torch.cat((s[:, None, :].expand(-1, MAX_TARGETS, -1), a), dim=-1)).squeeze(-1)
                logits.masked_fill_(~torch.as_tensor(data['mask'][rows]), -1e9)
                labels = torch.as_tensor(data['label'][rows], dtype=torch.long)
                total_nll += nn.functional.cross_entropy(logits, labels, reduction='sum').item()
                correct += int((logits.argmax(-1) == labels).sum())
        return {'multiple_target_samples': len(ids), 'behavior_nll': total_nll/max(1, len(ids)),
                'recorded_choice_agreement': correct/max(1, len(ids))}
    if not measure(actor, 1)['multiple_target_samples']:
        raise ValueError('Validation needs actual multiple-target choices')
    trace, validation = [], []
    best, best_nll, selected_update, stale = None, float('inf'), 0, 0
    for step in range(steps):
        ids = rng.choice(train_ids, size=min(256, len(train_ids)), replace=True)
        s, a, ns = (torch.as_tensor(data[k][ids], dtype=torch.float32) for k in ('s', 'a', 'next_s'))
        labels = torch.as_tensor(data['label'][ids], dtype=torch.long)
        mask = torch.as_tensor(data['mask'][ids], dtype=torch.bool)
        chosen = torch.cat((s, a[torch.arange(len(ids)), labels]), dim=-1)
        with torch.no_grad():
            q = torch.minimum(tq1(chosen), tq2(chosen)).squeeze(-1)
        delta = q - value(s).squeeze(-1)
        vloss = (torch.where(delta > 0, .7, .3) * delta.square()).mean()
        opts[0].zero_grad(); vloss.backward(); nn.utils.clip_grad_norm_(value.parameters(), 10); opts[0].step()
        with torch.no_grad():
            target = torch.as_tensor(data['reward'][ids], dtype=torch.float32) + torch.as_tensor(data['discount'][ids], dtype=torch.float32) * value(ns).squeeze(-1)
        losses = []
        for model, optimizer in zip((q1, q2), opts[1:3]):
            loss = nn.functional.mse_loss(model(chosen).squeeze(-1), target)
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 10); optimizer.step()
            losses.append(loss.item())
        with torch.no_grad():
            advantage = q - value(s).squeeze(-1)
            weights = torch.exp(3 * advantage).clamp(max=20)
        inputs = torch.cat((s[:, None, :].expand(-1, MAX_TARGETS, -1), a), dim=-1)
        logits = actor(inputs).squeeze(-1).masked_fill(~mask, -1e9)
        aloss = (weights * nn.functional.cross_entropy(logits, labels, reduction='none')).mean()
        opts[3].zero_grad(); aloss.backward(); nn.utils.clip_grad_norm_(actor.parameters(), 10); opts[3].step()
        with torch.no_grad():
            for online, target_net in ((q1, tq1), (q2, tq2)):
                for p, tp in zip(online.parameters(), target_net.parameters()):
                    tp.mul_(.995).add_(p, alpha=.005)
        if step == 0 or (step+1) % 500 == 0:
            item = dict(update=step+1, value_loss=vloss.item(), q_loss=sum(losses)/2, actor_loss=aloss.item())
            if not all(np.isfinite(v) for v in item.values()):
                raise ValueError('Nonfinite training loss')
            trace.append(item); print(item, flush=True)
        if (step+1) % 250 == 0 or step+1 == steps:
            measured = measure(actor, 1)
            validation.append({'update': step+1, **measured})
            if measured['behavior_nll'] < best_nll - .001:
                best, best_nll, selected_update, stale = copy.deepcopy(actor.state_dict()), measured['behavior_nll'], step+1, 0
            else:
                stale += 1
            if stale >= 4 and step+1 >= 1000:
                break
    actor.load_state_dict(best)
    executed = step+1
    metrics = {name: measure(actor, split_id) for split_id, name in enumerate(('train', 'validation', 'test'))}
    support = np.zeros((4, 4), np.int64)
    np.add.at(support, (data['kind'][train_ids], data['target_kind'][train_ids]), 1)
    model = dict(algorithm='target_iql_v1', feature_version=VERSION, state_dim=STATE_DIM, target_dim=TARGET_DIM,
                 actor=[{'weight': layer.weight.detach().tolist(), 'bias': layer.bias.detach().tolist()}
                        for layer in actor if isinstance(layer, nn.Linear)], type_support=support.tolist(),
                 features_sha256=manifest['features_sha256'], dataset_sha256=manifest['dataset_sha256'],
                 updates=executed, selected_update=selected_update, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, separators=(',', ':'))+'\n', encoding='utf-8')
    report = dict(algorithm='target_iql_v1', reference='https://arxiv.org/abs/2110.06169',
                  model_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), updates=executed, max_updates=steps,
                  selected_update=selected_update, checkpoint_selection='validation behavior NLL only; patience4, interval250',
                  validation_history=validation, trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), seed=seed,
                  training_trace=trace, offline_metrics=metrics, manifest=manifest,
                  limitations=['Offline metrics are behavior agreement, not match return or win probability.',
                               'This trains target ranking only; navigation, fire eligibility and teammates follow recorded controllers.',
                               'Only a new controlled match comparison can establish score improvement.'])
    (dataset / 'training-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metrics), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build'); b.add_argument('sources', nargs='+', type=Path); b.add_argument('--out', required=True, type=Path)
    b.add_argument('--gamma', type=float, default=.997); b.add_argument('--stride', type=int, default=4)
    t = sub.add_parser('train'); t.add_argument('dataset', type=Path); t.add_argument('--model', required=True, type=Path)
    t.add_argument('--steps', type=int, default=3000); t.add_argument('--seed', type=int, default=10098)
    args = p.parse_args(); logging.disable(logging.CRITICAL)
    if args.command == 'build':
        build_dataset(args.sources, args.out, args.gamma, args.stride)
    else:
        train(args.dataset, args.model, args.steps, args.seed)


if __name__ == '__main__':
    main()
