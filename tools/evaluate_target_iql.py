"""Check IQL integration on held-out public observations, not simulated returns."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'PythonCode'))
from cssim.algorithms import AlgorithmContext
from cssim.environment.env import CssimEnvironment
from cssim.protocol import ActionSetFactory
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RuleAgentAlgorithm import ReinforceAgentAlgorithm
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.LearnedTargeting import HybridIQLAgent
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.RLAgentAlgorithm import AlgorithmConfig
from train_target_iql import config


def evaluate(source, model, episodes):
    envs = {name: CssimEnvironment(config()) for name in ('rule', 'shadow', 'iql')}
    previous, episode, rules, rows, current = None, 0, {}, [], None
    limit, consumed, digest = source.stat().st_size, 0, hashlib.sha256()
    with source.open('rb') as stream:
        while consumed < limit:
            line = stream.readline(limit-consumed)
            if not line:
                break
            digest.update(line); consumed += len(line)
            row = json.loads(line); p = row.get('payload', {})
            if row.get('direction') != 'ue_to_python' or not isinstance(p, dict):
                continue
            if p.get('propertyArr'):
                for env in envs.values():
                    env._build_roster(p)
            if not p.get('dataArr'):
                continue
            step = p['dataGlobal']['timeCnt']
            if previous is None or step < previous:
                episode += 1; rules = {}; current = None
                for env in envs.values():
                    env.episode = episode
                if episode in episodes:
                    current = dict(episode=episode, frames=0, changed_commands=0, shadow_mismatches=0,
                                   changed_air_commands=0, invalid_vehicle_fire=0, pre_hq_diversions=0,
                                   objective_changes=0)
                    rows.append(current); print(f'evaluate episode {episode}', flush=True)
            previous = step
            if current is None:
                continue
            wires, objectives, metrics, specials = {}, {}, {}, {}
            for name, env in envs.items():
                snap = env._parse_state(p); state = env.build_team_state(snap, 0)
                if name not in rules:
                    context = AlgorithmContext(0, state.agents, state.visible_opponents,
                                               ActionSetFactory.standard(), 'cpu', {}, state.commanders)
                    rules[name] = (ReinforceAgentAlgorithm(context) if name == 'rule' else
                                   HybridIQLAgent(context, AlgorithmConfig(mode='hybrid_iql', iql_model=str(model),
                                                                          iql_shadow=name == 'shadow')))
                rule = rules[name]; actions = rule.act(state)
                wires[name] = [a.to_ue_string(u.uid, snap.entities) for u, a in zip(state.agents, actions)]
                specials[name] = [c.parameters() for c in rule.take_selected_special_commands()]
                objective = rule._route.objective(state)
                objectives[name] = objective.uid if objective else None
                metrics[name] = rule.training_metrics()
                if name == 'iql':
                    current['pre_hq_diversions'] += bool(objective and not rule._route.hq_secured and not rule._route.is_hq(objective))
                    for u, a in zip(state.agents, actions):
                        if u.entity_type in rule.VEHICLES and a.command == 'NormalAttacking':
                            current['invalid_vehicle_fire'] += a.target_uid not in {e.uid for e in rule._vehicle_fire.eligible(state, u)}
            current['frames'] += 1
            current['shadow_mismatches'] += sum(a != b for a, b in zip(wires['rule'], wires['shadow'])) + (specials['rule'] != specials['shadow'])
            current['changed_commands'] += sum(a != b for a, b in zip(wires['rule'], wires['iql']))
            current['changed_air_commands'] += sum(a != b and u.entity_type in {'BP_Base_UAV_C', 'BP_Helicopter_C'}
                                                   for u, a, b in zip(state.agents, wires['rule'], wires['iql']))
            current['objective_changes'] += objectives['rule'] != objectives['iql']
            current['metrics'] = metrics['iql']
    return dict(source=str(source), source_sha256=digest.hexdigest(), source_bytes=consumed,
                model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(), episodes=rows,
                limitation='Fixed public-state replay; changes and offline metrics are not new match scores.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path); p.add_argument('--model', type=Path, required=True)
    p.add_argument('--episodes', nargs='+', type=int, default=[1]); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); logging.disable(logging.CRITICAL)
    result = evaluate(args.source, args.model.resolve(), set(args.episodes))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    failures = sum(r[k] for r in result['episodes'] for k in
                   ('shadow_mismatches', 'invalid_vehicle_fire', 'pre_hq_diversions', 'objective_changes', 'changed_air_commands'))
    print(f'episodes={len(result["episodes"])} failures={failures}')
    return int(bool(failures) or not result['episodes'])


if __name__ == '__main__':
    raise SystemExit(main())
