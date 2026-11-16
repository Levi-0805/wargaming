"""Select an immutable historical tactic without reverting the runtime framework."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MethodType

from .Champion6cb5a79.RLAgentAlgorithm import ReinforceAgentAlgorithm as HistoricalAgent


def create_champion(context):
    selector = context.config.get('policy_baseline', '6cb5a79')
    folder = Path(__file__).with_name('Champion6cb5a79')
    manifest = json.loads((folder / 'snapshot.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Historical policy snapshot changed: ' + name)
    options = dict(context.config)
    options.pop('policy_baseline', None)
    model = Path(options.get('iql_model', 'Models/target_iql_v1.json'))
    if not model.is_absolute():
        model = Path(__file__).resolve().parents[1] / model
    options['iql_model'] = str(model.resolve())
    policy = HistoricalAgent(replace(context, config=options))
    overlay = None
    if selector in ('6cb5a79_hq_hold', '6cb5a79_hq_guard_entry'):
        from .ChampionGuardHold import TiedHQGuardRoute, HQGuardEntryRoute
        route_class = HQGuardEntryRoute if selector == '6cb5a79_hq_guard_entry' else TiedHQGuardRoute
        policy._route = route_class(adaptive=policy._route.adaptive, hq_first=policy._route.hq_first)
        overlay = Path(__file__).with_name('ChampionGuardHold.py')
        if selector == '6cb5a79_hq_guard_entry':
            original_defend = policy._defend

            def defend(self, state, agent, obj, contacts, local):
                preservation = self._route.preserve_guard(state, agent, obj, contacts, local)
                if preservation is not None:
                    return preservation
                return original_defend(state, agent, obj, contacts, local)

            policy._defend = MethodType(defend, policy)
            original_reward = policy.calculate_reward

            def calculate_reward(self, previous, current, events):
                self._route.observe_guard_casualties(previous, current, events)
                return original_reward(previous, current, events)

            policy.calculate_reward = MethodType(calculate_reward, policy)
            original_attack_target = policy._attack_target

            def attack_target(self, agent, enemies, local=None):
                # Keep the historical contact and weapon gates. Shared-only
                # contacts and air units keep their existing target selection.
                if agent.entity_type in {self._SOLDIER, self._DOG} and local:
                    nearby = self._route.infantry_finish_pool(
                        unit for unit in enemies if unit.uid in local)
                    if nearby:
                        return original_attack_target(agent, nearby, local)
                return original_attack_target(agent, enemies, local)

            policy._attack_target = MethodType(attack_target, policy)
            if policy._vehicle_fire.target_selector is not None:
                original_vehicle_selector = policy._vehicle_fire.target_selector

                def vehicle_target(state, agent, available, default):
                    # An eligible hostile vehicle still takes precedence.
                    armor = tuple(unit for unit in available if unit.entity_type in policy.VEHICLES)
                    pool = armor or policy._route.infantry_finish_pool(available)
                    if not pool:
                        return original_vehicle_selector(state, agent, available, default)
                    if not any(unit is default for unit in pool):
                        default = min(pool, key=lambda unit: (
                            float(unit.hp), policy._vehicle_fire.distance(agent, unit), unit.uid))
                    return original_vehicle_selector(state, agent, pool, default)

                policy._vehicle_fire.target_selector = vehicle_target
    original_metadata = policy.policy_metadata

    def metadata(self):
        # This is the collected profile artifact, not an applied tactical input.
        # Keep its audit hash for the existing collector; the explicit false
        # flag and historical snapshot identify the policy that actually acts.
        profile = Path(__file__).with_name('defense_profiles.json')
        return {**original_metadata(), 'policy_baseline': selector,
                'policy_source_commit': manifest['source_commit'],
                'policy_snapshot_sha256': hashlib.sha256((folder / 'snapshot.json').read_bytes()).hexdigest(),
                'defense_profile_sha256': hashlib.sha256(profile.read_bytes()).hexdigest(),
                'defense_profiles_applied': False,
                **({'policy_overlay': ('hq_tied_guard_retention_and_entry_margin'
                                      if selector == '6cb5a79_hq_guard_entry' else 'hq_tied_guard_retention'),
                    'policy_overlay_sha256': hashlib.sha256(overlay.read_bytes()).hexdigest()} if overlay else {})}

    policy.policy_metadata = MethodType(metadata, policy)
    return policy
