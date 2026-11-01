"""Select an immutable historical tactic without reverting the runtime framework."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MethodType

from .Champion6cb5a79.RLAgentAlgorithm import ReinforceAgentAlgorithm as HistoricalAgent


def create_champion(context):
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
    original_metadata = policy.policy_metadata

    def metadata(self):
        # This is the collected profile artifact, not an applied tactical input.
        # Keep its audit hash for the existing collector; the explicit false
        # flag and historical snapshot identify the policy that actually acts.
        profile = Path(__file__).with_name('defense_profiles.json')
        return {**original_metadata(), 'policy_baseline': '6cb5a79',
                'policy_source_commit': manifest['source_commit'],
                'policy_snapshot_sha256': hashlib.sha256((folder / 'snapshot.json').read_bytes()).hexdigest(),
                'defense_profile_sha256': hashlib.sha256(profile.read_bytes()).hexdigest(),
                'defense_profiles_applied': False}

    policy.policy_metadata = MethodType(metadata, policy)
    return policy
