"""IQL-trained target ranking, with the existing controller owning eligibility.

This is a learned sub-policy, not end-to-end navigation or online training.
Models are exported as numeric JSON, never executable/pickled checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import numpy as np

from .TargetFeatures import VERSION, STATE_DIM, TARGET_DIM, KINDS, state_features, target_features
from .RuleAgentAlgorithm import ReinforceAgentAlgorithm as RuleAgent

logger = logging.getLogger(__name__)


class TargetPolicy:
    def __init__(self, path, *, shadow=True):
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        if (data.get('feature_version') != VERSION or data.get('state_dim') != STATE_DIM
                or data.get('target_dim') != TARGET_DIM or data.get('algorithm') != 'target_iql_v1'):
            raise ValueError('Incompatible target IQL model schema')
        expected = hashlib.sha256(Path(__file__).with_name('TargetFeatures.py').read_bytes()).hexdigest()
        if data.get('features_sha256') != expected:
            raise ValueError('Target IQL feature source mismatch')
        self.weights = []
        width = STATE_DIM + TARGET_DIM
        for layer in data['actor']:
            w, b = np.asarray(layer['weight'], dtype=np.float32), np.asarray(layer['bias'], dtype=np.float32)
            if (w.ndim != 2 or w.shape[1] != width or b.shape != (w.shape[0],)
                    or not np.isfinite(w).all() or not np.isfinite(b).all()):
                raise ValueError('Invalid target IQL layer')
            self.weights.append((w, b))
            width = w.shape[0]
        if len(self.weights) != 3 or width != 1:
            raise ValueError('Invalid target IQL architecture')
        self.support = np.asarray(data['type_support'], dtype=np.int64)
        if self.support.shape != (4, 4) or (self.support < 0).any():
            raise ValueError('Invalid training support counts')
        self.shadow = shadow
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.decisions = self.proposed_changes = self.applied_changes = 0
        logger.info('Target IQL loaded: model_sha256=%s shadow=%s', self.sha256, shadow)

    def scores(self, inputs):
        x = np.asarray(inputs, dtype=np.float32)
        for i, (w, b) in enumerate(self.weights):
            x = x @ w.T + b
            if i < len(self.weights)-1:
                x = np.maximum(x, 0)
        return x[:, 0]

    def choose(self, state, agent, candidates, default):
        # Callers have already removed cooling/ineligible targets. Never add a
        # target or replace an air/self-defence choice with a different task.
        if agent.entity_type not in KINDS or default.entity_type not in KINDS:
            return default
        local = agent.perceived_uids()
        pool = [e for e in candidates if e.entity_type in KINDS and e.uid in local
                and e.alive and e.hp > 0 and e.team != agent.team]
        if len(pool) < 2 or default.uid not in {e.uid for e in pool}:
            return default
        kind = KINDS.index(agent.entity_type)
        if any(self.support[kind, KINDS.index(e.entity_type)] < 50 for e in pool):
            return default
        features = state_features(state, agent)
        values = self.scores([np.concatenate((features, target_features(state, agent, e))) for e in pool])
        if not np.isfinite(values).all():
            return default
        probs = np.exp(values - values.max()); probs /= probs.sum()
        index = int(np.argmax(probs))
        old_index = next(i for i, e in enumerate(pool) if e.uid == default.uid)
        self.decisions += 1
        # This is a conservative behavior-policy confidence threshold, not a
        # calibrated probability of winning or a claim of higher match score.
        if probs[index] < .35 or probs[index] - probs[old_index] < .10:
            return default
        self.proposed_changes += 1
        if self.shadow:
            return default
        self.applied_changes += 1
        return pool[index]


class HybridIQLAgent(RuleAgent):
    mode = 'hybrid_iql'

    def __init__(self, context, config):
        super().__init__(context, config.enable_parent_assignment,
                         enable_preservation_pause=config.enable_preservation_pause,
                         enable_shared_contact_maneuver=config.enable_shared_contact_maneuver,
                         enable_forced_entry_under_contact=config.enable_forced_entry_under_contact,
                         enable_continuous_column=config.enable_continuous_column,
                         enable_entry_waves=config.enable_entry_waves)
        path = Path(config.iql_model)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        self._learned = TargetPolicy(path, shadow=config.iql_shadow)
        self._rl_state = None
        self._vehicle_fire.target_selector = self._learned.choose

    def _prepare(self, state):
        self._rl_state = state
        return super()._prepare(state)

    def _attack_target(self, agent, enemies, local=None):
        default = super()._attack_target(agent, enemies, local)
        if self._rl_state is None:
            return default
        return self._learned.choose(self._rl_state, agent, enemies, default)

    def training_metrics(self):
        return {**super().training_metrics(), 'iql_decisions': self._learned.decisions,
                'iql_proposed_changes': self._learned.proposed_changes,
                'iql_applied_changes': self._learned.applied_changes,
                'iql_shadow': int(self._learned.shadow), 'iql_online_training': 0}

    def policy_metadata(self):
        return {**super().policy_metadata(), 'model_sha256': self._learned.sha256,
                'feature_version': VERSION, 'shadow': self._learned.shadow,
                'scope': 'ground_target_selection', 'online_training': False}
