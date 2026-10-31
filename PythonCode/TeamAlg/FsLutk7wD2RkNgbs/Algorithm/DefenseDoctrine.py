"""Bounded defence-conditioned tactics using public observations only."""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)
PROFILE = Path(__file__).with_name('defense_profiles.json')
KINDS = ('unknown', 'hq_only', 'hq_a', 'hq_b', 'hq_a_b')
BOUNDS = {'handoff_wait': (12, 32), 'b_join_wait': (8, 24),
          'screen_gap': (2000, 4000), 'hq_hold': (12, 24),
          'a_bias': (.85, 1.15), 'b_bias': (.85, 1.15)}


def validate(data):
    if data.get('schema') != 1 or set(data.get('profiles', {})) != set(KINDS):
        raise ValueError('Invalid defence profile schema')
    for profile in data['profiles'].values():
        if set(profile) != set(BOUNDS):
            raise ValueError('Invalid defence profile fields')
        for key, (low, high) in BOUNDS.items():
            value = profile[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('Out-of-bounds defence profile: ' + key)
            if key not in ('a_bias', 'b_bias') and type(value) is not int:
                raise ValueError('Formation limits must be integers')
    return data


class DefenseDoctrine:
    def __init__(self, path=PROFILE):
        raw = Path(path).read_bytes()
        self.config = validate(json.loads(raw))
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.kind = 'unknown'
        self._frame = None
        self._candidate, self._samples = None, 0
        self.observations = {}

    def observe(self, state, route):
        frame = (state.episode, state.step)
        if frame == self._frame:
            return
        if self._frame is None or frame[0] != self._frame[0] or frame[1] <= self._frame[1]:
            self.kind, self._candidate, self._samples = 'unknown', None, 0
            self.observations = {}
        elif frame[1] != self._frame[1] + 1:
            self._candidate, self._samples = None, 0
        self._frame = frame
        strategy = route._strategy
        if not route.known_map or strategy is None or len(route._map_order) != 3:
            return
        strategy.observe(state)
        objects = [state.key_object_by_uid(uid) for uid in route._map_order]
        # Missing/invalid public counts are unknown, never evidence of an empty point.
        counts = [strategy.occupants(state, obj) if obj is not None else None for obj in objects]
        if any(value is None for value in counts):
            self._candidate, self._samples = None, 0
            return
        pressures = [strategy.pressure(state, route, obj) for obj in objects]
        if pressures[0] <= 0:
            self._candidate, self._samples = None, 0
            return  # Retain the last confirmed HQ defence type after its capture.
        has_b, has_a = pressures[1] > 0, pressures[2] > 0
        kind = 'hq_a_b' if has_a and has_b else 'hq_a' if has_a else 'hq_b' if has_b else 'hq_only'
        self._samples = self._samples + 1 if self._candidate == kind else 1
        self._candidate = kind
        if self._samples < 3:
            return
        if kind != self.kind:
            self.kind = kind
            logger.info('Defense doctrine: episode=%s step=%s kind=%s profile_sha256=%s',
                        state.episode, state.step, kind, self.sha256)
        self.observations[kind] = self.observations.get(kind, 0) + 1

    def apply(self, route, formation):
        profile = self.config['profiles'][self.kind]
        route.HQ_STABLE_FOOTHOLD_STEPS = profile['hq_hold']
        if route._strategy is not None:
            route._strategy.secondary_bias = (profile['b_bias'], profile['a_bias'])
        formation.HANDOFF_WAIT = profile['handoff_wait']
        formation.B_JOIN_WAIT = profile['b_join_wait']
        formation.SCREEN_GAP = profile['screen_gap']

    def metadata(self):
        return dict(defense_kind=self.kind, defense_profile_sha256=self.sha256,
                    defense_observations=dict(self.observations))
