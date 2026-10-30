"""Versioned public features for the ground target-selection sub-policy.

UIDs, hidden enemies, final scores and historical policy versions are never
features. Training labels must be actual sent attacks with a locally seen target.
"""
from __future__ import annotations

import math
import numpy as np

VERSION = 1
KINDS = ('BP_BaseSoldier_C', 'BP_RoboDog_C',
         'BP_MNWS_Vehicle_Armored_C', 'BP_MNWS_Vehicle_6x6UGV_C')
ANCHORS = ((-41810, 32275), (-45450, -30945), (-3000, -27530))
STATE_DIM = 47
TARGET_DIM = 11


def clean(values):
    return np.clip(np.nan_to_num(np.asarray(values, dtype=np.float32),
                                nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)


def ammo(agent):
    try:
        values = [float(v) for v in agent.raw.get('interaction', ())]
        if values and all(math.isfinite(v) and v >= 0 for v in values):
            return sum(values)
    except (TypeError, ValueError):
        pass
    return None


def local_targets(state, agent):
    ids = agent.perceived_uids()
    return [e for e in state.perceived_opponents(agent)
            if e.uid in ids and e.alive and e.hp > 0 and e.team != agent.team and e.entity_type in KINDS]


def state_features(state, agent):
    nearby = [u for u in state.agents if u.alive and u.uid != agent.uid
              and math.dist(u.position[:2], agent.position[:2]) <= 8000]
    enemies = local_targets(state, agent)
    rounds = ammo(agent)
    values = [float(agent.entity_type == k) for k in KINDS]
    values += [agent.hp / 100, 0 if rounds is None else rounds / 1000,
               float(rounds is not None), float(agent.communication_ok() is not False), state.step / 500]
    values += [agent.position[0] / 100000, agent.position[1] / 100000, agent.position[2] / 20000]
    for population in ([u for u in state.agents if u.alive], nearby, enemies):
        values += [sum(u.entity_type == k for u in population) / 40 for k in KINDS]
    values += [min((math.dist(e.position, agent.position) / 30000 for e in enemies), default=1)]
    objects = [o for o in state.key_objects if o.valid]
    for anchor in ANCHORS:
        obj = min(objects, key=lambda o: math.dist(o.position[:2], anchor), default=None)
        if obj is None or math.dist(obj.position[:2], anchor) > 5000:
            values += [0, 0, 0, -1, -1, 0, 0]
            continue
        values += [(obj.position[0]-agent.position[0]) / 100000,
                   (obj.position[1]-agent.position[1]) / 100000,
                   (obj.position[2]-agent.position[2]) / 20000]
        for key in ('blueteamNum' if state.team == 0 else 'redTeamNum',
                    'redTeamNum' if state.team == 0 else 'blueteamNum'):
            try:
                value = float(obj.raw[key]) / 40
            except (KeyError, TypeError, ValueError):
                value = -1
            values.append(value)
        values += [float(obj.raw.get('percent', 0) or 0), 1]
    values += [float(agent.alive)]
    assert len(values) == STATE_DIM
    return clean(values)


def target_features(state, agent, target):
    values = [float(target.entity_type == k) for k in KINDS]
    values += [target.hp / 100]
    values += [(target.position[i]-agent.position[i]) / (30000 if i < 2 else 10000) for i in range(3)]
    values += [math.dist(agent.position, target.position) / 30000,
               sum(target.uid in u.perceived_uids() for u in state.agents if u.alive) / 40,
               float(target.uid in agent.perceived_uids())]
    assert len(values) == TARGET_DIM
    return clean(values)
