"""Reproducible three-calendar-day tactical search; returns need live games."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone, date
import math
import random
import statistics

CHINA = timezone(timedelta(hours=8))
FIELDS = {'handoff_wait': (12, 32, 4), 'b_join_wait': (8, 24, 2),
          'screen_gap': (2000, 4000, 500), 'hq_hold': (12, 24, 4),
          'a_bias': (.85, 1.15, .05), 'b_bias': (.85, 1.15, .05)}


def metrics(scores):
    if not scores or any(type(s) not in (int, float) or not math.isfinite(s) for s in scores):
        raise ValueError('Scores must be finite observed terminal totals')
    ordered = sorted(scores)
    tail = statistics.mean(ordered[:max(1, math.ceil(len(scores)*.2))])
    mean, spread = statistics.mean(scores), statistics.pstdev(scores)
    low = sum(s < 5000 for s in scores)
    return dict(count=len(scores), mean=mean, median=statistics.median(scores), minimum=min(scores),
                maximum=max(scores), stddev=spread, bottom_20_mean=tail, under_5000=low,
                under_4000=sum(s < 4000 for s in scores),
                energy=5000-mean + 1000*low/len(scores) + .25*spread + .5*max(0, 5000-tail),
                stability_target_met_in_sample=low == 0 and len(scores) >= 10)


def temperature(anchor, now=None):
    today = (now or datetime.now(CHINA)).astimezone(CHINA).date()
    elapsed = max(0, (today - date.fromisoformat(anchor)).days)
    window, day = divmod(elapsed, 3)
    # Calendar blocks, not rolling 72-hour timers. Reheat every third midnight.
    return dict(window=window, day=day, temperature=(800., 300., 100.)[day],
                starts_on=str(date.fromisoformat(anchor)+timedelta(days=window*3)),
                ends_on=str(date.fromisoformat(anchor)+timedelta(days=window*3+2)))


def assess(search, observation, *, round_number, anchor, now=None):
    """Accept/reject a previously proposed version on actual ten-game returns."""
    if observation['metrics']['count'] != 10:
        raise ValueError('Search evaluations require exactly ten cumulative games')
    result = deepcopy(search)
    clock = temperature(anchor, now)
    rng = random.Random(10098 + round_number + 100000*clock['window'])
    incumbent = result.get('current')
    tested_kind = observation.get('tested_kind')
    exposed = observation.get('defense_groups', {}).get(tested_kind, {}).get('count', 0)
    if incumbent is not None and tested_kind and exposed < 3:
        result['last_acceptance'] = dict(**clock, accepted=False,
            reason='insufficient_observed_defence_exposure', tested_kind=tested_kind,
            exposed_games=exposed, required_games=3,
            limitation='An unobserved or rare tactical branch cannot be credited for this round score.')
        return result
    delta = observation['metrics']['energy'] - incumbent['metrics']['energy'] if incumbent else 0.
    probability = math.exp(-max(0., delta)/clock['temperature'])
    draw = rng.random()
    # Never accept a catastrophic lower tail merely because the search is hot.
    tail_ok = not incumbent or (observation['metrics']['under_4000'] <= incumbent['metrics']['under_4000']+1
        and observation['metrics']['bottom_20_mean'] >= incumbent['metrics']['bottom_20_mean']-500)
    accepted = incumbent is None or tail_ok and (delta <= 0 or draw < probability)
    if accepted:
        result['current'] = observation
    champion = result.get('champion')
    if champion is None or observation['metrics']['energy'] < champion['metrics']['energy']:
        result['champion'] = observation
    result['last_acceptance'] = dict(**clock, accepted=accepted, delta=delta,
        probability=probability, draw=draw, tail_guard_passed=tail_ok,
        limitation='Unpaired ten-game estimates; neither causality nor out-of-sample stability is established.')
    return result


def propose(profiles, *, round_number, anchor, now=None, eligible=None):
    """Change one bounded parameter in one known defence class; unknown stays safe."""
    result = deepcopy(profiles)
    clock = temperature(anchor, now)
    rng = random.Random(731 + round_number + clock['window']*100000)
    kinds = tuple(k for k in ('hq_only', 'hq_a', 'hq_b', 'hq_a_b') if eligible is None or k in eligible)
    if not kinds:
        kinds = ('hq_only', 'hq_a', 'hq_b', 'hq_a_b')
    kind = kinds[(round_number-1) % len(kinds)]
    key = rng.choice(sorted(FIELDS))
    low, high, step = FIELDS[key]
    old = result['profiles'][kind][key]
    options = [round(old + sign*step, 4) for sign in (-1, 1) if low <= old+sign*step <= high]
    new = rng.choice(options)
    if key not in ('a_bias', 'b_bias'):
        new = int(new)
    result['profiles'][kind][key] = new
    return result, dict(kind=kind, field=key, before=old, after=new, **clock,
        status='unvalidated_proposal', required='Publish and collect a new ten-game version before evaluating scores.')
