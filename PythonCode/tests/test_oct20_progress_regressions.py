"""Observed Oct20 failure mechanisms, not synthetic match-score assertions."""
import pytest

from cssim.protocol import Action
from test_assault_formation import setup
from test_fslutk_rule import algorithm, entity, guards_at, map_objectives, objective, state


@pytest.mark.parametrize('kind', ['BP_BaseSoldier_C', 'BP_RoboDog_C'])
def test_blind_transit_scan_cannot_prevent_obstruction_recovery(kind):
    unit = entity(1, 0, 0, 0, kind=kind)
    objects = [objective(901, 10000, name='HQ')]
    rule = algorithm([unit])
    actions = [rule.act(state([unit], objectives=objects, step=step))[0] for step in range(14)]
    # Old two-in-six Guard scans never allowed eight movement frames in a row.
    assert all(a.command == 'Moving' for a in actions[:8])
    assert any(a.command == 'Moving' and a.points is None and a.direction for a in actions[8:])
    assert any(a.points is not None for a in actions[10:])


def test_contested_b_full_bar_cannot_release_main_body_to_a():
    objects = map_objectives()
    hq, base, _ = objects
    hq.raw.update(percent=1, redTeamNum=4, blueteamNum=0)
    base.raw.update(percent=1, redTeamNum=23, blueteamNum=10)
    units = guards_at(hq, first=1, count=4) + guards_at(base, first=10, count=23)
    rule = algorithm(units)
    for step in range(4):
        rule.act(state(units, objectives=objects, step=step))
    assert rule._route.stage == 1
    for step in range(4, 20):
        s = state(units, objectives=objects, step=step)
        rule.act(s)
        assert rule._route.objective(s).uid == base.uid
    base.raw['blueteamNum'] = 0
    for step in range(20, 24):
        s = state(units, objectives=objects, step=step)
        rule.act(s)
    assert rule._route.objective(s).uid == objects[2].uid



def test_defended_secondaries_still_replace_two_hq_support_vehicles():
    foot, cars, objects, route, form = setup()
    objects[1].raw['blueteamNum'] = 8
    leads = [u for u in cars if u.uid in form.front]
    for u in cars:
        u.raw['interaction'] = ['200']
    for u in leads[:-1]:
        u.alive = False
    old_reserves = set(form.reserve)
    form.begin(state(foot + cars, objectives=objects, step=2), route)
    assert len(old_reserves - set(form.reserve)) == 1
    assert len(form._active) == 2


def test_coordinate_recovery_survives_after_reserve_wait_retirement(monkeypatch):
    car = entity(1, 0, 0, 0, kind='BP_MNWS_Vehicle_Armored_C')
    rule = algorithm([car])
    requested = Action.move_at((10000, 0, 0))
    recovery = Action.move_at((5000, 2000, 0))
    monkeypatch.setattr(rule, '_prepare', lambda state: None)
    monkeypatch.setattr(rule, '_combat_action', lambda state, unit: requested)
    monkeypatch.setattr(rule._motion_recovery, 'apply', lambda state, unit, action: recovery)
    rule._cohesion.released = True
    result = rule.choose_action(state([car]), car)
    assert result.command == 'Guard' and result.points == recovery.points
