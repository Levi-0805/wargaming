"""Observed Oct20 failure mechanisms, not synthetic match-score assertions."""
import pytest

from cssim.protocol import Action
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.ColumnAdvance import ColumnAdvance
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


def reserve_case():
    foot, cars, objects, route, form = setup()
    objects[1].raw['blueteamNum'] = 8
    reserve = next(u for u in cars if u.uid in form.reserve)
    reserve.position[:2] = objects[0].position[:2] + [20000, 0]
    s = state(foot + cars, objectives=objects, step=1)
    route.update(s)
    form.begin(s, route, recon_ready=True)
    # A straight observed-road reference isolates ordering from map corners.
    column = ColumnAdvance()
    column.path = ((objects[0].position[0] + 60000, objects[0].position[1], 0),
                   tuple(objects[0].position))
    column.progress = {reserve.uid: 40000}
    column.anchor = 30000
    move = Action.move_at(objects[0].position)
    return foot, cars, objects, route, form, reserve, column, move


def test_reserved_vehicle_wait_is_local_and_bounded_despite_stalled_rear():
    foot, cars, objects, route, form, car, column, move = reserve_case()
    for step in range(1, form.RESERVE_FOLLOW_WAIT + 2):
        s = state(foot + cars, objectives=objects, step=step)
        form.begin(s, route, recon_ready=True)
        action = form.follow_reserve(s, car, move, route, column)
        assert form.follow_reserve(s, foot[0], move, route, column) is move
        if step <= form.RESERVE_FOLLOW_WAIT:
            assert action == Action.guard_position(car.position)
        else:
            assert action is move and not form.reserve_held
    # No rearming on later contact gaps in this same objective.
    s = state(foot + cars, objectives=objects, step=80)
    form.begin(s, route)
    assert form.follow_reserve(s, car, move, route, column) is move


def test_reserve_wait_never_overrides_fire_escape_damage_or_empty_secondary():
    foot, cars, objects, route, form, car, column, move = reserve_case()
    s = state(foot + cars, objectives=objects, step=1)
    for action in (Action.attack(999), Action.move('-X+Y')):
        assert form.follow_reserve(s, car, action, route, column) is action
    car.hp -= 1
    s = state(foot + cars, objectives=objects, step=2)
    form.begin(s, route)
    assert form.follow_reserve(s, car, move, route, column) is move
    objects[1].raw['blueteamNum'] = 0
    s = state(foot + cars, objectives=objects, step=3)
    form.begin(s, route)
    assert form.follow_reserve(s, car, move, route, column) is move


def test_coordinate_recovery_bypasses_reserve_wait(monkeypatch):
    car = entity(1, 0, 0, 0, kind='BP_MNWS_Vehicle_Armored_C')
    rule = algorithm([car])
    requested = Action.move_at((10000, 0, 0))
    recovery = Action.move_at((5000, 2000, 0))
    monkeypatch.setattr(rule, '_prepare', lambda state: None)
    monkeypatch.setattr(rule, '_combat_action', lambda state, unit: requested)
    monkeypatch.setattr(rule._motion_recovery, 'apply', lambda state, unit, action: recovery)
    monkeypatch.setattr(rule._formation, 'follow_reserve',
                        lambda *args: pytest.fail('Reserve wait must not inspect recovery moves.'))
    rule._cohesion.released = True
    result = rule.choose_action(state([car]), car)
    assert result.command == 'Guard' and result.points == recovery.points


def test_defended_secondaries_reinforce_hq_one_vehicle_at_a_time():
    foot, cars, objects, route, form, _, _, _ = reserve_case()
    leads = [u for u in cars if u.uid in form.front]
    for u in cars:
        u.raw['interaction'] = ['200']
    for u in leads[:-1]:
        u.alive = False
    old_reserves = set(form.reserve)
    form.begin(state(foot + cars, objectives=objects, step=2), route)
    assert set(form.reserve) == old_reserves  # One active support still lives.
    leads[-1].alive = False
    form.begin(state(foot + cars, objectives=objects, step=3), route)
    assert len(old_reserves - set(form.reserve)) == 1
    assert len(form._active) == 1


def test_reserve_released_for_own_target_and_reset_for_new_episode():
    foot, cars, objects, route, form, car, column, move = reserve_case()
    s = state(foot + cars, objectives=objects, step=1)
    form.follow_reserve(s, car, move, route, column)
    assert form._reserve_wait
    target = form.reserve[car.uid]
    route._reorder(s, (target, *(uid for uid in route.order if uid != target)))
    s = state(foot + cars, objectives=objects, step=2)
    form.begin(s, route)
    assert form.follow_reserve(s, car, move, route, column) is move
    s = state(foot + cars, objectives=objects, episode=2)
    route.update(s)
    form.begin(s, route)
    assert not form._reserve_wait
