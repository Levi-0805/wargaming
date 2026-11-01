"""Regression: tied full progress does not release an established garrison."""
from dataclasses import replace
import pytest
from test_adaptive_assault import opening, advance
from test_fslutk_rule import guards_at, state

def established():
    _, objects, route = opening()
    units = guards_at(objects[0], count=8)
    objects[0].raw.update(percent=1, redTeamNum=8, blueteamNum=0)
    advance(route, units, objects, range(4))
    assert route.stage == 1 and len(route._garrison) == 4
    objects[0].raw.update(percent=1, redTeamNum=4, blueteamNum=4)
    return units, objects, route

def test_tied_full_bar_preserves_assigned_guards_without_proving_ownership():
    units, objects, route = established()
    original = dict(route._garrison)
    current = state(units, objectives=objects, step=4)
    route._assign_garrison(current)
    assert route._garrison == original
    assert not route.owned(current, objects[0])
    assert route.stage == 1

@pytest.mark.parametrize('case', ['below_full', 'explicit_enemy_owner', 'no_previous', 'dead_guard', 'distant_guard'])
def test_tie_retention_is_limited_to_existing_eligible_local_guards(case):
    units, objects, route = established()
    chosen = next(u for u in units if u.uid in route._garrison)
    if case == 'below_full': objects[0].raw['percent'] = .9
    if case == 'explicit_enemy_owner': objects[0] = replace(objects[0], team=1)
    if case == 'no_previous': route._garrison.clear()
    if case == 'dead_guard': units = [replace(u, alive=False) if u.uid == chosen.uid else u for u in units]
    if case == 'distant_guard':
        chosen.position[:] = (objects[0].position[0] + 30000, objects[0].position[1], objects[0].position[2])
    route._assign_garrison(state(units, objectives=objects, step=4))
    assert len(route._garrison) == (3 if case in {'dead_guard', 'distant_guard'} else 0)

def test_tied_current_point_does_not_create_a_new_garrison_or_advance():
    _, objects, route = opening()
    route.hq_first = True
    units = guards_at(objects[0], count=8)
    objects[0].raw.update(percent=1, redTeamNum=8, blueteamNum=8)
    advance(route, units, objects, range(30))
    assert route.stage == 0
    assert not route._garrison
