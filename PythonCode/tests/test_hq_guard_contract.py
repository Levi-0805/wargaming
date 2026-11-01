"""HQ departure must leave independent soldiers when residual defenders remain."""
from dataclasses import replace
from test_adaptive_assault import opening, advance
from test_fslutk_rule import guards_at

def mixed_units(soldiers):
    _, objects, route = opening()
    route.hq_first = True
    units = guards_at(objects[0], count=12)
    units = [replace(u, entity_type='BP_RoboDog_C') if i < 12-soldiers else u
             for i, u in enumerate(units)]
    return units, objects, route

def test_residual_defender_departure_keeps_two_independent_soldiers():
    units, objects, route = mixed_units(8)
    objects[0].raw.update(percent=1, redTeamNum=12, blueteamNum=1)
    advance(route, units, objects, range(30))
    guards = [u for u in units if u.uid in route._garrison]
    assert route.stage == 1
    assert len(guards) >= 4
    assert all(u.entity_type == 'BP_BaseSoldier_C' for u in guards)

def test_one_soldier_and_dogs_do_not_prove_residual_hq_secure():
    units, objects, route = mixed_units(1)
    objects[0].raw.update(percent=1, redTeamNum=12, blueteamNum=1)
    advance(route, units, objects, range(50))
    assert route.stage == 0
    assert not route.hq_secured

def test_clear_hq_can_release_even_when_only_one_soldier_survives():
    units, objects, route = mixed_units(1)
    objects[0].raw.update(percent=1, redTeamNum=12, blueteamNum=0)
    advance(route, units, objects, range(10))
    assert route.stage == 1
