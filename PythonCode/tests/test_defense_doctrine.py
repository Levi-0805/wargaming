import copy
from dataclasses import replace

import pytest

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.DefenseDoctrine import DefenseDoctrine, validate
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultFormation import AssaultFormation
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from test_fslutk_rule import entity, map_objectives, state, algorithm


@pytest.mark.parametrize('a,b,expected', [(0,0,'hq_only'), (4,0,'hq_a'), (0,4,'hq_b'), (4,4,'hq_a_b')])
def test_public_defence_confirmation_keeps_hq_first(a,b,expected):
    objects = map_objectives()
    for obj, count in zip(objects, (10,b,a)):
        obj.raw['blueteamNum'] = count
    route, doctrine, formation = AssaultRoute(adaptive=True), DefenseDoctrine(), AssaultFormation()
    units = [entity(i,0,i,60000,12000) for i in range(8)]
    for step in range(3):
        s = state(units, objectives=objects, step=step)
        route.update(s); doctrine.observe(s,route); doctrine.apply(route,formation)
        assert route.objective(s).uid == route.hq_uid
        if step < 2:
            assert doctrine.kind == 'unknown'
    assert doctrine.kind == expected
    assert 8 <= formation.B_JOIN_WAIT <= 24
    assert 12 <= formation.HANDOFF_WAIT <= 32
    assert route.HQ_FRONT[0] != (-40107,36086)


def test_unknown_public_data_and_hidden_truth_do_not_prove_empty():
    objects = map_objectives()
    objects[0].raw['blueteamNum'] = 12
    objects[2].raw.pop('blueteamNum')
    unit = entity(1,0,0,60000,12000)
    rule = algorithm([unit])
    for step in range(10):
        s = replace(state([unit],objectives=objects,step=step), raw={'hidden_enemy_positions':[1,2,3]})
        rule.act(s)
    assert rule._doctrine.kind == 'unknown'
    assert rule._route.objective(s).uid == rule._route.hq_uid
    assert rule._formation.HANDOFF_WAIT == 28
    assert rule.training_metrics()['defense_kind_code'] == 0


def test_episode_reset_drops_previous_defence_type_and_observation_gaps_restart_confirmation():
    objects=map_objectives(); objects[0].raw['blueteamNum']=9
    route, doctrine=AssaultRoute(adaptive=True),DefenseDoctrine()
    units=[entity(1,0,0,60000,12000)]
    for step in (0,1,8,9):
        s=state(units,objectives=objects,step=step)
        route.update(s);doctrine.observe(s,route)
    assert doctrine.kind=='unknown'
    s=state(units,objectives=objects,step=10);route.update(s);doctrine.observe(s,route)
    assert doctrine.kind=='hq_only'
    s=state(units,objectives=objects,step=0,episode=2);route.update(s);doctrine.observe(s,route)
    assert doctrine.kind=='unknown'


@pytest.mark.parametrize('value', [True, 0, 1000, float('nan'), float('inf'), '18'])
def test_unsafe_unbounded_or_non_numeric_waits_rejected(value):
    data=copy.deepcopy(DefenseDoctrine().config)
    data['profiles']['hq_b']['b_join_wait']=value
    with pytest.raises(ValueError): validate(data)
