from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.HQExit import HQExit
from test_remnant_capture import battle, reduce, step
from test_fslutk_rule import entity, state
from test_rear_defense_rule import captured


def test_armed_car_stranded_at_spawn_does_not_block_last_pair():
    units, objects, rule = battle()
    reduce(units, 2)
    car = entity(999,0,25,63700,32000,kind='BP_MNWS_Vehicle_Armored_C')
    car.raw.update(interaction=['200'],rSVD1='car;1')
    units.append(car)
    s,_=step(rule,units,objects)
    assert rule._route._remnant.active
    assert rule._route.objective(s).uid==objects[2].uid


def test_empty_a_footprint_is_not_proof_of_capture_arrival():
    units, objects, rule = battle()
    reduce(units,2)
    step(rule,units,objects)
    a=objects[2]
    a.raw.update(scale={'x':10500,'y':9000},rotation={'yaw':-12.8125})
    for u in units[:2]:u.position[:]=(-2245,-19195,5634)
    s,actions=step(rule,units,objects,9)
    assert rule._route.inside_garrison(a,units[0].position)
    assert all(x.command=='Moving' for x in actions[:2])
    assert all(x.points==tuple(a.position) for x in actions[:2])


def test_b_exit_tolerance_cannot_trap_unit_nine_metres_from_exit_forever():
    units,objects,rule=battle();reduce(units,2)
    units[0].position[:]=(-31267,-27808,5655)
    step(rule,units,objects)
    for i in range(9,35):s,actions=step(rule,units,objects,i)
    assert rule._route._remnant._cursor[units[0].uid]>0


def test_stalled_courtyard_exit_uses_reverse_front_corridor():
    foot,_,objects,rule=captured(count=8,point=(-42795,28221,6429))
    unit=foot[-1];exit=HQExit()
    for i in range(4,13):
        s=state(foot,objectives=objects,step=i)
        action=exit.choose(s,unit,rule._route,rule._route.HQ_B[0])
    assert action is not None and action.points in HQExit.PATH
    assert action.points!=rule._route.HQ_B[0]


def test_successful_departure_and_guard_are_not_rerouted():
    foot,_,objects,rule=captured(count=8,point=(-42795,28221,6429))
    exit=HQExit();unit=foot[-1];guard=foot[0]
    for i in range(4,20):
        unit.position[0]+=300
        s=state(foot,objectives=objects,step=i)
        assert exit.choose(s,unit,rule._route,rule._route.HQ_B[0]) is None
        assert exit.choose(s,guard,rule._route,rule._route.HQ_B[0]) is None


def test_combat_gap_does_not_count_as_stalled_departure():
    foot,_,objects,rule=captured(count=8,point=(-42795,28221,6429))
    exit=HQExit()
    for i in range(4,80,4):
        assert exit.choose(state(foot,objectives=objects,step=i),foot[-1],rule._route,rule._route.HQ_B[0]) is None


def test_mobile_b_assault_respects_failed_target_cooldown():
    units,objects,rule=battle()
    u=units[-1];u.position[:]=objects[1].position
    enemy=entity(999,1,0,u.position[0]+1000,u.position[1])
    enemy.position[2]=u.position[2]+1500
    u.raw['agentPerception']=[999]
    rule._hit_watch[(u.uid,999)]={'retry':20}
    s=state(units,[enemy],objects,step=8)
    action=rule._engage(s,u,[enemy],{999})
    assert action.command=='Moving' and action.target_uid is None


def test_fresh_target_remains_attackable_when_another_target_cools():
    units,objects,rule=battle();u=units[-1]
    enemies=[entity(i,1,0,u.position[0]+1000,u.position[1]) for i in (998,999)]
    u.raw['agentPerception']=[998,999]
    rule._hit_watch[(u.uid,999)]={'retry':20}
    action=rule._engage(state(units,enemies,objects,step=8),u,enemies,{998,999})
    assert action.command=='NormalAttacking' and action.target_uid==998
