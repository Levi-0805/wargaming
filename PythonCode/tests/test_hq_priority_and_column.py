"""HQ ordering and forward-progress contracts; not a battle-score simulator."""
from types import SimpleNamespace

from cssim.protocol import Action
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.ColumnAdvance import ColumnAdvance
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.EntryWaves import EntryWaves
from test_fslutk_rule import entity, guards_at, map_objectives, objective, state, algorithm


def test_production_never_diverts_to_empty_ab_before_hq_confirmation():
    objects = map_objectives()
    units = guards_at(objects[1], count=8)
    objects[0].raw['blueteamNum'] = 30
    objects[1].raw.update(percent=1, redTeamNum=8)
    rule = algorithm(units)
    for step in range(80):
        s = state(units, objectives=objects, step=step)
        rule._prepare(s)
        assert rule._route.objective(s).uid == objects[0].uid
        assert not rule._route.hq_secured and rule._route.stage == 0


def test_full_bar_with_enemy_or_absent_guards_cannot_unlock_ab():
    objects = map_objectives()
    units = guards_at(objects[0], count=6)
    route = AssaultRoute(adaptive=True)
    objects[0].raw.update(percent=1, redTeamNum=6, blueteamNum=2)
    for step in range(10):
        route.update(state(units, objectives=objects, step=step))
        assert route.stage == 0 and not route.hq_secured
    objects[0].raw['blueteamNum'] = 0
    for u in units:
        u.position[0] += 30000
    for step in range(10, 20):
        route.update(state(units, objectives=objects, step=step))
        assert route.stage == 0 and not route.hq_secured
    for u in units:
        u.position[0] -= 30000
    for step in range(20, 24):
        route.update(state(units, objectives=objects, step=step))
    assert route.stage == 1 and route.hq_secured


def test_lost_hq_rearms_priority_and_episode_reset_clears_confirmation():
    objects = map_objectives()
    units = guards_at(objects[0], count=8)
    objects[0].raw.update(percent=1, redTeamNum=8)
    route = AssaultRoute(adaptive=True)
    for step in range(4):
        route.update(state(units, objectives=objects, step=step))
    assert route.hq_secured
    objects[0].raw.update(percent=0, redTeamNum=0, blueteamNum=20)
    for step in (4, 5):
        s = state(units, objectives=objects, step=step)
        route.update(s)
    assert not route.hq_secured and route.objective(s).uid == objects[0].uid
    route.update(state(units, objectives=objects, episode=2))
    assert not route.hq_secured


def straight_column(units):
    objects = [objective(901, 100000, name='HQ')]
    route = AssaultRoute(adaptive=True)
    route.update(state(units, objectives=objects))
    route._routes = (((0.,0.,0.), (100000.,0.,0.)),)
    return objects, route, ColumnAdvance(), SimpleNamespace(front={u.uid for u in units if u.entity_type in ColumnAdvance.VEHICLES}, reserve={})


def test_leading_car_waits_but_rear_keeps_advancing_and_fire_is_never_blocked():
    foot = [entity(i, 0, i, 20000) for i in range(8)]
    car = entity(100, 0, 8, 28000, kind='BP_MNWS_Vehicle_6x6UGV_C')
    units = [*foot,car]
    objects, route, column, formation = straight_column(units)
    s = state(units, objectives=objects)
    column.begin(s, route)
    move = Action.move_at((50000,0,0))
    assert column.apply(s, car, move, route, formation).command == 'Guard'
    assert all(column.apply(s, u, move, route, formation).command == 'Moving' for u in foot)
    assert column.apply(s, car, Action.attack(999), route, formation).command == 'NormalAttacking'
    assert column.apply(s, car, Action.move_at((25000,0,0)), route, formation).command == 'Moving'


def test_stuck_tail_cannot_permanently_freeze_moving_main_body():
    rear = [entity(i, 0, i, 0) for i in range(2)]
    main = [entity(i+2, 0, i+2, 20000) for i in range(3)]
    units = rear+main
    objects, route, column, formation = straight_column(units)
    advance_frames = 0
    for step in range(70):
        s = state(units, objectives=objects, step=step)
        column.begin(s, route)
        actions = [column.apply(s, u, Action.move_at((80000,0,0)), route, formation) for u in units]
        assert all(a.command == 'Moving' for a in actions[:2])
        # Only the simulated rear is mechanically blocked; this is a navigation
        # liveness model, not UE physics or evidence of a score improvement.
        for u,a in zip(main,actions[2:]):
            if a.command == 'Moving':
                u.position[0] += 600
                advance_frames += 1
    assert {u.uid for u in rear} <= column.excluded
    assert advance_frames > 60 and min(u.position[0] for u in main) > 40000


def test_recovered_tail_rejoins_reference_and_disconnected_dead_never_anchor():
    units = [entity(i,0,i,20000) for i in range(6)]
    objects, route, column, formation = straight_column(units)
    units[0].alive = False
    units[1].raw['rSVD1'] = 'unit;0'
    column.begin(state(units, objectives=objects), route)
    assert {u.uid for u in column.foot} == {2,3,4,5}
    column._watch[2] = (20000,24,0)
    column.begin(state(units, objectives=objects, step=1), route)
    assert 2 in column.excluded
    units[2].position[0] += 400
    column.begin(state(units, objectives=objects, step=2), route)
    assert 2 not in column.excluded


def test_tail_joining_road_is_not_misclassified_as_stalled_while_arc_goes_backwards():
    units = [entity(i,0,i,20000,-20000) for i in range(5)]
    objects,route,column,formation=straight_column(units)
    for step in range(40):
        s=state(units,objectives=objects,step=step)
        column.begin(s,route)
        assert not column.excluded and len(column.foot)==5
        for u in units:
            column.apply(s,u,Action.move_at((10000,0,0)),route,formation)
            u.position[0]-=200
            u.position[1]+=400


def entry_case():
    objects = map_objectives()
    rally = AssaultRoute.LOWER[-1]
    units = [entity(i+1,0,i,rally[0]+(i%4)*400,rally[1],
                    kind='BP_RoboDog_C' if i%3==2 else 'BP_BaseSoldier_C') for i in range(20)]
    route, column, waves = AssaultRoute(adaptive=True), ColumnAdvance(), EntryWaves()
    return objects,units,route,column,waves


def test_entry_waits_for_main_body_then_releases_mixed_groups():
    objects,units,route,column,waves=entry_case()
    for u in units[3:]:u.position[0]+=20000
    s=state(units,objectives=objects)
    route.update(s); column.begin(s,route); waves.begin(s,route,column)
    assert not waves.released and len(waves._waiting)==3
    for u in units[3:]:u.position[0]-=20000
    s=state(units,objectives=objects,step=1)
    route.update(s); column.begin(s,route); waves.begin(s,route,column)
    assert len(waves.released)==8
    assert {u.entity_type for u in units if u.uid in waves.released}==ColumnAdvance.FOOT
    waiting=next(u for u in units if u.uid in waves._waiting)
    action=Action.move_at(objects[0].position)
    assert waves.apply(s,waiting,action,route,column).command=='Guard'
    assert waves.apply(s,waiting,Action.attack(999),route,column).command=='NormalAttacking'


def test_stalled_first_wave_releases_following_group_without_waiting_forever():
    objects,units,route,column,waves=entry_case()
    for step in (0,1,23,24):
        s=state(units,objectives=objects,step=step)
        route.update(s); column.begin(s,route); waves.begin(s,route,column)
        assert len(waves.released)==(16 if step==24 else 8)


def test_waiting_second_wave_does_not_tether_released_wave_at_the_corner():
    units=[entity(i,0,i,20000 if i<8 else 0) for i in range(20)]
    objects,route,column,formation=straight_column(units)
    s=state(units,objectives=objects)
    column.begin(s,route)
    move=Action.move_at((50000,0,0))
    assert column.apply(s,units[0],move,route,formation).command=='Guard'
    assert column.apply(s,units[0],move,route,formation,set(range(8))).command=='Moving'


def test_deliberate_column_wait_does_not_trigger_false_obstacle_escape():
    rear = [entity(i,0,i,20000) for i in range(8)]
    lead = entity(100,0,8,30000,kind='BP_MNWS_Vehicle_6x6UGV_C')
    units = rear + [lead]
    objects, route, _, _ = straight_column(units)
    rule = algorithm(units)
    rule._route = route
    rule._combat_action = lambda s, u: Action.move_at((80000,0,0))
    for step in range(20):
        s = state(units,objectives=objects,step=step)
        actions = rule.act(s)
        assert actions[-1].command == 'Guard'
        assert actions[-1].points == tuple(lead.position)
        assert lead.uid not in rule._motion_recovery._escape
        assert lead.uid not in rule._route._waypoint_watch
        assert lead.uid not in rule._hq_entry._leg_watch
        # Tail advances normally rather than being frozen by the lead's wait.
        assert all(a.command == 'Moving' for a in actions[:-1])
        for u in rear:
            u.position[0] += 100


def test_replanned_secondary_leg_retains_road_reference_in_both_directions():
    objects=map_objectives()
    units=guards_at(objects[0],count=8)
    route=AssaultRoute(adaptive=True)
    s=state(units,objectives=objects)
    route.update(s)
    route.hq_secured=True
    route._reorder(s,(901,902,903),1)
    column=ColumnAdvance()
    column.begin(s,route)
    assert len(column.path)>2  # _routes contains only one destination after replanning.
    assert column.project(objects[1].position,column.path)>column.project(objects[0].position,column.path)
    for u in units:u.position[:2]=objects[2].position[:2]
    # Return to B from A after an objective loss, so progress must reverse.
    column=ColumnAdvance()
    column.begin(state(units,objectives=objects,step=1),route)
    assert column.project(objects[1].position,column.path)>column.project(objects[2].position,column.path)


def test_hq_wave_membership_does_not_leak_into_secondary_objectives():
    objects,units,route,column,waves=entry_case()
    s=state(units,objectives=objects)
    route.update(s);column.begin(s,route);waves.begin(s,route,column)
    assert waves.waves
    route.hq_secured=True
    route._reorder(s,(901,902,903),1)
    s=state(units,objectives=objects,step=1)
    column.begin(s,route);waves.begin(s,route,column)
    assert not waves.waves and not waves.released and not waves._waiting
