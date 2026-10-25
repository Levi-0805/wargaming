"""Contested B must explore inward and escape ineffective dense-target fire."""
import pytest

from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultFormation import AssaultFormation
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.UrbanContact import UrbanContact
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.VehicleFireControl import VehicleFireControl
from test_fslutk_rule import algorithm, entity, guards_at, map_objectives, state


def b_case():
    objects = map_objectives()
    hq, base, _ = objects
    hq.raw.update(percent=1, redTeamNum=4, blueteamNum=0)
    base.raw.update(scale={'x': 10500, 'y': 9500}, rotation={'yaw': -12.8125},
                    percent=0, redTeamNum=0, blueteamNum=10)
    home = guards_at(hq, first=100, count=4)
    # Reached positions in the Oct24 14:32 batch, not enemy replay coordinates.
    cars = [entity(461+i, 0, 4+i, -52356-i*350, -23338+i*140,
                   kind='BP_MNWS_Vehicle_6x6UGV_C' if i == 0 else 'BP_MNWS_Vehicle_Armored_C')
            for i in range(3)]
    for car in cars:
        car.position[2] = 5520
        car.raw.update(agentPerception=[], interaction=['126'], rSVD1='car;1')
    agents = home + cars
    rule = algorithm(agents)
    for step in range(4):
        rule.act(state(agents, objectives=objects, step=step))
    assert rule._route.hq_secured and rule._route.objective(state(agents, objectives=objects)).uid == base.uid
    return agents, cars, objects, rule


def dense_case():
    car = entity(1, 0, 0, 0, kind='BP_MNWS_Vehicle_6x6UGV_C')
    enemies = [entity(20+i, 1, i, 5000+i*100) for i in range(4)]
    car.raw.update(agentPerception=[e.uid for e in enemies], interaction=['200'], rSVD1='car;1')
    return car, enemies, VehicleFireControl()


@pytest.mark.parametrize('kind', ['BP_BaseSoldier_C', 'BP_RoboDog_C', *sorted(VehicleFireControl.VEHICLES)])
def test_first_b_search_moves_inward_not_to_nearest_outer_friendly(kind):
    agents, cars, objects, rule = b_case()
    base = objects[1]
    unit = cars[0]
    unit.entity_type = kind
    contact = UrbanContact()
    s = state(agents, objectives=objects, step=20)
    contact.begin(s)
    action = contact.search_point(s, unit, base, rule._route)
    assert action.command == 'Moving' and action.target_uid is None
    assert rule._route.inside_garrison(base, action.points)
    extent = rule._route._footprint(base)
    x, y, _ = rule._route._local_point(base, action.points)
    assert max(abs(x)/extent[0], abs(y)/extent[1]) == pytest.approx(.3)
    assert rule._route.distance(action.points, base.position) < rule._route.distance(unit.position, base.position)
    assert action.points[2] == unit.position[2]


def test_first_vehicle_goals_are_spread_and_reach_supported_guard_patrol():
    agents, cars, objects, rule = b_case()
    actions = rule.act(state(agents, objectives=objects, step=4))[-3:]
    goals = [rule._urban._plans[u.uid][1] for u in cars]
    assert len(set(goals)) == 3
    for car, action, goal in zip(cars, actions, goals):
        assert rule._route.inside_garrison(objects[1], goal)
        assert rule._route.distance(goal, objects[1].position) < 5000
        assert action.command == 'Guard' and action.target_uid is None
    # Heading alignment can precede waypoint patrol by at most two frames.
    actions = rule.act(state(agents, objectives=objects, step=5))[-3:]
    assert all(a.command == 'Guard' and a.points == p for a, p in zip(actions, goals))


def test_same_lane_remainder_does_not_stack_vehicles_on_one_destination():
    agents, cars, objects, rule = b_case()
    contact = UrbanContact()
    for i, car in enumerate(cars):
        car.team_index = 4 + 3*i
    s = state(agents, objectives=objects, step=20)
    contact.begin(s)
    goals = [contact.search_point(s, car, objects[1], rule._route).points for car in cars]
    assert len(set(goals)) == 3
    assert all(rule._route.distance(a, b) > 600 for i, a in enumerate(goals) for b in goals[i+1:])


def test_b_search_alternates_reached_inner_and_outer_views_and_never_leaves_rotated_box():
    agents, cars, objects, rule = b_case()
    contact, car, base = UrbanContact(), cars[0], objects[1]
    extent = rule._route._footprint(base)
    radii = []
    for step in range(0, 12, 3):
        s = state(agents, objectives=objects, step=step)
        contact.begin(s)
        action = contact.search_point(s, car, base, rule._route)
        x, y, _ = rule._route._local_point(base, action.points)
        radii.append(max(abs(x)/extent[0], abs(y)/extent[1]))
        assert rule._route.inside_garrison(base, action.points)
        car.position[:] = action.points
        for scan_step in (step + 1, step + 2):
            s = state(agents, objectives=objects, step=scan_step)
            contact.begin(s)
            assert contact.search_point(s, car, base, rule._route).command == 'Guard'
    assert radii == pytest.approx([.3, .8, .3, .8])


def test_b_blocked_fixed_goal_still_allows_real_vehicle_escape():
    agents, cars, objects, rule = b_case()
    car = cars[0]
    first = rule._urban._plans[car.uid][1]
    actions = [rule.act(state(agents, objectives=objects, step=step))[-3] for step in range(4, 17)]
    assert any(a.command == 'Moving' and a.points is None for a in actions)
    assert rule._urban._plans[car.uid][1] == first


def test_dense_targets_cannot_reset_vehicle_wide_failed_fire_budget():
    car, enemies, control = dense_case()
    actions = []
    for step in range(12):
        s = state([car], enemies, step=step)
        actions.append(control.choose(s, car, advance_on_stall=True))
    assert sum(a is not None and a.command == 'NormalAttacking' for a in actions) == 4
    assert control.advancing(s, car)
    assert all(a is None for a in actions[6:])
    # Repeated decisions in a frame do not consume or extend a movement window.
    until = control._progress[car.uid]['until']
    for _ in range(5):
        assert control.choose(s, car, advance_on_stall=True) is None
    assert control._progress[car.uid]['until'] == until


@pytest.mark.parametrize('release', ['distance', 'deadline', 'scope', 'episode', 'rewind'])
def test_bounded_fire_maneuver_releases_and_resets(release):
    car, enemies, control = dense_case()
    for step in range(7):
        control.choose(state([car], enemies, step=step), car, advance_on_stall=True)
    assert control.advancing(state([car], enemies, step=6), car)
    step, episode, enabled = 7, 1, True
    if release == 'distance':
        car.position[1] += control.ADVANCE_DISTANCE_CM
    elif release == 'deadline':
        step = control._progress[car.uid]['until']
    elif release == 'scope':
        enabled = False
    elif release == 'episode':
        step, episode = 0, 2
    else:
        step = 0
    s = state([car], enemies, step=step, episode=episode)
    action = control.choose(s, car, advance_on_stall=enabled)
    assert not control.advancing(s, car)
    assert action is not None and action.command == 'Guard'


def test_effective_fire_continues_and_unknown_or_shared_targets_do_not_authorize_fire():
    car, enemies, control = dense_case()
    for step in range(20):
        car.raw['interaction'] = [str(200-step)]
        enemies[0].hp = 100-step
        s = state([car], enemies, step=step)
        action = control.choose(s, car, advance_on_stall=True)
        assert action.command == ('Guard' if step == 0 else 'NormalAttacking')
        assert not control.advancing(s, car)
    car.raw['agentPerception'] = []
    assert control.choose(state([car], enemies, step=20), car, advance_on_stall=True) is None


@pytest.mark.parametrize('evidence', ['ammo_only', 'damage_only'])
def test_partial_fire_evidence_does_not_reset_cross_target_failure_budget(evidence):
    car, enemies, control = dense_case()
    for step in range(12):
        if evidence == 'ammo_only':
            car.raw['interaction'] = [str(200-step)]
        else:
            for target in enemies:
                target.hp = 100-step
        s = state([car], enemies, step=step)
        control.choose(s, car, advance_on_stall=True)
    assert control.advancing(s, car)


def test_full_rule_turns_failed_b_bursts_into_navigation_then_resumes_local_aim():
    agents, cars, objects, rule = b_case()
    car = cars[0]
    enemies = [entity(9000+i, 1, i, car.position[0]+5000+i*100, car.position[1]) for i in range(4)]
    for enemy in enemies:
        enemy.position[2] = car.position[2]
    car.raw['agentPerception'] = [e.uid for e in enemies]
    actions = []
    for step in range(4, 18):
        s = state(agents, enemies, objects, step=step)
        actions.append(rule.act(s)[-3])
    assert sum(a.command == 'NormalAttacking' for a in actions) == 4
    assert any(a.command == 'Guard' and a.points is not None for a in actions[6:])
    assert rule._vehicle_fire.advancing(s, car)
    car.position[0] += 2100
    action = rule.act(state(agents, enemies, objects, step=18))[-3]
    assert not rule._vehicle_fire.advancing(state(agents, enemies, objects, step=18), car)
    assert action.command == 'Guard' and action.points is None


def test_contested_b_reserve_outside_box_is_not_parked_by_a_full_bar():
    agents, cars, objects, rule = b_case()
    base, car = objects[1], cars[0]
    base.raw.update(percent=1, redTeamNum=12, blueteamNum=10)
    car.position[:2] = base.position[:2] + [15000, 0]
    form = AssaultFormation()
    s = state(agents, objectives=objects, step=40)
    form.begin(s, rule._route, recon_ready=True)
    form.reserve[car.uid] = base.uid
    assert rule._route.owned(s, base) and not rule._route.inside_garrison(base, car.position)
    assert form.choose(s, car, rule._route) is None
    base.raw['blueteamNum'] = 0
    assert form.choose(s, car, rule._route).command == 'Guard'
