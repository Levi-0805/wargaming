from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultFormation import AssaultFormation
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from cssim.protocol import Action
from test_fslutk_rule import entity, map_objectives, state


def setup():
    objects = map_objectives()
    x, y = objects[0].position[:2]
    foot = [entity(i + 1, 0, i, x + 30000, y) for i in range(10)]
    cars = [entity(100+i, 0, i+10, x+40000, y,
                   kind='BP_MNWS_Vehicle_Armored_C' if i % 2 else 'BP_MNWS_Vehicle_6x6UGV_C') for i in range(8)]
    route, formation = AssaultRoute(adaptive=True), AssaultFormation()
    s = state(foot + cars, objectives=objects)
    route.update(s)
    formation.begin(s, route)
    return foot, cars, objects, route, formation


def test_half_vehicles_lead_and_two_are_reserved_for_each_secondary():
    foot, cars, objects, route, form = setup()
    assert len(form.front) == 4
    assert sorted(form.reserve.values()).count(objects[1].uid) == 2
    assert sorted(form.reserve.values()).count(objects[2].uid) == 2
    s = state(foot+cars, objectives=objects)
    assert all(form.choose(s, u, route).command == 'Guard' for u in foot)
    assert all(form.choose(s, u, route) is None for u in cars)
    rear = foot[-1]
    rear.position[0] += 20000
    assert form.choose(s, rear, route) is None


def test_reserves_stop_before_first_objective_and_release_for_their_target():
    foot, cars, objects, route, form = setup()
    reserve = next(u for u in cars if u.uid in form.reserve)
    reserve.position[0] = objects[0].position[0] + 11000
    s = state(foot+cars, objectives=objects, step=1)
    route.update(s)
    form.begin(s, route, recon_ready=True)
    assert form.choose(s, reserve, route).command == 'Guard'
    target = form.reserve[reserve.uid]
    route._reorder(s, (target, *(uid for uid in route.order if uid != target)))
    s = state(foot+cars, objectives=objects, step=2)
    form.begin(s, route, recon_ready=True)
    assert form.choose(s, reserve, route) is None


def test_recon_wait_then_vehicle_volley_releases_capture_wave():
    foot, cars, objects, route, form = setup()
    foot.append(entity(1000, 0, 99, 0, kind='BP_Base_UAV_C'))
    front = [u for u in cars if u.uid in form.front]
    for u in front:
        u.position[0] = objects[0].position[0] + 11000
    s = state(foot+cars, objectives=objects, step=1)
    form.begin(s, route)
    assert form.choose(s, front[0], route).command == 'Guard'
    s = state(foot+cars, objectives=objects, step=2)
    form.begin(s, route, recon_ready=True)
    assert form.choose(s, front[0], route) is None
    enemy = entity(999, 1, 0, objects[0].position[0]+5000, objects[0].position[1])
    for u in front[:2]:
        u.raw['agentPerception'] = [999]
        u.raw['interaction'] = ['100']
        form.record(state(foot+cars, [enemy], objects, step=2), u, Action.attack(enemy), route)
        u.raw['interaction'] = ['99']
    s = state(foot+cars, objectives=objects, step=3)
    form.begin(s, route, recon_ready=True)
    assert all(form.choose(s, u, route) is None for u in foot)


def test_attack_order_without_ammo_use_does_not_confirm_vehicle_volley():
    foot, cars, objects, route, form = setup()
    front = [u for u in cars if u.uid in form.front]
    enemy = entity(999, 1, 0, *objects[0].position[:2])
    for u in front:
        u.raw.update(agentPerception=[999], interaction=['100'])
        form.record(state(foot+cars, [enemy], objects), u, Action.attack(enemy), route)
    form.begin(state(foot+cars, [enemy], objects, step=1), route, recon_ready=True)
    assert not form._gates[objects[0].uid]['released']


def test_distant_vehicle_contact_does_not_pull_reserve_into_first_wave():
    foot, cars, objects, route, form = setup()
    reserve = next(u for u in cars if u.uid in form.reserve)
    reserve.position[0] = objects[0].position[0] + 11000
    enemy = entity(999, 1, 0, reserve.position[0]-20000, reserve.position[1])
    reserve.raw['agentPerception'] = [999]
    s = state(foot+cars, [enemy], objects, step=1)
    form.begin(s, route, recon_ready=True)
    assert form.choose(s, reserve, route).command == 'Guard'


def test_no_endless_wait_or_recall_of_units_already_inside():
    foot, cars, objects, route, form = setup()
    s = state(foot+cars, objectives=objects, step=28)
    form.begin(s, route)
    assert form.choose(s, foot[0], route) is None
    foot[0].position[:] = objects[0].position
    s = state(foot+cars, objectives=objects, step=29)
    form.begin(s, route)
    assert form.choose(s, foot[0], route) is None


def test_dead_lead_promotes_reserve_and_local_contacts_bypass_assembly():
    foot, cars, objects, route, form = setup()
    for u in cars:
        if u.uid in form.front:
            u.alive = False
    enemy = entity(999, 1, 0, *foot[0].position[:2])
    foot[0].raw['agentPerception'] = [999]
    s = state(foot+cars, [enemy], objects, step=1)
    form.begin(s, route)
    assert len(form._active) == 1 and len(form.reserve) == 3
    assert form.choose(s, foot[0], route) is None


def test_episode_reset_restores_assignments_and_clears_volley_history():
    foot, cars, objects, route, form = setup()
    form._gates[objects[0].uid]['released'] = True
    s = state(foot+cars, objectives=objects, episode=2)
    route.update(s)
    form.begin(s, route)
    assert not form._gates[objects[0].uid]['released']
    assert len(form.front) == len(form.reserve) == 4


def test_six_vehicle_roster_leads_with_both_lynx_and_replenishes_after_hq():
    objects = map_objectives()
    cars = [entity(100+i, 0, i, 0, kind='BP_MNWS_Vehicle_Armored_C' if i < 4 else 'BP_MNWS_Vehicle_6x6UGV_C')
            for i in range(6)]
    route, form = AssaultRoute(adaptive=True), AssaultFormation()
    s = state(cars, objectives=objects)
    route.update(s)
    form.begin(s, route)
    assert len(form.front) == len(form.reserve) == 3
    assert len({u.entity_type for u in cars if u.uid in form.front}) == 2
    assert all(u.uid in form.front for u in cars if u.entity_type == 'BP_MNWS_Vehicle_6x6UGV_C')
    objects[0].raw.update(percent=1, redTeamNum=4)
    route.hq_secured = True
    s = state(cars, objectives=objects, step=1)
    form.begin(s, route)
    assert len(form.front) == 2
    assert all(list(form.reserve.values()).count(o.uid) == 2 for o in objects[1:])
