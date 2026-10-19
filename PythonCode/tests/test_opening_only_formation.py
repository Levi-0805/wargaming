"""Progress contracts after retiring repeated formation stops."""
from cssim.protocol import Action
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.AssaultRoute import AssaultRoute
from test_fslutk_rule import algorithm, entity, map_objectives, state


def test_first_entrance_wait_cannot_wait_forever_for_a_fighting_rear_group():
    objects = map_objectives()
    x, y, _ = AssaultRoute.LOWER[-1]
    units = [entity(i+1, 0, i, x+(0 if i<6 else 40000), y) for i in range(20)]
    default = algorithm(units)
    legacy = algorithm(units, enable_continuous_column=True, enable_entry_waves=True)
    # The rear group keeps guarding/fighting and never moves toward assembly.
    # This is a liveness counterexample, not a simulation of battle outcomes.
    for rule in (default, legacy):
        rule._combat_action = lambda s, u: (Action.move_at(objects[0].position)
            if u.uid <= 6 else Action.guard_position(u.position))
    released_steps = []
    for step in range(70):
        s = state(units, objectives=objects, step=step)
        new = default.act(s)
        old = legacy.act(s)
        assert all(a.command == 'Guard' for a in old[:6])
        assert not legacy._column.excluded  # Guard is not a failed movement.
        if new[0].command == 'Moving':
            released_steps.append(step)
    assert released_steps and min(released_steps) <= default._cohesion.MAX_WAIT
    assert not default._entry_waves.waves and not default._entry_waves.held
    assert not default._column.held


def test_column_waits_at_opening_then_releases_and_rearms_only_next_episode():
    foot = [entity(i, 0, i, *AssaultRoute.LOWER[5][:2]) for i in range(8)]
    car = entity(100, 0, 8, *AssaultRoute.LOWER[6][:2], kind='BP_MNWS_Vehicle_6x6UGV_C')
    units = foot + [car]
    objects = map_objectives()
    rule = algorithm(units)
    point = AssaultRoute.LOWER[7]
    rule._combat_action = lambda s, u: Action.move_at(point)
    first = rule.act(state(units, objectives=objects))[-1]
    assert first == Action.guard_position(car.position)
    rule._cohesion.released = True
    moving = rule.act(state(units, objectives=objects, step=1))[-1]
    assert moving == Action.guard_at(point)  # A forward patrol, not a stop.
    assert not rule._column.held
    rule.act(state(units, objectives=objects, episode=2))
    assert not rule._cohesion.released and car.uid in rule._column.held


def test_early_spacing_wait_expires_before_reaching_the_rendezvous_radius():
    rear = [entity(i, 0, i, *AssaultRoute.LOWER[1][:2]) for i in range(8)]
    lead = [entity(i+8, 0, i+8, *AssaultRoute.LOWER[3][:2]) for i in range(4)]
    units = rear + lead
    objects = map_objectives()
    rule = algorithm(units)
    rule._combat_action = lambda s, u: (Action.move_at(AssaultRoute.LOWER[4])
        if u.uid >= 8 else Action.guard_position(u.position))
    for step in range(rule._cohesion.MAX_WAIT + 1):
        actions = rule.act(state(units, objectives=objects, step=step))
        assert rule._cohesion.started is None
        assert not rule._cohesion.released
        if step < rule._cohesion.MAX_WAIT:
            assert all(a == Action.guard_position(u.position) for a, u in zip(actions[8:], lead))
        else:
            assert all(a.command == 'Moving' for a in actions[8:])
            assert not rule._column.held
