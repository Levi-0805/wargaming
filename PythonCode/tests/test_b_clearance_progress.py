"""B searches must distinguish attempted movement, reached goals and combat."""
import pytest

from cssim.protocol import Action
from TeamAlg.FsLutk7wD2RkNgbs.Algorithm.UrbanContact import UrbanContact
from test_b_point_assault import b_case
from test_fslutk_rule import entity, state


def probe_case():
    agents, cars, objects, rule = b_case()
    return agents, cars[0], objects, rule._route, UrbanContact()


def request(contact, agents, car, objects, route, step, *, episode=1):
    s = state(agents, objectives=objects, step=step, episode=episode)
    contact.begin(s)
    action = contact.search_point(s, car, objects[1], route)
    contact.record(s, car, action)
    return action


def inner(route, base, point):
    x, y, _ = route._local_point(base, point)
    extent = route._footprint(base)
    return max(abs(x)/extent[0], abs(y)/extent[1]) == pytest.approx(.3)


def test_near_goal_stall_allows_recovery_then_tries_another_inner_goal():
    agents, car, objects, route, contact = probe_case()
    first = request(contact, agents, car, objects, route, 0).points
    # The recorded failure stopped 7.1 m short; this is not a reached goal.
    car.position[:] = first
    car.position[0] += 710
    for step in range(1, contact.B_STALLED_MOVES + 1):
        action = request(contact, agents, car, objects, route, step)
        assert action.points == first
    action = request(contact, agents, car, objects, route, contact.B_STALLED_MOVES + 1)
    assert action.command == 'Moving' and action.target_uid is None
    assert action.points != first and inner(route, objects[1], action.points)
    # The replacement remains fixed too; it does not drift every frame.
    assert request(contact, agents, car, objects, route, contact.B_STALLED_MOVES + 2).points == action.points


def test_real_progress_resets_failed_movement_count():
    agents, car, objects, route, contact = probe_case()
    first = request(contact, agents, car, objects, route, 0).points
    for step in range(1, 22):
        if step in (10, 20):
            vector = first - car.position
            car.position[:] += vector / route.distance(first, car.position) * 250
        assert request(contact, agents, car, objects, route, step).points == first
    assert contact._b_search[car.uid]['stalled'] < contact.B_STALLED_MOVES


def test_navigation_timeout_does_not_claim_inner_goal_was_reached():
    agents, car, objects, route, contact = probe_case()
    first = request(contact, agents, car, objects, route, 0).points
    replacement = request(contact, agents, car, objects, route, contact.CLEAR_PLAN_STEPS)
    assert replacement.points != first and inner(route, objects[1], replacement.points)


def test_fire_interrupt_resumes_unfinished_goal_without_extending_deadline():
    agents, car, objects, route, contact = probe_case()
    first = request(contact, agents, car, objects, route, 0).points
    target = entity(9000, 1, 0, car.position[0] + 1000, car.position[1])
    until = contact._b_search[car.uid]['until']
    for step in range(1, 18):
        s = state(agents, [target], objects, step=step)
        contact.begin(s)
        contact.record(s, car, Action.attack(target.uid))
    assert contact._b_search[car.uid]['stalled'] == 1
    assert request(contact, agents, car, objects, route, 18).points == first
    assert contact._b_search[car.uid]['until'] == until


def test_combat_expiry_changes_probe_without_marking_untried_path_blocked():
    agents, car, objects, route, contact = probe_case()
    first = request(contact, agents, car, objects, route, 0).points
    target = entity(9000, 1, 0, car.position[0] + 1000, car.position[1])
    for step in range(1, contact.CLEAR_PLAN_STEPS):
        s = state(agents, [target], objects, step=step)
        contact.begin(s)
        contact.record(s, car, Action.attack(target.uid))
    action = request(contact, agents, car, objects, route, contact.CLEAR_PLAN_STEPS)
    assert action.points != first and inner(route, objects[1], action.points)
    assert not contact._b_failed


def test_short_contact_reposition_does_not_discard_unfinished_inner_goal():
    agents, car, objects, route, contact = probe_case()
    first = request(contact, agents, car, objects, route, 0).points
    target = entity(9000, 1, 0, car.position[0] + 1000, car.position[1])
    s = state(agents, [target], objects, step=1)
    contact.begin(s)
    action = contact.reposition(s, car, target, duration=2)
    contact.record(s, car, action)
    assert action.points != first
    assert request(contact, agents, car, objects, route, 3).points == first


def test_failed_goal_is_temporarily_excluded_and_all_failed_still_moves():
    agents, car, objects, route, contact = probe_case()
    request(contact, agents, car, objects, route, 0)
    failed_cursor = contact._b_search[car.uid]['cursor']
    for step in range(1, contact.B_STALLED_MOVES + 1):
        request(contact, agents, car, objects, route, step)
    failed_key = (car.uid, objects[1].uid, failed_cursor)
    assert contact._b_failed[failed_key] == contact.B_STALLED_MOVES + contact.B_FAILED_RETRY_STEPS
    until = contact._b_search[car.uid]['until']
    for i in range(16):
        contact._b_failed[(car.uid, objects[1].uid, i)] = until + 48 + i
    action = request(contact, agents, car, objects, route, until)
    assert action.command == 'Moving' and route.inside_garrison(objects[1], action.points)
    assert contact._b_search[car.uid]['cursor'] == 0


@pytest.mark.parametrize('reset', ['episode', 'rewind', 'cleared'])
def test_b_search_and_failure_history_reset(reset):
    agents, car, objects, route, contact = probe_case()
    request(contact, agents, car, objects, route, 0)
    for step in range(1, contact.B_STALLED_MOVES + 1):
        request(contact, agents, car, objects, route, step)
    assert contact._b_search and contact._b_failed
    if reset == 'cleared':
        objects[1].raw['blueteamNum'] = 0
        contact.begin(state(agents, objectives=objects, step=25))
        assert not contact._b_search
    else:
        contact.begin(state(agents, objectives=objects, step=0, episode=2 if reset == 'episode' else 1))
        assert not contact._b_search and not contact._b_failed
