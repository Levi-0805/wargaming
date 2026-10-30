import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from cssim_ui_flow import DesktopAdapter, FlowError, Snapshot, load_recipe, run_steps, visual_match


class Clock:
    def __init__(self): self.now = 0
    def __call__(self): return self.now
    def sleep(self, amount): self.now += amount


class Adapter:
    def __init__(self, before, after):
        self.before, self.after = before, after
        self.image = before
        self.calls, self.captures = [], 0
        self.meta = dict(hwnd=42, pid=77, exe_path='C:/CSSIM/Client/CSSIMClient-Win64-Shipping.exe',
                         client_rect=[100, 50, 1700, 950], captured_at_unix=1000)
        self.change_on_click = True
        self.break_after_click = False
        self.unexpected_before_second_capture = None
        self.fail_after_input = False

    def capture(self):
        self.captures += 1
        if self.fail_after_input and self.calls:
            raise RuntimeError('capture failed after input')
        if self.captures == 2 and self.unexpected_before_second_capture is not None:
            self.image = self.unexpected_before_second_capture
        if self.break_after_click and self.calls:
            self.meta['pid'] = 78
        return Snapshot(self.image, copy.deepcopy(self.meta))

    def click(self, x, y, snapshot):
        self.calls.append(('click', x, y, self.captures))
        if self.change_on_click:
            self.image = self.after

    def key(self, value, snapshot):
        self.calls.append(('key', value, self.captures))

    def text(self, value, snapshot):
        self.calls.append(('text', value, self.captures))
        self.image = self.after


def test_native_adapter_captures_without_moving_then_parks_explicitly(setup):
    tmp_path, _, fake, _ = setup
    events = []

    class Desktop:
        def capture(self, hwnd, path):
            events.append(('capture', hwnd, str(path)))
            path.write_bytes(cv2.imencode('.png', fake.image)[1].tobytes())
            return {**fake.meta, 'image_path': str(path)}

        def move(self, observed, x, y):
            events.append(('move', observed['image_path'], x, y))

    adapter = DesktopAdapter(42, tmp_path/'captures', desktop=Desktop())
    result = adapter.capture()
    assert [event[0] for event in events] == ['capture']
    adapter.park(result)
    assert [event[0] for event in events] == ['capture', 'move']
    assert events[1][1] == events[0][2]
    assert events[1][2:] == (1500, 700)
    assert result.metadata['image_path'] == events[0][2]


def test_native_adapter_does_not_park_unknown_window_size(setup):
    tmp_path, _, fake, _ = setup
    fake.meta['client_rect'] = [100, 50, 1380, 770]

    class Desktop:
        def capture(self, hwnd, path):
            path.write_bytes(cv2.imencode('.png', fake.image)[1].tobytes())
            return fake.meta

        def move(self, *args):
            pytest.fail('must reject unknown dimensions before moving pointer')

    with pytest.raises(FlowError, match='1600x900'):
        adapter = DesktopAdapter(42, tmp_path/'captures', desktop=Desktop())
        adapter.park(adapter.capture())


@pytest.fixture
def setup(tmp_path):
    rng = np.random.default_rng(91)
    patterns = [rng.integers(0, 256, (12, 16, 3), dtype=np.uint8) for _ in range(2)]
    images = [np.zeros((900, 1600, 3), dtype=np.uint8) for _ in range(2)]
    for name, pattern, image in zip(('before', 'after'), patterns, images):
        (tmp_path / (name+'.png')).write_bytes(cv2.imencode('.png', pattern)[1].tobytes())
        image[20:32, 30:46] = pattern
    recipe = dict(size=[1600,900], steps=[dict(name='login', phase='pre_python',
        before=dict(template=str(tmp_path/'before.png'), region=[0,0,100,60]),
        actions=[dict(click=[50,40])],
        after=dict(template=str(tmp_path/'after.png'), region=[0,0,100,60]), timeout=1)])
    return tmp_path, recipe, Adapter(*images), patterns


def run(setup, **kwargs):
    root, recipe, adapter, _ = setup
    clock = Clock()
    return run_steps(recipe, adapter, root/'logs', root/'STOP', clock=clock,
                     wall_clock=lambda:1000, sleep=clock.sleep, **kwargs)


def test_click_uses_new_capture_and_verifies_result(setup):
    _, _, adapter, _ = setup
    result = run(setup)
    assert result['state'] == 'complete' and result['completed'] == ['login']
    assert adapter.calls == [('click',50,40,2)]
    assert adapter.captures >= 4
    assert result['pending'] is None


def test_multi_action_field_fill_refreshes_after_each_input(setup):
    _, recipe, adapter, _ = setup
    recipe['steps'][0]['actions'] += [dict(key='CTRL+A'), dict(text='10')]
    adapter.change_on_click = False
    result = run(setup)
    assert result['state'] == 'complete'
    assert [c[:2] for c in adapter.calls] == [('click',50), ('key','Ctrl+A'), ('text','10')]
    assert [c[-1] for c in adapter.calls] == [2,4,6]


def test_field_edit_keeps_pointer_in_place_until_text_is_complete(setup):
    _, recipe, adapter, _ = setup
    recipe['steps'][0]['actions'] += [dict(key='Ctrl+A'), dict(text='10')]
    adapter.change_on_click = False
    adapter.park = lambda snapshot: adapter.calls.append(('park', adapter.captures))
    result = run(setup)
    assert result['state'] == 'complete'
    assert [call[0] for call in adapter.calls] == ['park', 'click', 'key', 'text', 'park']
    assert result['last_observation']['captured_at_unix'] == 1000


def test_state_change_before_click_fails_without_input(setup):
    _, _, adapter, _ = setup
    adapter.unexpected_before_second_capture = adapter.after
    result = run(setup)
    assert result['state'] == 'needs_attention' and not adapter.calls
    assert 'immediately before' in result['error']


@pytest.mark.parametrize('change', ['size', 'pid_after_click', 'stale', 'wrong_app'])
def test_invalid_or_changed_window_never_receives_later_input(setup, change):
    _, recipe, adapter, _ = setup
    recipe['steps'][0]['actions'] += [dict(key='Ctrl+A'), dict(text='10')]
    if change == 'size': adapter.meta['client_rect'][2] = 1701
    elif change == 'stale': adapter.meta['captured_at_unix'] = 990
    elif change == 'wrong_app': adapter.meta['exe_path'] = 'C:/other.exe'
    else: adapter.break_after_click = True
    result = run(setup)
    assert result['state'] == 'needs_attention'
    assert len(adapter.calls) == (1 if change == 'pid_after_click' else 0)


def test_ambiguous_duplicate_template_refuses_click(setup):
    _, _, adapter, patterns = setup
    adapter.before[40:52,60:76] = patterns[0]
    result = run(setup)
    assert result['state'] == 'needs_attention' and 'Ambiguous' in result['error']
    assert not adapter.calls


def test_missing_template_times_out_without_input(setup):
    _, _, adapter, _ = setup
    adapter.image = np.zeros_like(adapter.before)
    result = run(setup)
    assert result['state'] == 'needs_attention' and 'Timed out' in result['error']
    assert not adapter.calls


def test_stopfile_prevents_capture_and_input(setup):
    root, _, adapter, _ = setup
    (root/'STOP').touch()
    result = run(setup)
    assert result['state'] == 'stopped' and adapter.captures == 0 and not adapter.calls


def test_uncertain_input_is_only_resumed_if_after_state_is_visible(setup):
    _, _, adapter, _ = setup
    adapter.fail_after_input = True
    first = run(setup)
    assert first['state'] == 'needs_attention' and first['pending']['name'] == 'login'
    adapter.fail_after_input = False
    adapter.image = adapter.before
    second = run(setup)
    assert second['state'] == 'needs_attention' and 'uncertain' in second['error']
    assert len(adapter.calls) == 1
    adapter.image = adapter.after
    third = run(setup)
    assert third['state'] == 'complete' and len(adapter.calls) == 1


def test_completed_steps_are_not_reclicked_and_changed_templates_invalidate_progress(setup):
    root, _, adapter, _ = setup
    assert run(setup)['state'] == 'complete'
    assert run(setup)['state'] == 'complete' and len(adapter.calls) == 1
    (root/'before.png').write_bytes((root/'after.png').read_bytes())
    with pytest.raises(FlowError, match='changed'):
        run(setup)


def test_phases_and_no_input_wait_steps(setup):
    _, recipe, adapter, _ = setup
    after_step = copy.deepcopy(recipe['steps'][0])
    after_step.update(name='start', phase='post_python', actions=[])
    after_step['before'] = copy.deepcopy(after_step['after'])
    recipe['steps'].append(after_step)
    assert run(setup)['completed'] == ['login']
    result = run(setup, phase='post_python')
    assert result['completed'] == ['login','start'] and len(adapter.calls) == 1


@pytest.mark.parametrize('change', ['multi_click', 'low_threshold', 'flat_template', 'flat_color_template',
                                     'key_only', 'newline_text'])
def test_recipe_rejects_blind_input_or_weak_visual_proofs(setup, change):
    root, recipe, adapter, _ = setup
    step = recipe['steps'][0]
    if change == 'multi_click': step['actions'] += [dict(click=[10,10])]
    elif change == 'low_threshold': step['before']['threshold'] = .7
    elif change in ('flat_template', 'flat_color_template'):
        pixels = np.zeros((12,16,3), np.uint8)
        if change == 'flat_color_template': pixels[:] = [0,100,200]
        (root/'before.png').write_bytes(cv2.imencode('.png', pixels)[1].tobytes())
    elif change == 'key_only': step['actions'] = [dict(key='Enter')]
    else: step['actions'] += [dict(key='Ctrl+A'),dict(text='10\n')]
    with pytest.raises(FlowError):
        run(setup)
    assert not adapter.calls


def test_completed_progress_cannot_claim_current_unknown_screen_is_ready(setup):
    _, _, adapter, _ = setup
    assert run(setup)['state'] == 'complete'
    adapter.image = adapter.before
    assert run(setup)['state'] == 'needs_attention'
    assert len(adapter.calls) == 1


def test_all_required_visual_evidence_must_match_before_input(setup):
    _, recipe, adapter, _ = setup
    step = recipe['steps'][0]
    original = copy.deepcopy(step['before'])
    step['before'] = {'all_of': [original, copy.deepcopy(step['after'])]}
    result = run(setup)
    assert result['state'] == 'needs_attention' and not adapter.calls


def test_all_of_success_and_invalid_empty_predicate(setup):
    _, recipe, adapter, _ = setup
    step = recipe['steps'][0]
    step['before'] = {'all_of': [step['before']]}
    step['after'] = {'all_of': [step['after']]}
    assert run(setup)['state'] == 'complete'
    step['before'] = {'all_of': []}
    with pytest.raises(FlowError, match='all_of'):
        load_recipe(recipe)
