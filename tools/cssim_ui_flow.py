"""Fail-closed visual CSSIM recipes, with fresh evidence before every input.

This module never launches processes or executes shell commands. It supports a
single click per step, optionally followed by Ctrl+A and literal field text.
Interrupted input is never blindly repeated: only a matching final-state image
can complete an uncertain step during recovery.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

SIZE = (1600, 900)
CLIENT = 'cssimclient-win64-shipping.exe'


class FlowError(ValueError):
    pass


@dataclass
class Snapshot:
    image: np.ndarray
    metadata: dict


def read_image(path):
    pixels = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise FlowError(f'Cannot decode visual template/capture: {path}')
    return pixels


class DesktopAdapter:
    """Thin adapter around the scoped native desktop implementation."""
    def __init__(self, hwnd, capture_dir, desktop=None):
        if desktop is None:
            from cssim_desktop import Desktop
            desktop = Desktop()
        self.desktop, self.hwnd = desktop, int(hwnd)
        self.capture_dir = Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.serial = 0

    def capture(self):
        while True:
            self.serial += 1
            path = self.capture_dir / f'frame-{self.serial:06d}.png'
            if not path.exists():
                break
        meta = self.desktop.capture(self.hwnd, path)
        return Snapshot(read_image(path), meta)

    def park(self, snapshot):
        # Call only between steps: moving away during a spinbox text edit can
        # discard its input focus, even though no mouse button is pressed.
        identity(snapshot)
        self.desktop.move(snapshot.metadata, 1500, 700)

    def click(self, x, y, snapshot):
        self.desktop.click(snapshot.metadata, x, y)

    def key(self, key, snapshot):
        self.desktop.key(snapshot.metadata, key)

    def text(self, text, snapshot):
        self.desktop.type(snapshot.metadata, text)


def identity(snapshot):
    meta = snapshot.metadata
    try:
        rect = list(meta['client_rect'])
        pid, hwnd = meta['pid'], meta['hwnd']
        executable = str(meta['exe_path']).replace('\\', '/').lower()
        if type(pid) is not int or pid <= 0 or type(hwnd) is not int or hwnd <= 0:
            raise ValueError('invalid pid/hwnd')
        if len(rect) != 4 or any(type(n) is not int for n in rect):
            raise ValueError('invalid client rectangle')
        if executable.rsplit('/', 1)[-1] != CLIENT:
            raise ValueError('not the CSSIM client')
        if (rect[2]-rect[0], rect[3]-rect[1]) != SIZE or snapshot.image.shape[:2] != (SIZE[1], SIZE[0]):
            raise ValueError('require observed 1600x900 client area')
        if snapshot.image.dtype != np.uint8 or snapshot.image.ndim != 3 or snapshot.image.shape[2] != 3:
            raise ValueError('require a BGR uint8 capture')
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise FlowError(f'Invalid desktop observation: {exc}') from exc
    return dict(hwnd=hwnd, pid=pid, exe_path=executable, client_rect=rect)


def fresh(snapshot, expected_identity, wall_clock):
    if identity(snapshot) != expected_identity:
        raise FlowError('Window identity or rectangle changed; rebind from a new observation')
    captured = snapshot.metadata.get('captured_at_unix')
    if type(captured) not in (int, float) or not 0 <= wall_clock()-captured <= 5:
        raise FlowError('Screenshot metadata is stale or has an invalid timestamp')


def visual_match(snapshot, pattern):
    """Require one connected high-confidence template match within the ROI."""
    if 'all_of' in pattern:
        matches = [visual_match(snapshot, child) for child in pattern['all_of']]
        return dict(matched=all(item['matched'] for item in matches),
                    score=min(item['score'] for item in matches), evidence=matches)
    x, y, w, h = pattern['region']
    patch = snapshot.image[y:y+h, x:x+w]
    template = pattern['_pixels']
    response = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
    if not np.isfinite(response).all():
        raise FlowError('Nonfinite visual match result')
    _, peak, _, location = cv2.minMaxLoc(response)
    mask = (response >= pattern['threshold']).astype(np.uint8)
    components, _ = cv2.connectedComponents(mask, connectivity=8)
    if components > 2:
        raise FlowError(f'Ambiguous visual state: {pattern["template"]}')
    if components == 2:
        ys, xs = np.nonzero(mask)
        if (xs.max()-xs.min()+1 > max(3, template.shape[1]//4)
                or ys.max()-ys.min()+1 > max(3, template.shape[0]//4)):
            raise FlowError(f'Visual state has a broad ambiguous match plateau: {pattern["template"]}')
    return dict(template=pattern['template'], matched=components == 2, score=float(peak),
                position=[x+location[0], y+location[1]])


def pattern_label(pattern):
    if 'all_of' in pattern:
        return 'all_of(' + ', '.join(pattern_label(child) for child in pattern['all_of']) + ')'
    return pattern['template']


def load_recipe(recipe):
    if isinstance(recipe, (str, Path)):
        path = Path(recipe).resolve()
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        base = path.parent
    else:
        value = json.loads(json.dumps(recipe))
        base = Path.cwd()
    if not isinstance(value, dict) or tuple(value.get('size', SIZE)) != SIZE:
        raise FlowError('Recipe must use the calibrated 1600x900 client size')
    steps = value.get('steps')
    if not isinstance(steps, list) or not steps:
        raise FlowError('Recipe needs an ordered nonempty steps list')
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())

    def compile_pattern(pattern, depth=0):
        if not isinstance(pattern, dict):
            raise FlowError('Require template evidence or all_of evidence')
        if 'all_of' in pattern:
            children = pattern['all_of']
            if (depth or set(pattern) != {'all_of'} or not isinstance(children, list)
                    or not 1 <= len(children) <= 16):
                raise FlowError('all_of must contain 1..16 nonnested visual predicates')
            for child in children:
                compile_pattern(child, depth+1)
            return
        if not isinstance(pattern.get('template'), str):
            raise FlowError('Require an explicit template path')
        region = pattern.get('region')
        threshold = pattern.get('threshold', .93)
        if (not isinstance(region, list) or len(region) != 4 or any(type(n) is not int for n in region)
                or min(region[:2]) < 0 or min(region[2:]) <= 0
                or region[0]+region[2] > SIZE[0] or region[1]+region[3] > SIZE[1]):
            raise FlowError('Invalid template search region')
        if type(threshold) not in (int, float) or not .93 <= threshold <= 1:
            raise FlowError('Visual thresholds must be at least 0.93')
        template = (base / pattern['template']).resolve()
        raw = template.read_bytes()
        digest.update(str(template).encode()); digest.update(raw)
        pixels = read_image(template)
        spatial_variance = pixels.astype(np.float32).std(axis=(0, 1)).max()
        if pixels.shape[1] > region[2] or pixels.shape[0] > region[3] or spatial_variance < 1:
            raise FlowError('Template must fit its region and contain distinguishing visual detail')
        pattern.update(threshold=float(threshold), _pixels=pixels)

    names = set()
    for step in steps:
        name = step.get('name')
        if not isinstance(name, str) or not name or name in names:
            raise FlowError('Every step requires a unique nonempty name')
        names.add(name)
        phase = step.get('phase', step.get('entry', 'pre_python'))
        if phase not in ('pre_python', 'post_python'):
            raise FlowError('Step phase must be pre_python or post_python')
        step['phase'] = phase
        timeout = step.get('timeout', 30)
        if type(timeout) not in (int, float) or not 0 < timeout <= 300:
            raise FlowError('Step timeout must be within 0..300 seconds')
        step['timeout'] = float(timeout)
        actions = step.get('actions', [])
        if not isinstance(actions, list) or len(actions) not in (0, 1, 3):
            raise FlowError('Use one click per step, optionally click + Ctrl+A + text')
        if actions:
            if not isinstance(actions[0], dict) or set(actions[0]) != {'click'}:
                raise FlowError('Input steps must begin with one explicit click')
            point = actions[0]['click']
            if (not isinstance(point, list) or len(point) != 2 or any(type(n) is not int for n in point)
                    or not 0 <= point[0] < SIZE[0] or not 0 <= point[1] < SIZE[1]):
                raise FlowError('Click is outside the calibrated client area')
            if len(actions) == 3:
                if actions[1] not in ({'key': 'Ctrl+A'}, {'key': 'CTRL+A'}):
                    raise FlowError('Field input must explicitly select all before replacing text')
                if (not isinstance(actions[2], dict) or set(actions[2]) != {'text'}
                        or not isinstance(actions[2]['text'], str) or len(actions[2]['text']) > 200
                        or any(ord(c) < 32 for c in actions[2]['text'])):
                    raise FlowError('Field text must be bounded literal text without control characters')
                actions[1]['key'] = 'Ctrl+A'
        step['actions'] = actions
        for label in ('before', 'after'):
            compile_pattern(step.get(label))
    return value, digest.hexdigest()


def save_progress(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def run_steps(recipe, adapter, logdir, stopfile, *, phase='pre_python', clock=time.monotonic,
              wall_clock=time.time, sleep=time.sleep, poll_interval=.5):
    value, recipe_sha = load_recipe(recipe)
    if phase not in ('pre_python', 'post_python') or not 0 < poll_interval <= 5:
        raise FlowError('Invalid phase or poll interval')
    steps = [step for step in value['steps'] if step['phase'] == phase]
    if not steps:
        raise FlowError(f'Recipe has no {phase} steps')
    progress_path = Path(logdir) / 'progress.json'
    progress = (json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else
                dict(recipe_sha256=recipe_sha, completed=[], pending=None, window_identity=None))
    if progress.get('recipe_sha256') != recipe_sha:
        raise FlowError('Recipe or template bytes changed; use a new verified run directory')
    completed = progress.get('completed', [])
    known = [step['name'] for step in value['steps']]
    if not isinstance(completed, list) or len(completed) != len(set(completed)) or any(n not in known for n in completed):
        raise FlowError('Malformed saved step progress')
    progress.update(phase=phase, state='running', error=None)

    def stopped():
        return stopfile is not None and Path(stopfile).exists()

    def capture():
        if stopped():
            raise InterruptedError('Stop file present')
        snapshot = adapter.capture()
        current = identity(snapshot)
        if progress['window_identity'] is None:
            progress['window_identity'] = current
        fresh(snapshot, progress['window_identity'], wall_clock)
        return snapshot

    def observe(pattern, deadline):
        while True:
            snapshot = capture()
            match = visual_match(snapshot, pattern)
            progress['last_observation'], progress['last_match'] = snapshot.metadata, match
            if match['matched']:
                return snapshot, match
            if clock() >= deadline:
                raise FlowError(f'Timed out waiting for visual state: {pattern_label(pattern)}')
            sleep(min(poll_interval, max(0, deadline-clock())))

    def park(snapshot=None):
        if callable(getattr(adapter, 'park', None)):
            if snapshot is None:
                snapshot = capture()
            fresh(snapshot, progress['window_identity'], wall_clock)
            adapter.park(snapshot)
            sleep(.35)

    try:
        park()
        advanced = False
        for step in steps:
            if step['name'] in completed:
                continue
            advanced = True
            progress['current_step'] = step['name']
            deadline = clock() + step['timeout']
            pending = progress.get('pending')
            if pending is not None:
                if pending.get('name') != step['name']:
                    raise FlowError('Pending input belongs to another step; cannot resume this phase')
                # An input may have happened before a crash. Never retry it.
                snapshot = capture()
                match = visual_match(snapshot, step['after'])
                if not match['matched']:
                    raise FlowError('Prior input outcome is uncertain; final-state template not present')
                progress['last_observation'] = snapshot.metadata
                progress['last_match'] = match
            else:
                snapshot, match = observe(step['before'], deadline)
                progress['last_observation'] = snapshot.metadata
                progress['last_match'] = match
                save_progress(progress_path, progress)
                for index, action in enumerate(step['actions']):
                    # Capture directly before the click; the earlier observation
                    # may have taken time to persist, so it is not reused.
                    snapshot = capture()
                    if index == 0 and not visual_match(snapshot, step['before'])['matched']:
                        raise FlowError('Visual state changed immediately before the click')
                    fresh(snapshot, progress['window_identity'], wall_clock)
                    progress['pending'] = dict(name=step['name'], action_index=index)
                    progress['last_observation'] = snapshot.metadata
                    save_progress(progress_path, progress)
                    fresh(snapshot, progress['window_identity'], wall_clock)
                    if 'click' in action:
                        adapter.click(*action['click'], snapshot)
                    elif 'key' in action:
                        adapter.key(action['key'], snapshot)
                    else:
                        adapter.text(action['text'], snapshot)
                    # Rebind after each input; even a field-fill group must stop
                    # if focus was taken by another window or geometry changed.
                    refreshed = capture()
                    progress['last_observation'] = refreshed.metadata
                    save_progress(progress_path, progress)
                if step['actions']:
                    park(refreshed)
                snapshot, match = observe(step['after'], deadline)
                progress['last_observation'] = snapshot.metadata
                progress['last_match'] = match
            completed.append(step['name'])
            progress['pending'] = None
            save_progress(progress_path, progress)
        if not advanced:
            # A completed progress file is not evidence that the live client is
            # still in that state. Resume only after observing the last result.
            snapshot, match = observe(steps[-1]['after'], clock()+steps[-1]['timeout'])
            progress['last_observation'], progress['last_match'] = snapshot.metadata, match
        progress.update(state='complete', current_step=None)
    except InterruptedError as exc:
        progress.update(state='stopped', error=str(exc))
    except Exception as exc:
        progress.update(state='needs_attention', error=f'{type(exc).__name__}: {exc}')
    save_progress(progress_path, progress)
    return {**progress, 'progress_path': str(progress_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe', type=Path)
    parser.add_argument('--hwnd', type=int, required=True)
    parser.add_argument('--logdir', type=Path, required=True)
    parser.add_argument('--stopfile', type=Path, required=True)
    parser.add_argument('--phase', choices=('pre_python', 'post_python'), default='pre_python')
    args = parser.parse_args()
    adapter = DesktopAdapter(args.hwnd, args.logdir / 'captures')
    result = run_steps(args.recipe, adapter, args.logdir, args.stopfile, phase=args.phase)
    print(json.dumps(result, ensure_ascii=True))
    return 0 if result['state'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
