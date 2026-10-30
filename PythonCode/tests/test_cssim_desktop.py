import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from cssim_desktop import Desktop, NativeWindows, validate_observation


class FakeNative:
    def __init__(self):
        self.window = dict(hwnd=123, pid=456, exe_path=r'C:\CSSIM\CSSIMClient-Win64-Shipping.exe',
                           title='CSSIM', visible=True, client_rect=[160, 70, 1760, 970])
        self.actions = []
        self.foreground = 123

    def describe(self, hwnd):
        assert hwnd == 123
        return dict(self.window)

    def focus(self, hwnd):
        self.foreground = hwnd

    def ensure_foreground(self, hwnd):
        if self.foreground != hwnd:
            raise RuntimeError('No longer foreground')

    def grab(self, rect):
        class Image:
            def save(self, path, format):
                assert format == 'PNG'
                Path(path).write_bytes(b'fake captured pixels')
        return Image()

    def click_screen(self, hwnd, x, y):
        self.actions.append(('click', hwnd, x, y))

    def move_screen(self, hwnd, x, y):
        self.actions.append(('move', hwnd, x, y))

    def type_text(self, hwnd, text):
        self.actions.append(('type', hwnd, text))

    def key_chord(self, hwnd, key):
        self.actions.append(('key', hwnd, key))


@pytest.fixture
def observed(tmp_path):
    native = FakeNative()
    desktop = Desktop(native)
    metadata = desktop.capture(123, tmp_path / 'capture.png')
    return desktop, native, metadata, tmp_path / 'capture.json'


def test_capture_saves_bound_evidence_and_click_uses_client_coordinates(observed):
    desktop, native, metadata, path = observed
    assert json.loads(path.read_text(encoding='utf-8')) == metadata
    desktop.click(path, 800, 584)
    assert native.actions == [('click', 123, 960, 654)]


@pytest.mark.parametrize('change', [
    {'pid': 999}, {'hwnd': 789}, {'client_rect': [0, 0, 1600, 900]},
    {'exe_path': r'D:\Different\CSSIMClient-Win64-Shipping.exe'},
    {'exe_path': r'C:\Windows\System32\cmd.exe'}, {'visible': False},
])
def test_changed_window_identity_refuses_input(observed, change):
    desktop, native, metadata, _ = observed
    native.window.update(change)
    with pytest.raises(ValueError):
        desktop.click(metadata, 1, 1)
    assert not native.actions


@pytest.mark.parametrize('age', [121, -1])
def test_stale_or_future_evidence_is_rejected(observed, age):
    _, native, metadata, _ = observed
    with pytest.raises(ValueError, match='expired or from the future'):
        validate_observation(metadata, native.window, now=metadata['captured_at_unix'] + age)


def test_modified_screenshot_refuses_input(observed):
    desktop, native, metadata, _ = observed
    Path(metadata['image_path']).write_bytes(b'replaced')
    with pytest.raises(ValueError, match='evidence changed'):
        desktop.click(metadata, 1, 1)
    assert not native.actions


@pytest.mark.parametrize('point', [(-1, 0), (1600, 0), (0, 900), (True, 1), (1.5, 3)])
def test_click_outside_observed_client_is_rejected(observed, point):
    desktop, native, metadata, _ = observed
    with pytest.raises(ValueError, match='inside the observed client area'):
        desktop.click(metadata, *point)
    assert not native.actions


def test_typing_and_key_whitelist(observed):
    desktop, native, metadata, _ = observed
    desktop.type(metadata, '比赛想定10')
    desktop.key(metadata, 'Ctrl+A')
    with pytest.raises(ValueError):
        desktop.type(metadata, 'do something\n')
    with pytest.raises(ValueError):
        desktop.key(metadata, 'Ctrl+Shift+Escape')
    assert native.actions == [('type', 123, '比赛想定10'), ('key', 123, 'Ctrl+A')]


def test_pointer_parking_is_bound_to_fresh_evidence(observed):
    desktop, native, metadata, _ = observed
    desktop.move(metadata, 1500, 700)
    assert native.actions == [('move', 123, 1660, 770)]
    with pytest.raises(ValueError):
        desktop.move(metadata, 1600, 0)
    native.window['pid'] = 999
    with pytest.raises(ValueError):
        desktop.move(metadata, 1500, 700)
    assert len(native.actions) == 1


def test_already_foreground_window_is_not_reactivated():
    class Foreground:
        def GetForegroundWindow(self):
            return 123

        def IsIconic(self, hwnd):
            assert hwnd == 123
            return False

        def SetForegroundWindow(self, hwnd):
            raise AssertionError('Reactivation would reset an in-progress edit')

    native = object.__new__(NativeWindows)
    native.user = Foreground()
    native.focus(123)


@pytest.mark.parametrize('executable', ['Server.exe', 'pycharm64.exe', 'cmd.exe'])
def test_nonclient_terminal_or_ide_entry_refused(observed, executable):
    desktop, native, metadata, _ = observed
    metadata['exe_path'] = native.window['exe_path'] = 'C:\\Applications\\' + executable
    with pytest.raises(ValueError):
        desktop.type(metadata, 'command')
    with pytest.raises(ValueError):
        desktop.key(metadata, 'Enter')
    assert not native.actions


def test_capture_rejects_other_app_and_foreground_loss(tmp_path):
    native = FakeNative()
    desktop = Desktop(native)
    native.window['exe_path'] = r'C:\Windows\System32\cmd.exe'
    with pytest.raises(ValueError, match='Only CSSIM'):
        desktop.capture(123, tmp_path / 'bad.png')
    assert not (tmp_path / 'bad.png').exists()
    native.window['exe_path'] = r'C:\CSSIM\CSSIMClient-Win64-Shipping.exe'
    grab = native.grab

    def switch_window(rect):
        image = grab(rect)
        native.foreground = 999
        return image

    native.grab = switch_window
    with pytest.raises(RuntimeError, match='foreground'):
        desktop.capture(123, tmp_path / 'bad.png')
    assert not (tmp_path / 'bad.png').exists()
