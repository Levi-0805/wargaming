"""Observed, single-action Windows UI access for the local CSSIM workflow.

Screenshots use the visible desktop (Pillow ImageGrab), not WGC. Every input
requires a recent, intact screenshot and the same window/process/client bounds.
No terminal typing, arbitrary key chords, or general process control is exposed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import ntpath
import os
from pathlib import Path
import time

ALLOWED_EXES = frozenset({'cssimclient-win64-shipping.exe', 'server.exe', 'pycharm64.exe'})
CLIENT_EXE = 'cssimclient-win64-shipping.exe'
KEYS = ('Ctrl+A', 'Enter', 'Tab', 'Escape', 'Alt+F4')
MAX_OBSERVATION_AGE = 120


def exe_name(path):
    return ntpath.basename(str(path)).casefold()


def require_allowed(window):
    if exe_name(window.get('exe_path', '')) not in ALLOWED_EXES:
        raise ValueError('Only CSSIM Client, CSSIM Server, and PyCharm windows are allowed')


def read_observation(value):
    if isinstance(value, dict):
        return dict(value)
    return json.loads(Path(value).read_text(encoding='utf-8'))


def validate_observation(observation, current, *, now=None):
    require_allowed(observation)
    require_allowed(current)
    now = time.time() if now is None else now
    age = now - float(observation['captured_at_unix'])
    if not 0 <= age <= MAX_OBSERVATION_AGE:
        raise ValueError('Screenshot is expired or from the future; capture again')
    for field in ('hwnd', 'pid', 'client_rect'):
        if observation.get(field) != current.get(field):
            raise ValueError(f'Window {field} changed; capture again')
    if ntpath.normcase(observation['exe_path']) != ntpath.normcase(current['exe_path']):
        raise ValueError('Window executable changed; capture again')
    if not current.get('visible', False):
        raise ValueError('Target window is not visible')
    image = Path(observation['image_path'])
    if not image.is_absolute() or not image.is_file():
        raise ValueError('Screenshot evidence is missing')
    if hashlib.sha256(image.read_bytes()).hexdigest() != observation['image_sha256']:
        raise ValueError('Screenshot evidence changed; capture again')


class Desktop:
    def __init__(self, backend=None):
        self.backend = backend if backend is not None else NativeWindows()

    def windows(self):
        return self.backend.windows()

    def capture(self, hwnd, output):
        window = self.backend.describe(hwnd)
        require_allowed(window)
        self.backend.focus(hwnd)
        window = self.backend.describe(hwnd)
        require_allowed(window)
        if not window['visible']:
            raise ValueError('Target window is not visible')
        left, top, right, bottom = window['client_rect']
        if right <= left or bottom <= top:
            raise ValueError('Window has no visible client area')
        picture = self.backend.grab(window['client_rect'])
        self.backend.ensure_foreground(hwnd)
        # Refuse evidence captured while the window moved or was replaced.
        after = self.backend.describe(hwnd)
        if any(after[k] != window[k] for k in ('hwnd', 'pid', 'exe_path', 'client_rect')):
            raise ValueError('Window changed during capture; capture again')
        output = Path(output).resolve()
        if output.suffix.lower() != '.png':
            raise ValueError('Screenshot output must be PNG')
        output.parent.mkdir(parents=True, exist_ok=True)
        picture.save(output, format='PNG')
        observed = time.time()
        metadata = dict(window, schema_version=1, captured_at_unix=observed,
                        captured_at_utc=datetime.fromtimestamp(observed, timezone.utc).isoformat(),
                        image_path=str(output), image_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        meta_path = output.with_suffix('.json')
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return metadata

    def _action_window(self, observation):
        observed = read_observation(observation)
        current = self.backend.describe(observed['hwnd'])
        validate_observation(observed, current)
        self.backend.focus(observed['hwnd'])
        current = self.backend.describe(observed['hwnd'])
        validate_observation(observed, current)
        return observed, current

    def click(self, observation, x, y):
        observed, current = self._action_window(observation)
        left, top, right, bottom = current['client_rect']
        if type(x) is not int or type(y) is not int or not (0 <= x < right-left and 0 <= y < bottom-top):
            raise ValueError('Click coordinates must be integer pixels inside the observed client area')
        self.backend.click_screen(current['hwnd'], left+x, top+y)
        return dict(action='click', hwnd=observed['hwnd'], x=x, y=y)

    def type(self, observation, text):
        observed = read_observation(observation)
        if exe_name(observed.get('exe_path', '')) != CLIENT_EXE:
            raise ValueError('Text entry is restricted to CSSIM Client; terminal/IDE typing is disabled')
        if not isinstance(text, str) or len(text) > 4096 or any(ord(c) < 32 or ord(c) == 127 for c in text):
            raise ValueError('Text must be at most 4096 characters without control characters')
        _, current = self._action_window(observed)
        self.backend.type_text(current['hwnd'], text)
        return dict(action='type', hwnd=current['hwnd'], characters=len(text))

    def move(self, observation, x, y):
        """Park the pointer away from a control without sending a button event."""
        observed, current = self._action_window(observation)
        left, top, right, bottom = current['client_rect']
        if type(x) is not int or type(y) is not int or not (0 <= x < right-left and 0 <= y < bottom-top):
            raise ValueError('Pointer coordinates must be inside the observed client area')
        self.backend.move_screen(current['hwnd'], left+x, top+y)
        return dict(action='move', hwnd=observed['hwnd'], x=x, y=y)

    def key(self, observation, key):
        if key not in KEYS:
            raise ValueError('Unsupported key chord')
        observed = read_observation(observation)
        if key not in ('Escape', 'Alt+F4') and exe_name(observed.get('exe_path', '')) != CLIENT_EXE:
            raise ValueError('Editing keys are restricted to CSSIM Client; terminal/IDE entry is disabled')
        _, current = self._action_window(observed)
        self.backend.key_chord(current['hwnd'], key)
        return dict(action='key', hwnd=current['hwnd'], key=key)


class NativeWindows:
    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('Native desktop access is available only on Windows')
        import ctypes as c
        from ctypes import wintypes as w

        self.c, self.w = c, w
        self.user = c.WinDLL('user32', use_last_error=True)
        self.kernel = c.WinDLL('kernel32', use_last_error=True)
        u, k = self.user, self.kernel
        self.callback = c.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)

        class MOUSEINPUT(c.Structure):
            _fields_ = [('dx', w.LONG), ('dy', w.LONG), ('mouseData', w.DWORD),
                        ('dwFlags', w.DWORD), ('time', w.DWORD), ('dwExtraInfo', c.c_size_t)]

        class KEYBDINPUT(c.Structure):
            _fields_ = [('wVk', w.WORD), ('wScan', w.WORD), ('dwFlags', w.DWORD),
                        ('time', w.DWORD), ('dwExtraInfo', c.c_size_t)]

        class INPUTUNION(c.Union):
            _fields_ = [('mi', MOUSEINPUT), ('ki', KEYBDINPUT)]

        class INPUT(c.Structure):
            _anonymous_ = ('data',)
            _fields_ = [('type', w.DWORD), ('data', INPUTUNION)]

        self.INPUT = INPUT
        signatures = [
            (u.EnumWindows, [self.callback, w.LPARAM], w.BOOL),
            (u.IsWindow, [w.HWND], w.BOOL), (u.IsWindowVisible, [w.HWND], w.BOOL),
            (u.IsIconic, [w.HWND], w.BOOL),
            (u.GetWindowThreadProcessId, [w.HWND, c.POINTER(w.DWORD)], w.DWORD),
            (u.GetClientRect, [w.HWND, c.POINTER(w.RECT)], w.BOOL),
            (u.GetWindowRect, [w.HWND, c.POINTER(w.RECT)], w.BOOL),
            (u.WindowFromPoint, [w.POINT], w.HWND),
            (u.GetAncestor, [w.HWND, w.UINT], w.HWND),
            (u.SendMessageTimeoutW, [w.HWND, w.UINT, w.WPARAM, w.LPARAM,
                                     w.UINT, w.UINT, c.POINTER(c.c_size_t)], w.LPARAM),
            (u.ClientToScreen, [w.HWND, c.POINTER(w.POINT)], w.BOOL),
            (u.GetWindowTextLengthW, [w.HWND], c.c_int),
            (u.GetWindowTextW, [w.HWND, w.LPWSTR, c.c_int], c.c_int),
            (u.ShowWindow, [w.HWND, c.c_int], w.BOOL),
            (u.SetForegroundWindow, [w.HWND], w.BOOL),
            (u.BringWindowToTop, [w.HWND], w.BOOL),
            (u.AttachThreadInput, [w.DWORD, w.DWORD, w.BOOL], w.BOOL),
            (u.GetForegroundWindow, [], w.HWND),
            (u.SetCursorPos, [c.c_int, c.c_int], w.BOOL),
            (u.SendInput, [w.UINT, c.POINTER(INPUT), c.c_int], w.UINT),
            (k.OpenProcess, [w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            (k.QueryFullProcessImageNameW, [w.HANDLE, w.DWORD, w.LPWSTR, c.POINTER(w.DWORD)], w.BOOL),
            (k.CloseHandle, [w.HANDLE], w.BOOL),
            (k.GetCurrentThreadId, [], w.DWORD),
        ]
        for function, arguments, result in signatures:
            function.argtypes, function.restype = arguments, result
        u.SetProcessDPIAware()

    def describe(self, hwnd):
        c, w, u, k = self.c, self.w, self.user, self.kernel
        if type(hwnd) is not int or hwnd <= 0 or not u.IsWindow(hwnd):
            raise ValueError('Window no longer exists')
        pid = w.DWORD()
        u.GetWindowThreadProcessId(hwnd, c.byref(pid))
        process = k.OpenProcess(0x1000, False, pid.value)
        if not process:
            raise c.WinError(c.get_last_error())
        try:
            size = w.DWORD(32768)
            image = c.create_unicode_buffer(size.value)
            if not k.QueryFullProcessImageNameW(process, 0, image, c.byref(size)):
                raise c.WinError(c.get_last_error())
            executable = image.value
        finally:
            k.CloseHandle(process)
        rect = w.RECT()
        if not u.GetClientRect(hwnd, c.byref(rect)):
            raise c.WinError(c.get_last_error())
        origin, end = w.POINT(rect.left, rect.top), w.POINT(rect.right, rect.bottom)
        if not u.ClientToScreen(hwnd, c.byref(origin)) or not u.ClientToScreen(hwnd, c.byref(end)):
            raise c.WinError(c.get_last_error())
        title = c.create_unicode_buffer(u.GetWindowTextLengthW(hwnd) + 1)
        u.GetWindowTextW(hwnd, title, len(title))
        return dict(hwnd=hwnd, pid=pid.value, exe_path=executable, title=title.value,
                    visible=bool(u.IsWindowVisible(hwnd)),
                    client_rect=[origin.x, origin.y, end.x, end.y])

    def windows(self):
        results = []

        def visit(hwnd, _):
            if self.user.IsWindowVisible(hwnd):
                try:
                    info = self.describe(int(hwnd))
                    if exe_name(info['exe_path']) in ALLOWED_EXES:
                        results.append(info)
                except (OSError, ValueError):
                    pass
            return True

        callback = self.callback(visit)
        if not self.user.EnumWindows(callback, 0):
            raise self.c.WinError(self.c.get_last_error())
        return results

    def focus(self, hwnd):
        if self.user.GetForegroundWindow() == hwnd and not self.user.IsIconic(hwnd):
            return  # Re-activating a focused Unreal window can reset widget focus.
        if self.user.IsIconic(hwnd):
            self.user.ShowWindow(hwnd, 9)
        self.user.SetForegroundWindow(hwnd)
        if self.user.GetForegroundWindow() != hwnd:
            current_thread = self.kernel.GetCurrentThreadId()
            foreground_thread = self.user.GetWindowThreadProcessId(self.user.GetForegroundWindow(), None)
            attached = bool(foreground_thread and foreground_thread != current_thread and
                            self.user.AttachThreadInput(current_thread, foreground_thread, True))
            try:
                self.user.BringWindowToTop(hwnd)
                self.user.SetForegroundWindow(hwnd)
            finally:
                if attached:
                    self.user.AttachThreadInput(current_thread, foreground_thread, False)
        time.sleep(.15)
        if self.user.GetForegroundWindow() != hwnd:
            self._activate_visible_caption(hwnd)
        self.ensure_foreground(hwnd)

    def _activate_visible_caption(self, hwnd):
        """Windows may deny foreground stealing; activate only a verified title bar."""
        c, w, u = self.c, self.w, self.user
        window = self.describe(hwnd)
        require_allowed(window)
        rect = w.RECT()
        if not u.GetWindowRect(hwnd, c.byref(rect)):
            return
        left, top, right, _ = window['client_rect']
        x, y = (left+right)//2, top-12
        if not rect.top < y < top:
            return
        point = w.POINT(x, y)
        if u.GetAncestor(u.WindowFromPoint(point), 2) != hwnd:
            return  # Covered by another app: never click through it.
        hit = c.c_size_t()
        packed = ((y & 0xffff) << 16) | (x & 0xffff)
        if not u.SendMessageTimeoutW(hwnd, 0x84, 0, packed, 2, 500, c.byref(hit)) or hit.value != 2:
            return  # HTCAPTION only, excluding close/minimize/menu controls.
        if not u.SetCursorPos(x, y) or u.GetAncestor(u.WindowFromPoint(point), 2) != hwnd:
            return
        down, up = self.INPUT(type=0), self.INPUT(type=0)
        down.mi.dwFlags, up.mi.dwFlags = 2, 4
        try:
            self._send([down]); time.sleep(.08)
        finally:
            self._send([up])
        time.sleep(.15)

    def ensure_foreground(self, hwnd):
        if self.user.GetForegroundWindow() != hwnd:
            raise RuntimeError('Target did not become foreground; no input sent')

    @staticmethod
    def grab(rect):
        from PIL import ImageGrab
        return ImageGrab.grab(bbox=tuple(rect), all_screens=True)

    def _send(self, inputs):
        array = (self.INPUT * len(inputs))(*inputs)
        if self.user.SendInput(len(array), array, self.c.sizeof(self.INPUT)) != len(array):
            raise RuntimeError('Windows rejected some input events')

    def click_screen(self, hwnd, x, y):
        self.ensure_foreground(hwnd)
        if not self.user.SetCursorPos(x, y):
            raise self.c.WinError(self.c.get_last_error())
        # Unreal updates hover/hit testing on its render tick. Let it observe
        # the new pointer location before button-down and before parking away.
        time.sleep(.15)
        self.ensure_foreground(hwnd)
        down, up = self.INPUT(type=0), self.INPUT(type=0)
        down.mi.dwFlags, up.mi.dwFlags = 0x0002, 0x0004
        try:
            self._send([down])
            time.sleep(.12)
        finally:
            self._send([up])
        time.sleep(.15)

    def move_screen(self, hwnd, x, y):
        self.ensure_foreground(hwnd)
        if not self.user.SetCursorPos(x, y):
            raise self.c.WinError(self.c.get_last_error())
        self.ensure_foreground(hwnd)

    def type_text(self, hwnd, text):
        self.ensure_foreground(hwnd)
        units = text.encode('utf-16-le')
        inputs = []
        for index in range(0, len(units), 2):
            scan = int.from_bytes(units[index:index+2], 'little')
            for flags in (0x0004, 0x0004 | 0x0002):
                event = self.INPUT(type=1)
                event.ki.wScan, event.ki.dwFlags = scan, flags
                inputs.append(event)
        if inputs:
            self._send(inputs)

    def key_chord(self, hwnd, key):
        self.ensure_foreground(hwnd)
        codes = {'Ctrl+A': (0x11, 0x41), 'Enter': (0x0D,), 'Tab': (0x09,),
                 'Escape': (0x1B,), 'Alt+F4': (0x12, 0x73)}[key]
        downs, ups = [], []
        for code in codes:
            event = self.INPUT(type=1)
            event.ki.wVk = code
            downs.append(event)
        for code in reversed(codes):
            event = self.INPUT(type=1)
            event.ki.wVk, event.ki.dwFlags = code, 0x0002
            ups.append(event)
        try:
            self._send(downs)
            time.sleep(.08)
        finally:
            self._send(ups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('windows')
    capture = commands.add_parser('capture')
    capture.add_argument('--hwnd', type=lambda value: int(value, 0), required=True)
    capture.add_argument('--out', type=Path, required=True)
    for name in ('click', 'move', 'type', 'key'):
        command = commands.add_parser(name)
        command.add_argument('--observation', type=Path, required=True)
        if name in ('click', 'move'):
            command.add_argument('--x', type=int, required=True)
            command.add_argument('--y', type=int, required=True)
        elif name == 'type':
            command.add_argument('--text', required=True)
        else:
            command.add_argument('--key', choices=KEYS, required=True)
    args = parser.parse_args()
    try:
        desktop = Desktop()
        if args.command == 'windows':
            result = desktop.windows()
        elif args.command == 'capture':
            result = desktop.capture(args.hwnd, args.out)
        elif args.command == 'click':
            result = desktop.click(args.observation, args.x, args.y)
        elif args.command == 'move':
            result = desktop.move(args.observation, args.x, args.y)
        elif args.command == 'type':
            result = desktop.type(args.observation, args.text)
        else:
            result = desktop.key(args.observation, args.key)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({'state': 'error', 'error': str(error)}, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
