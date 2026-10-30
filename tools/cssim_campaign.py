"""Repeated, version-bound CSSIM collection with owned-process recovery.

Default: keep collecting ten-game batches with the published model. Optional
candidate training never promotes weights. An optional iteration gate waits for
an external review/publish step. Only each successful result report is committed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import ntpath
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import batch_training as batches
from autotrain_runtime import SessionLock
from prepare_training_config import prepare as prepare_config

ROOT = Path(__file__).resolve().parents[1]


class CampaignStop(RuntimeError):
    pass


class CampaignBlocked(RuntimeError):
    pass


def normalized(path):
    return ntpath.normcase(str(path)).replace('/', '\\')


def checked_git(repo, *arguments, env=None):
    result = subprocess.run(['git', '-C', str(repo), *arguments], check=True, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
    return result.stdout.decode('utf-8').strip()


def same_process(first, second):
    return (first['pid'] == second['pid'] and first['created_ticks'] == second['created_ticks']
            and normalized(first.get('exe_path') or '') == normalized(second.get('exe_path') or ''))


def descendant_candidates(rows, owners, allowed_paths):
    """Claim only new descendants of recorded identities with exact paths."""
    allowed = {normalized(p) for p in allowed_paths}
    known = {row['pid']: row for row in owners}
    live = {row['pid']: row for row in rows}
    claimed = []
    changed = True
    while changed:
        changed = False
        for row in rows:
            parent = known.get(row.get('ppid'))
            if row['pid'] in known or parent is None or normalized(row.get('exe_path') or '') not in allowed:
                continue
            live_parent = live.get(parent['pid'])
            if (live_parent and not same_process(parent, live_parent)) or row['created_ticks'] < parent['created_ticks']:
                continue
            claimed.append(row)
            known[row['pid']] = row
            changed = True
    return claimed


def exact_platform_command(argv, main, config, cwd):
    """Accept the observed plain main.py invocation, never arbitrary Python."""
    if not argv or ntpath.basename(argv[0]).lower() not in ('python.exe', 'pythonw.exe'):
        return False
    arguments = argv[1:]
    if len(arguments) not in (3, 4) or arguments[1] not in ('-c', '--config'):
        return False
    if len(arguments) == 4 and arguments[3] != '-s':
        return False

    def absolute(value):
        return normalized(ntpath.normpath(value if ntpath.isabs(value) else ntpath.join(str(cwd), value)))

    return absolute(arguments[0]) == normalized(main) and absolute(arguments[2]) == normalized(config)


def verified_ide_workdir(project, argv):
    """Bind relative default-config arguments to an exact saved IDE run config."""
    if len(argv) < 4 or not ntpath.isabs(argv[1]):
        return None  # A relative script cannot identify the intended project.
    if ntpath.isabs(argv[3]):
        return dict(source='absolute_main_and_config')
    candidates = [project / '.idea/workspace.xml']
    candidates += list((project / '.idea/runConfigurations').glob('*.xml'))
    for path in candidates:
        if not path.exists():
            continue
        content = path.read_bytes()
        try:
            document = ET.fromstring(content)
        except ET.ParseError:
            continue
        for configuration in document.iter('configuration'):
            if configuration.get('type') != 'PythonConfigurationType':
                continue
            options = {item.get('name'): item.get('value', '') for item in configuration.findall('option')}
            expand = lambda value: normalized(ntpath.normpath(value.replace('$PROJECT_DIR$', str(project))))
            parameter = re.fullmatch(r'(-c|--config)\s+(?:"([^"\r\n]+)"|([^\s"\r\n]+))', options.get('PARAMETERS', ''))
            if (expand(options.get('SCRIPT_NAME', '')) == normalized(project / 'main.py')
                    and expand(options.get('WORKING_DIRECTORY', '')) == normalized(project)
                    and options.get('INTERPRETER_OPTIONS', '') == ''
                    and options.get('MODULE_MODE', 'false') == 'false'
                    and parameter and parameter[1] == argv[2]
                    and normalized(parameter[2] or parameter[3]) == normalized(argv[3])):
                return dict(source='saved_ide_run_configuration', path=str(path),
                            sha256=hashlib.sha256(content).hexdigest(), configuration=configuration.get('name'))
    return None


def ready_runtime(manifest):
    """Require actual run identity plus handshake and reset request before Play."""
    process_path = Path(manifest['runtime_exit_path']).with_name('runtime-process.json')
    if not process_path.exists():
        return None
    process = batches.read_json(process_path)
    matches = []
    for metadata_path in Path(manifest['log_root']).glob('*/run.json'):
        metadata = batches.read_json(metadata_path)
        if metadata.get('process_id') == process['pid']:
            batches.validate_runtime_identity(metadata, manifest)
            matches.append(metadata_path.parent)
    if len(matches) > 1:
        raise CampaignBlocked('Several log directories claim the managed runtime PID')
    if not matches:
        return None
    protocol = matches[0] / 'protocol.jsonl'
    if not protocol.exists():
        return None
    handshake, prepare = False, False
    with protocol.open('rb') as stream:
        # Handshake and the first prepare are at the start, before any frames.
        for line in stream.read(1024 * 1024).splitlines(keepends=True):
            if not line.endswith(b'\n'):
                break
            row = json.loads(line)
            handshake |= row.get('direction') == 'ue_to_python' and row.get('channel') == 'handshake'
            prepare |= (row.get('direction') == 'python_to_ue' and row.get('channel') == 'prepare'
                        and row.get('type') == 'reset_property')
    return dict(pid=process['pid'], directory=str(matches[0])) if handshake and prepare else None


@dataclass
class Settings:
    campaign_id: str
    recipe: Path
    repo: Path = ROOT
    cssim_root: Path = Path('C:/CSSIM')
    server_pid: int | None = None
    gate_after_success: bool = False
    train_candidate: bool = False
    sources: list[Path] = field(default_factory=list)
    max_batches: int = 0
    startup_seconds: float = 300
    timeout_minutes: float = 360
    idle_minutes: float = 10
    minimum_free_gb: float = 10
    retry_seconds: float = 15

    def __post_init__(self):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,39}', self.campaign_id):
            raise ValueError('Campaign ID requires 1..40 letters, digits, underscore or hyphen')
        if min(self.startup_seconds, self.timeout_minutes, self.idle_minutes, self.minimum_free_gb) <= 0:
            raise ValueError('Timeout and disk limits must be positive')
        if self.max_batches < 0 or self.retry_seconds < 0:
            raise ValueError('Batch limit and retry delay cannot be negative')
        if self.server_pid is not None and self.server_pid <= 0:
            raise ValueError('Borrowed manager PID must be positive')
        self.repo, self.cssim_root, self.recipe = (Path(p).resolve() for p in (self.repo, self.cssim_root, self.recipe))


class WindowsProcesses:
    """Read process ancestry and operate only on a rechecked creation identity."""
    NAMES = {'server.exe', 'cssimclient-win64-shipping.exe', 'cssimserver-win64-shipping.exe',
             'pycharm64.exe', 'python.exe', 'pythonw.exe'}

    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('The real campaign requires Windows')
        import ctypes as c
        from ctypes import wintypes as w
        self.c, self.w = c, w
        self.k = c.WinDLL('kernel32', use_last_error=True)
        self.shell = c.WinDLL('shell32', use_last_error=True)

        class Entry(c.Structure):
            _fields_ = [('dwSize', w.DWORD), ('cntUsage', w.DWORD), ('th32ProcessID', w.DWORD),
                        ('th32DefaultHeapID', c.c_size_t), ('th32ModuleID', w.DWORD), ('cntThreads', w.DWORD),
                        ('th32ParentProcessID', w.DWORD), ('pcPriClassBase', w.LONG), ('dwFlags', w.DWORD),
                        ('szExeFile', w.WCHAR * 260)]
        self.Entry = Entry
        signatures = [
            (self.k.OpenProcess, [w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            (self.k.CloseHandle, [w.HANDLE], w.BOOL),
            (self.k.QueryFullProcessImageNameW, [w.HANDLE, w.DWORD, w.LPWSTR, c.POINTER(w.DWORD)], w.BOOL),
            (self.k.GetProcessTimes, [w.HANDLE] + [c.POINTER(w.FILETIME)] * 4, w.BOOL),
            (self.k.CreateToolhelp32Snapshot, [w.DWORD, w.DWORD], w.HANDLE),
            (self.k.Process32FirstW, [w.HANDLE, c.POINTER(Entry)], w.BOOL),
            (self.k.Process32NextW, [w.HANDLE, c.POINTER(Entry)], w.BOOL),
            (self.k.TerminateProcess, [w.HANDLE, w.UINT], w.BOOL),
            (self.k.WaitForSingleObject, [w.HANDLE, w.DWORD], w.DWORD),
            (self.shell.CommandLineToArgvW, [w.LPCWSTR, c.POINTER(c.c_int)], c.POINTER(w.LPWSTR)),
            (self.k.LocalFree, [w.HLOCAL], w.HLOCAL),
        ]
        for function, args, result in signatures:
            function.argtypes, function.restype = args, result

    def identity(self, pid, *, handle=None):
        c, w, k = self.c, self.w, self.k
        owned_handle = handle is None
        handle = handle or k.OpenProcess(0x1000, False, pid)
        if not handle:
            raise c.WinError(c.get_last_error())
        try:
            times = [w.FILETIME() for _ in range(4)]
            if not k.GetProcessTimes(handle, *(c.byref(value) for value in times)):
                raise c.WinError(c.get_last_error())
            created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            size = w.DWORD(32768)
            buffer = c.create_unicode_buffer(size.value)
            path = buffer.value if k.QueryFullProcessImageNameW(handle, 0, buffer, c.byref(size)) else None
            return dict(pid=int(pid), created_ticks=int(created), exe_path=path)
        finally:
            if owned_handle:
                k.CloseHandle(handle)

    def snapshot(self):
        c, k = self.c, self.k
        snapshot = k.CreateToolhelp32Snapshot(2, 0)
        if snapshot == c.c_void_p(-1).value:
            raise c.WinError(c.get_last_error())
        rows = []
        try:
            entry = self.Entry(); entry.dwSize = c.sizeof(entry)
            valid = k.Process32FirstW(snapshot, c.byref(entry))
            while valid:
                name = entry.szExeFile
                if name.casefold() in self.NAMES:
                    try:
                        item = self.identity(entry.th32ProcessID)
                        rows.append(dict(item, ppid=int(entry.th32ParentProcessID), name=name))
                    except OSError:
                        # Never claim or kill a process lacking creation evidence.
                        rows.append(dict(pid=int(entry.th32ProcessID), ppid=int(entry.th32ParentProcessID),
                                         name=name, exe_path=None, created_ticks=None))
                valid = k.Process32NextW(snapshot, c.byref(entry))
        finally:
            k.CloseHandle(snapshot)
        return rows

    def command_line(self, pid):
        # Query one already identified PID. No command from the target is run.
        script = f"(Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}').CommandLine"
        text = subprocess.check_output(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
                                       encoding='utf-8', errors='replace', timeout=30,
                                       creationflags=subprocess.CREATE_NO_WINDOW).strip()
        count = self.c.c_int()
        pointer = self.shell.CommandLineToArgvW(text, self.c.byref(count))
        if not pointer:
            return []
        try:
            return [pointer[i] for i in range(count.value)]
        finally:
            self.k.LocalFree(pointer)

    def stop(self, expected):
        if expected.get('created_ticks') is None or not expected.get('exe_path'):
            raise CampaignBlocked('Cannot stop a process without verified path and creation identity')
        handle = self.k.OpenProcess(0x1000 | 0x100000 | 1, False, expected['pid'])
        if not handle:
            current = next((p for p in self.snapshot() if p['pid'] == expected['pid']), None)
            if current is None:
                return
            raise CampaignBlocked(f'Cannot stop owned PID {expected["pid"]}; Windows denied access')
        try:
            if not same_process(expected, self.identity(expected['pid'], handle=handle)):
                raise CampaignBlocked('PID was reused; refusing to terminate it')
            if self.k.WaitForSingleObject(handle, 0) == 0:
                return  # A parent may still retain a handle to an exited child.
            if not self.k.TerminateProcess(handle, 2):
                raise self.c.WinError(self.c.get_last_error())
            if self.k.WaitForSingleObject(handle, 15000) != 0:
                raise CampaignBlocked('Owned process did not exit after termination')
        finally:
            self.k.CloseHandle(handle)


class RealBackend:
    def __init__(self, settings):
        self.s = settings
        from cssim_desktop import Desktop
        self.desktop, self.processes = Desktop(), WindowsProcesses()
        root = self.s.cssim_root
        self.paths = dict(manager=root / 'Server/Server/Server.exe',
                          client=root / 'Client/CSSIM/Binaries/Win64/CSSIMClient-Win64-Shipping.exe',
                          dedicated=root / 'Server/CSSIM/Binaries/Win64/CSSIMServer-Win64-Shipping.exe',
                          python=root / 'Client/Data/AlgData/PythonEnv/python.exe',
                          pycharm=root / 'Client/Data/AlgData/Pycharm_2023.3.5_Protable/bin/pycharm64.exe',
                          server_pycharm=root / 'Server/Data/AlgData/Pycharm_2023.3.5_Protable/bin/pycharm64.exe')
        self.code = root / 'Client/Data/AlgData/PythonCode'
        self.platform_config = self.code / 'TaskConfig/cssim/cssim_25vs25.jsonc'
        self.known_ides = []

    def free_bytes(self, path):
        return shutil.disk_usage(path).free

    def preflight(self):
        from cssim_ui_flow import load_recipe
        recipe, _ = load_recipe(self.s.recipe)
        if {step['phase'] for step in recipe['steps']} != {'pre_python', 'post_python'}:
            raise CampaignBlocked('Recipe must verify both setup and the final Play step')
        relative_recipe = self.s.recipe.relative_to(self.s.repo).as_posix()
        checked_git(self.s.repo, 'ls-files', '--error-unmatch', '--', relative_recipe)
        head = batches.verify_sources(self.s.repo, self.s.cssim_root, clean_all=True)
        remote = checked_git(self.s.repo, 'ls-remote', 'origin', 'refs/heads/main').split()
        if not remote or remote[0] != head:
            raise CampaignBlocked('GitHub main does not match the deployed source commit')
        if checked_git(self.s.repo, 'branch', '--show-current') != 'main':
            raise CampaignBlocked('Campaign requires the published main branch')
        return head

    def _start(self, context, role, command, hidden=True, cwd=None):
        info = subprocess.STARTUPINFO()
        info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        info.wShowWindow = 0 if hidden else 1
        console = (context['directory'] / f'{role}-console.log').open('ab')
        try:
            process = subprocess.Popen([str(x) for x in command], cwd=cwd, stdout=console,
                                       stderr=subprocess.STDOUT, startupinfo=info,
                                       creationflags=subprocess.CREATE_NO_WINDOW if hidden else 0)
        finally:
            console.close()
        owned = dict(role=role, process=process, identity=None)
        context['owned'].append(owned)  # Cleanup is installed before querying child details.
        owned['identity'] = self.processes.identity(process.pid)
        if normalized(owned['identity']['exe_path']) != normalized(command[0]):
            raise CampaignBlocked(f'Launched {role} executable differs from the exact configured path')
        self._save_processes(context)
        return owned

    def _save_processes(self, context):
        batches.write_json(context['directory'] / 'processes.json',
                           dict(owned=[dict(role=p['role'], **(p['identity'] or {'pid': p['process'].pid}))
                                       for p in context['owned']], borrowed_manager=context.get('borrowed_manager'),
                                observed_dedicated=context.get('observed_dedicated', []),
                                observed_project_ides=self.known_ides))

    def refresh(self, context):
        rows = self.processes.snapshot()
        owners = [p['identity'] for p in context['owned'] if p['identity']]
        owners += self.known_ides
        if context.get('borrowed_manager'):
            borrowed = context['borrowed_manager']
            current = next((p for p in rows if p['pid'] == borrowed['pid']), None)
            if current is None or not same_process(borrowed, current):
                raise CampaignBlocked('Borrowed manager exited or its identity changed')
            owners.append(borrowed)
        allowed = [self.paths[k] for k in ('dedicated', 'python', 'pycharm', 'server_pycharm')]
        for item in descendant_candidates(rows, owners, allowed):
            if item['pid'] in context['baseline_pids']:
                continue
            if item['created_ticks'] < context['started_ticks']:
                continue
            if normalized(item['exe_path']) in {normalized(self.paths['pycharm']), normalized(self.paths['server_pycharm'])}:
                if not any(same_process(item, known) for known in self.known_ides):
                    self.known_ides.append(item)
                continue
            if context.get('borrowed_manager') and normalized(item['exe_path']) == normalized(self.paths['dedicated']):
                observed = context.setdefault('observed_dedicated', [])
                if not any(same_process(item, prior) for prior in observed):
                    observed.append(item)
                continue
            if normalized(item['exe_path']) == normalized(self.paths['python']):
                supervisors = {p['identity']['pid'] for p in context['owned']
                               if p['role'] == 'supervisor' and p['identity']}
                if item['ppid'] not in supervisors:
                    # IDE ancestry alone never authorizes terminating arbitrary
                    # user code. Verify exact default main/config before owning it.
                    candidates = context.setdefault('platform_python_candidates', [])
                    if not any(same_process(item, prior) for prior in candidates):
                        candidates.append(item)
                    continue
            if not any(p['identity'] and same_process(item, p['identity']) for p in context['owned']):
                context['owned'].append(dict(role='descendant', process=None, identity=item))
        # Record an elevated DS even if Windows will not disclose its path. It
        # is observed, never force-stopped, and must disappear after client close.
        parent_ids = {p['pid'] for p in owners}
        for item in rows:
            if (item['name'].casefold() == 'cssimserver-win64-shipping.exe'
                    and item['ppid'] in parent_ids and item['pid'] not in context['baseline_pids']):
                observed = context.setdefault('observed_dedicated', [])
                if not any(p['pid'] == item['pid'] for p in observed):
                    observed.append(item)
        self._save_processes(context)
        return rows

    def launch_apps(self, context):
        # Native process creation timestamps use 100 ns since 1601. Baseline
        # exclusion plus this lower bound also protects reused IDE processes.
        context['started_ticks'] = time.time_ns() // 100 + 116444736000000000
        rows = self.processes.snapshot()
        context['baseline_pids'] = {p['pid'] for p in rows}
        blockers = [p for p in rows if p['name'].casefold() in
                    ('server.exe', 'cssimclient-win64-shipping.exe', 'cssimserver-win64-shipping.exe')]
        if self.s.server_pid:
            manager = next((p for p in blockers if p['pid'] == self.s.server_pid and p['name'].casefold() == 'server.exe'), None)
            if manager is None or manager.get('created_ticks') is None:
                raise CampaignBlocked('Explicit borrowed manager PID lacks a stable process identity')
            if manager.get('exe_path') and normalized(manager['exe_path']) != normalized(self.paths['manager']):
                raise CampaignBlocked('Borrowed manager has a different executable path')
            context['borrowed_manager'] = manager
            blockers = [p for p in blockers if p['pid'] != self.s.server_pid]
        if blockers:
            raise CampaignBlocked('Pre-existing CSSIM instance is not owned by this attempt; close it before starting')
        context['config_before'] = self.platform_config.stat().st_mtime_ns if self.platform_config.exists() else None
        if not self.s.server_pid:
            self._start(context, 'manager', [self.paths['manager']], cwd=self.paths['manager'].parent)
            time.sleep(2)
        client = self._start(context, 'client', [self.paths['client']], hidden=False, cwd=self.paths['client'].parent)
        context['client_pid'] = client['process'].pid

    def _wait(self, context, guard, condition, seconds, description):
        deadline = time.monotonic() + seconds
        while True:
            guard()
            self.refresh(context)
            result = condition()
            if result:
                return result
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Timed out: {description}')
            time.sleep(1)

    def wait_client(self, context, guard):
        def found():
            candidates = [w for w in self.desktop.windows() if w['pid'] == context['client_pid']
                          and normalized(w['exe_path']) == normalized(self.paths['client'])
                          and w['client_rect'][2]-w['client_rect'][0] == 1600
                          and w['client_rect'][3]-w['client_rect'][1] == 900]
            if len(candidates) > 1:
                raise CampaignBlocked('Several 1600x900 client windows match this attempt')
            return candidates[0] if candidates else None
        context['window'] = self._wait(context, guard, found, self.s.startup_seconds, '1600x900 client window')

    def flow(self, context, phase, guard):
        from cssim_ui_flow import DesktopAdapter, run_steps
        adapter = DesktopAdapter(context['window']['hwnd'], context['directory'] / 'ui/captures', self.desktop)
        capture = adapter.capture

        def guarded_capture():
            guard()
            self.refresh(context)
            return capture()
        adapter.capture = guarded_capture
        result = run_steps(self.s.recipe, adapter, context['directory'] / 'ui', context['stopfile'], phase=phase)
        if result['state'] != 'complete':
            raise RuntimeError(f'UI {phase}: {result.get("error", result["state"])}')

    def check_platform(self, context, guard):
        def ide_present():
            allowed = {normalized(self.paths['pycharm']), normalized(self.paths['server_pycharm'])}
            candidates = [w for w in self.desktop.windows() if normalized(w['exe_path']) in allowed
                          and re.search(r'(?<!\w)PythonCode(?!\w)', w.get('title', ''), re.IGNORECASE)]
            for window in candidates:
                identity = self.processes.identity(window['pid'])
                if normalized(identity.get('exe_path') or '') != normalized(window['exe_path']):
                    continue
                # An existing project window may be reused by the platform.
                # Binding its ancestry does not authorize killing the IDE, or
                # any old/arbitrary Python child. resolve_auto_python still
                # requires a new process with the exact main/config command.
                if not any(same_process(identity, prior) for prior in self.known_ides):
                    self.known_ides.append(identity)
                return window
            return None
        context['pycharm_window'] = self._wait(context, guard, ide_present, 120, 'PyCharm startup window')
        if not self.platform_config.exists() or self.platform_config.stat().st_mtime_ns == context['config_before']:
            raise RuntimeError('Platform did not write a new task configuration for this attempt')
        context['platform_config'] = str(self.platform_config)
        self._save_processes(context)

    def resolve_auto_python(self, context):
        rows = self.refresh(context)
        expected = normalized(self.paths['python'])
        own = {p['identity']['pid']: p for p in context['owned'] if p['identity']}
        candidates = {p['pid']: p for p in context.get('platform_python_candidates', [])}
        verified = []
        for item in rows:
            if item['pid'] == os.getpid() or normalized(item.get('exe_path') or '') != expected:
                continue
            owned = own.get(item['pid'])
            candidate = candidates.get(item['pid'])
            if owned is None and (candidate is None or not same_process(candidate, item)):
                raise CampaignBlocked(f'Unowned CSSIM Python PID {item["pid"]}; refusing parallel runtimes')
            argv = self.processes.command_line(item['pid'])
            if not exact_platform_command(argv, self.code / 'main.py', self.platform_config, self.code):
                raise CampaignBlocked('Owned Python has an unexpected command; no process was stopped')
            workdir = verified_ide_workdir(self.code, argv)
            if workdir is None:
                raise CampaignBlocked('Cannot verify the default Python project/config working directory')
            verified.append((item, owned, workdir))
        # Validate all conflicts before stopping any verified platform child.
        for item, owned, workdir in verified:
            if owned is None:
                context['owned'].append(dict(role='platform_auto_python', process=None, identity=item))
            self.processes.stop(item)
            context.setdefault('stopped_platform_python', []).append(dict(pid=item['pid'],
                reason='platform started default config before managed runtime', working_directory_evidence=workdir))
        batches.write_json(context['directory'] / 'platform-preparation.json',
                           dict(pycharm_window=context['pycharm_window'], platform_config=context['platform_config'],
                                stopped_platform_python=context.get('stopped_platform_python', [])))

    def bind(self, context):
        # UI preparation can take minutes; recheck remote/deployment rather
        # than silently accepting a new local commit written during setup.
        if self.preflight() != context['source_commit']:
            raise CampaignBlocked('Source commit changed while configuring the batch')
        effective = context['directory'] / 'effective.jsonc'
        prepare_config(self.platform_config, effective, cssim_root=self.s.cssim_root,
                       template=self.s.repo / 'PythonCode/TaskConfig/cssim/iql_eval_red.jsonc')
        context['manifest'] = batches.prepare(self.s.repo, self.s.cssim_root, effective, context['batch_id'])
        return context['manifest']['manifest']

    def start_runtime(self, context):
        command = [self.paths['python'], self.s.repo / 'tools/autotrain_runtime.py', context['manifest']['manifest'],
                   '--timeout-minutes', str(self.s.timeout_minutes), '--idle-minutes', str(self.s.idle_minutes),
                   '--minimum-free-gb', str(self.s.minimum_free_gb)]
        context['supervisor'] = self._start(context, 'supervisor', command, cwd=self.s.repo)

    def wait_ready(self, context, guard):
        def ready():
            process = context['supervisor']['process']
            if process.poll() is not None:
                raise RuntimeError('Runtime supervisor exited before Python was ready')
            return ready_runtime(context['manifest'])
        return self._wait(context, guard, ready, 120, 'Python handshake/reset readiness')

    def wait_result(self, context, guard, progress):
        process = context['supervisor']['process']
        deadline = time.monotonic() + self.s.timeout_minutes * 60 + 60
        while process.poll() is None:
            guard()
            self.refresh(context)
            path = Path(context['manifest']['manifest']).with_name('supervisor-status.json')
            if path.exists():
                progress(batches.read_json(path))
            if time.monotonic() >= deadline:
                raise RuntimeError('Supervisor exceeded the configured batch deadline')
            time.sleep(5)
        if process.returncode != 0:
            receipt = Path(context['manifest']['runtime_exit_path'])
            detail = batches.read_json(receipt).get('stop_reason') if receipt.exists() else 'missing exit receipt'
            raise RuntimeError(f'Runtime exited with {process.returncode}: {detail}')
        return batches.finish(context['manifest']['manifest'])

    def cleanup(self, context):
        errors = []
        if not context['owned'] and not context.get('borrowed_manager'):
            batches.write_json(context['directory'] / 'cleanup.json', dict(at=batches.now(), errors=[]))
            return
        manifest = context.get('manifest')
        if manifest:
            Path(manifest['runtime_exit_path']).with_name('STOP').touch()
        supervisor = context.get('supervisor')
        if supervisor and supervisor['process'].poll() is None:
            try:
                supervisor['process'].wait(timeout=45)
            except subprocess.TimeoutExpired:
                errors.append('Supervisor did not acknowledge STOP within 45 seconds')
        try:
            self.refresh(context)
        except Exception as error:
            errors.append(str(error))
        # Close the client first: the platform normally tears down its elevated
        # dedicated server within seconds after this graceful disconnect.
        clients = [p for p in context['owned'] if p['role'] == 'client']
        for owned in clients:
            try:
                if owned['process'].poll() is None:
                    for window in self.desktop.windows():
                        if window['pid'] == owned['process'].pid and normalized(window['exe_path']) == normalized(self.paths['client']):
                            native = self.desktop.backend
                            native.user.PostMessageW.argtypes = [native.w.HWND, native.w.UINT, native.w.WPARAM, native.w.LPARAM]
                            native.user.PostMessageW.restype = native.w.BOOL
                            native.user.PostMessageW(window['hwnd'], 0x0010, 0, 0)
                    try:
                        owned['process'].wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        owned['process'].terminate()
                        owned['process'].wait(timeout=15)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                errors.append(str(error))
        observed = {p['pid']: p for p in context.get('observed_dedicated', [])}
        # A scene closed before Python connected can take over a minute for
        # the manager to reap. Do not turn its normal grace period into a retry.
        deadline = time.monotonic() + 150
        while observed:
            try:
                rows = {p['pid']: p for p in self.processes.snapshot()}
            except OSError as error:
                errors.append(f'Unable to recheck dedicated server exit: {error}')
                break  # Continue cleaning verified owned root processes.
            observed = {pid: original for pid, original in observed.items()
                        if pid in rows and rows[pid]['created_ticks'] == original['created_ticks']}
            if not observed or time.monotonic() >= deadline:
                break
            time.sleep(1)
        if observed:
            errors.append('Dedicated server remains after client close; no elevated or unverified process was force-stopped')
        for owned in reversed(context['owned']):
            if owned['role'] == 'client':
                continue
            identity = owned['identity']
            if identity and normalized(identity.get('exe_path') or '') == normalized(self.paths['dedicated']):
                continue  # Dedicated server closure is checked above, never force-killed.
            try:
                if owned['process'] is not None:
                    if owned['process'].poll() is None:
                        owned['process'].terminate(); owned['process'].wait(timeout=15)
                elif identity:
                    self.processes.stop(identity)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                errors.append(str(error))
        batches.write_json(context['directory'] / 'cleanup.json', dict(at=batches.now(), errors=errors,
                           borrowed_manager_preserved=context.get('borrowed_manager')))
        if errors:
            raise CampaignBlocked('Cleanup requires attention: ' + '; '.join(errors))

    def commit_report(self, result):
        report = Path(result['result_path']).resolve()
        relative = report.relative_to(self.s.repo).as_posix()
        dirty = subprocess.check_output(['git', '-C', str(self.s.repo), 'status', '--porcelain',
                                         '--untracked-files=all', '-z'], timeout=90).decode('utf-8').split('\0')
        if any(row and row[3:] != relative for row in dirty):
            raise CampaignBlocked('Other uncommitted work exists; result preserved without committing it')
        if checked_git(self.s.repo, 'diff', '--cached', '--name-only'):
            raise CampaignBlocked('Existing staged changes prevent an automatic report-only commit')
        env = os.environ.copy()
        env['GIT_AUTHOR_NAME'] = env['GIT_COMMITTER_NAME'] = checked_git(self.s.repo, 'log', '-1', '--format=%an')
        env['GIT_AUTHOR_EMAIL'] = env['GIT_COMMITTER_EMAIL'] = checked_git(self.s.repo, 'log', '-1', '--format=%ae')
        for args in (['add', '--', relative], ['commit', '-m', f'Record automated ten-game batch {result["batch_id"]}', '--', relative],
                     ['push', 'origin', 'main']):
            checked_git(self.s.repo, *args, env=env)
        head = checked_git(self.s.repo, 'rev-parse', 'HEAD')
        remote = checked_git(self.s.repo, 'ls-remote', 'origin', 'refs/heads/main').split()
        if not remote or remote[0] != head:
            raise CampaignBlocked('Result commit was not verified on GitHub')
        return head

    def candidate(self, context, reports, guard):
        if len(self.s.sources) + len(reports) < 3:
            return dict(state='waiting_for_three_source_sessions')
        candidate_id = context['batch_id'] + '-candidate'
        command = [self.paths['python'], self.s.repo / 'tools/train_batch_candidate.py', '--candidate-id', candidate_id,
                   '--batches', *reports]
        if self.s.sources:
            command += ['--sources', *self.s.sources]
        process = self._start(context, 'candidate_trainer', command, cwd=self.s.repo)['process']
        while process.poll() is None:
            guard()
            time.sleep(5)
        if process.returncode:
            return dict(state='candidate_failed', exit_code=process.returncode)
        return batches.read_json(self.s.repo / 'logs/autotrain-candidates' / candidate_id / 'review.json')


class Campaign:
    def __init__(self, settings, backend=None, *, sleep=time.sleep):
        self.s, self.sleep = settings, sleep
        self.directory = settings.repo / 'logs/campaigns' / settings.campaign_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stopfile, self.statefile = self.directory / 'STOP', self.directory / 'state.json'
        self.backend = backend if backend is not None else RealBackend(settings)
        self.state = dict(campaign_id=settings.campaign_id, created_at=batches.now(), attempts=[], successful_reports=[],
                          consecutive_failures=0, state='created', borrowed_manager_pid=settings.server_pid,
                          gate_after_success=settings.gate_after_success, model_promotion='external_review_only')
        if self.statefile.exists():
            raise CampaignBlocked('Campaign ID already exists; preserve its receipts and choose a new ID')
        self.save()

    def save(self, **values):
        self.state.update(values, updated_at=batches.now())
        batches.write_json(self.statefile, self.state)

    def guard(self):
        if self.stopfile.exists():
            raise CampaignStop('STOP requested')
        if self.backend.free_bytes(self.directory) < self.s.minimum_free_gb * 1024**3:
            raise CampaignStop('Low disk space')

    def phase(self, context, name):
        self.guard()
        self.save(state=name, current_attempt=context['batch_id'])

    def attempt(self, number):
        context = dict(batch_id=f'{self.s.campaign_id}-{number:04d}', directory=self.directory / f'attempt-{number:04d}',
                       owned=[], baseline_pids=set(), stopfile=self.stopfile)
        context['directory'].mkdir()
        record = dict(batch_id=context['batch_id'], started_at=batches.now(), state='starting')
        result, failure = None, None
        try:
            self.phase(context, 'preflight')
            record['source_commit'] = self.backend.preflight()
            context['source_commit'] = record['source_commit']
            self.phase(context, 'launching')
            self.backend.launch_apps(context)
            self.backend.wait_client(context, self.guard)
            self.phase(context, 'configuring_ui')
            self.backend.flow(context, 'pre_python', self.guard)
            self.backend.check_platform(context, self.guard)
            self.backend.resolve_auto_python(context)
            record['manifest_path'] = self.backend.bind(context)
            self.phase(context, 'starting_python')
            self.backend.start_runtime(context)
            record['runtime_ready'] = self.backend.wait_ready(context, self.guard)
            self.backend.flow(context, 'post_python', self.guard)
            self.phase(context, 'collecting')
            result = self.backend.wait_result(context, self.guard, lambda value: self.save(runtime=value))
            record.update(state='complete', result_path=result['result_path'], scores=result.get('scores', []))
        except BaseException as error:
            failure = error
            record.update(state='failed', error=f'{type(error).__name__}: {error}')
        finally:
            try:
                self.backend.cleanup(context)
            except BaseException as error:
                record['cleanup_error'] = f'{type(error).__name__}: {error}'
                failure = CampaignBlocked(str(error))
            record['ended_at'] = batches.now()
            batches.write_json(context['directory'] / 'attempt.json', record)
        if result and failure is None:
            try:
                self.phase(context, 'recording_report')
                record['report_commit'] = self.backend.commit_report(result)
                self.state['successful_reports'].append(result['result_path'])
                self.state['consecutive_failures'] = 0
                if self.s.train_candidate:
                    self.phase(context, 'training_candidate')
                    record['candidate'] = self.backend.candidate(context, self.state['successful_reports'], self.guard)
            except BaseException as error:
                record['postprocessing_error'] = f'{type(error).__name__}: {error}'
                failure = error if isinstance(error, (CampaignStop, KeyboardInterrupt, SystemExit)) else CampaignBlocked(str(error))
            finally:
                if self.s.train_candidate:
                    try:
                        self.backend.cleanup(context)
                    except BaseException as error:
                        record['cleanup_error'] = f'{type(error).__name__}: {error}'
                        failure = CampaignBlocked(str(error))
                batches.write_json(context['directory'] / 'attempt.json', record)
        else:
            self.state['consecutive_failures'] += 1
        self.state['attempts'].append(record)
        self.save()
        if isinstance(failure, (CampaignStop, CampaignBlocked, KeyboardInterrupt, SystemExit)):
            raise failure
        return failure is None

    def gate(self):
        token = f'{self.s.campaign_id}:{len(self.state["attempts"])}'
        self.save(state='waiting_for_iteration', gate_token=token)
        release = self.directory / 'gate-release.json'
        while True:
            self.guard()
            if release.exists():
                value = batches.read_json(release)
                if value.get('gate_token') != token or value.get('campaign_id') != self.s.campaign_id:
                    raise CampaignBlocked('Iteration release does not match the current gate')
                self.backend.preflight()  # New weights must already be pushed and deployed.
                release.rename(self.directory / f'gate-release-{len(self.state["attempts"]):04d}.json')
                self.save(gate_token=None)
                return
            self.sleep(2)

    def run(self):
        try:
            with SessionLock(self.s.repo / 'logs/autotrain/campaign.lock'):
                number = 0
                while not self.s.max_batches or len(self.state['successful_reports']) < self.s.max_batches:
                    self.guard()
                    number += 1
                    successful = self.attempt(number)
                    if self.state['consecutive_failures'] >= 3:
                        raise CampaignBlocked('Three consecutive failed attempts; automatic retries stopped')
                    if successful and self.s.max_batches and len(self.state['successful_reports']) >= self.s.max_batches:
                        break
                    if successful and self.s.gate_after_success:
                        self.gate()
                    elif not successful:
                        self.save(state='retry_wait')
                        for _ in range(int(self.s.retry_seconds)):
                            self.guard(); self.sleep(1)
            self.save(state='completed')
            return 0
        except (CampaignStop, KeyboardInterrupt) as error:
            self.save(state='stopped', stop_reason=str(error))
            return 2
        except Exception as error:
            self.save(state='needs_attention', error=f'{type(error).__name__}: {error}')
            return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--campaign-id', required=True)
    run.add_argument('--recipe', type=Path, required=True)
    run.add_argument('--repo', type=Path, default=ROOT)
    run.add_argument('--cssim-root', type=Path, default=Path('C:/CSSIM'))
    run.add_argument('--server-pid', type=int)
    run.add_argument('--gate-after-success', action='store_true')
    run.add_argument('--train-candidate', action='store_true')
    run.add_argument('--sources', type=Path, nargs='*', default=[])
    run.add_argument('--max-batches', type=int, default=0)
    run.add_argument('--timeout-minutes', type=float, default=360)
    run.add_argument('--idle-minutes', type=float, default=10)
    run.add_argument('--startup-seconds', type=float, default=300)
    run.add_argument('--minimum-free-gb', type=float, default=10)
    for name in ('status', 'stop', 'release'):
        command = commands.add_parser(name)
        command.add_argument('campaign_directory', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'run':
            values = vars(args).copy(); values.pop('command')
            return Campaign(Settings(**values)).run()
        directory = args.campaign_directory.resolve()
        state = batches.read_json(directory / 'state.json')
        if args.command == 'stop':
            (directory / 'STOP').touch()
        elif args.command == 'release':
            if state['state'] != 'waiting_for_iteration' or not state.get('gate_token'):
                raise CampaignBlocked('Campaign is not waiting at an iteration gate')
            batches.write_json(directory / 'gate-release.json', dict(campaign_id=state['campaign_id'],
                               gate_token=state['gate_token'], released_at=batches.now()))
        print(json.dumps(state, ensure_ascii=True))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(json.dumps(dict(state='error', error=str(error)), ensure_ascii=True))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
