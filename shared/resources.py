from shared.paths import environment
"""OS exclusion plus durable queue. Never kills processes or resets a board."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import subprocess
import time
import uuid

from shared.io import ROOT, STAGES, atomic_json, read

_DETACHED_CHILDREN = {}


def reap_finished_children():
    """Retain handles for deliberately un-killed children; reap only after exit."""
    for pid, child in list(_DETACHED_CHILDREN.items()):
        if child.poll() is not None:
            child.wait()
            del _DETACHED_CHILDREN[pid]


def now():
    return datetime.now(timezone.utc).isoformat()


class Busy(RuntimeError):
    pass


class ResourceLock:
    """Keep the lock file in place. Owner JSON is diagnostic, never the lock."""
    def __init__(self, directory, name):
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,39}', name):
            raise ValueError('Invalid resource name')
        self.directory, self.name, self.handle = Path(directory), name, None
        self.token = uuid.uuid4().hex

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.handle = (self.directory / (self.name + '.lock')).open('a+b')
        if self.handle.seek(0, os.SEEK_END) == 0:
            self.handle.write(b'\0')
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise Busy('Resource already held: ' + self.name) from exc
        try:
            atomic_json(self.directory / (self.name + '.owner.json'),
                        dict(pid=os.getpid(), token=self.token, acquired_utc=now(), resource=self.name))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        if self.handle is None:
            return
        try:
            atomic_json(self.directory / (self.name + '.owner.json'),
                        dict(pid=os.getpid(), token=self.token, released_utc=now(), resource=self.name))
        finally:
            self.handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class Queue:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.locks = self.root / 'state/locks'
        self.path = self.root / 'board_queue.json'

    def _load(self):
        return read(self.path) if self.path.exists() else dict(schema='finedge_resource_queue_v1', requests=[])

    @contextmanager
    def edit(self):
        with ResourceLock(self.locks, 'queue_metadata'):
            data = self._load()
            yield data
            atomic_json(self.path, data)

    def initialize(self):
        with self.edit() as data:
            data.setdefault('policy', dict(owner='G', software_restart_preferred=True,
                physical_power_cycle_default=False, stale_active_requires_explicit_reconciliation=True,
                resources=['board', 'full_vivado', 'rtl_simulation']))
        return self._load()

    def submit(self, *, resource, stage, candidate_manifest, candidate_sha256, command):
        # Pure simulation has isolated output/cache directories and never
        # mutates the active implementation or accesses the board. Keep its
        # FIFO independent so a long router cannot block corrective RTL tests.
        if resource not in ('board', 'full_vivado', 'rtl_simulation') or stage not in STAGES:
            raise ValueError('Unknown resource/stage')
        if not isinstance(command, list) or not command or any(type(s) is not str or not s for s in command):
            raise ValueError('Command must be a nonempty argv list; no shell interpolation')
        item = dict(request_id=uuid.uuid4().hex, owner='G', resource=resource, stage=stage,
                    candidate_manifest=str(Path(candidate_manifest).resolve()),
                    candidate_manifest_sha256=candidate_sha256, command=command,
                    state='queued', submitted_utc=now(), finished_utc=None)
        with self.edit() as data:
            data['requests'].append(item)
        return item

    def run(self, request_id, *, qualify, timeout_seconds=3600):
        reap_finished_children()
        if timeout_seconds <= 0:
            raise ValueError('Timeout must be positive')
        requests = self._load()['requests']
        request = next((r for r in requests if r['request_id'] == request_id), None)
        if request is None:
            raise ValueError('Unknown request')
        resource = request['resource']
        with ResourceLock(self.locks, resource):
            with self.edit() as data:
                request = next(r for r in data['requests'] if r['request_id'] == request_id)
                if request['state'] != 'queued':
                    raise ValueError('Request is one-use and is not queued')
                unresolved = [r for r in data['requests'] if r['resource'] == resource and
                              r['state'] in ('active', 'needs_reconciliation', 'finished_pending_quiescence')]
                if unresolved:
                    raise Busy('Prior execution unresolved; explicit evidence required, not automatic reuse')
                earlier = [r for r in data['requests'] if r['resource'] == resource and r['state'] == 'queued']
                if earlier[0]['request_id'] != request_id:
                    raise Busy('Earlier queued request must finish or be explicitly cancelled first')
                qualify(dict(request))
                request.update(state='active', started_utc=now(), supervisor_pid=os.getpid())
            run_dir = self.root / 'artifacts/resource_runs' / request_id
            run_dir.mkdir(parents=True, exist_ok=False)
            started = time.perf_counter_ns()
            process = None
            state, code, error = 'needs_reconciliation', None, None
            try:
                with (run_dir / 'stdout.log').open('xb') as stdout, (run_dir / 'stderr.log').open('xb') as stderr:
                    process = subprocess.Popen(request['command'], cwd=self.root, stdout=stdout, stderr=stderr,
                                               shell=False, env=environment())
                    with self.edit() as data:
                        current = next(r for r in data['requests'] if r['request_id'] == request_id)
                        current.update(child_pid=process.pid, child_started_utc=now())
                    code = process.wait(timeout=timeout_seconds)
                state = ('finished_pending_quiescence' if resource == 'board' else 'completed') if code == 0 else 'needs_reconciliation'
            except subprocess.TimeoutExpired:
                # The child may still access hardware. Do not kill it or permit the next request.
                _DETACHED_CHILDREN[process.pid] = process
                error = 'Timeout; child was not killed; resource remains logically quarantined'
            except BaseException as exc:
                error = type(exc).__name__ + ': ' + str(exc)
                if process is not None and process.poll() is None:
                    _DETACHED_CHILDREN[process.pid] = process
                # Even a failed launcher can leave prior hardware producers active.
            finally:
                result = dict(request_id=request_id, state=state, return_code=code, error=error,
                    elapsed_ns=time.perf_counter_ns() - started,
                    child_pid=process.pid if process else None, finished_utc=now())
                atomic_json(run_dir / 'result.json', result)
                with self.edit() as data:
                    current = next(r for r in data['requests'] if r['request_id'] == request_id)
                    current.update(result)
            return result

    def cancel(self, request_id, reason):
        if not reason.strip():
            raise ValueError('Cancellation reason required')
        with self.edit() as data:
            item = next((r for r in data['requests'] if r['request_id'] == request_id), None)
            if item is None or item['state'] != 'queued':
                raise ValueError('Only a queued request can be cancelled')
            item.update(state='cancelled', reason=reason, finished_utc=now())
        return item

    def reconcile(self, request_id, evidence_path):
        """Explicit G evidence, no arbitrary force-unlock/kill shortcut."""
        from shared.io import sha
        evidence = read(evidence_path)
        needed = ('request_id', 'child_access_stopped', 'resource_quiescent', 'method', 'observed_utc', 'observations')
        if any(k not in evidence for k in needed) or evidence['request_id'] != request_id:
            raise ValueError('Reconciliation evidence missing/mismatched')
        if evidence['child_access_stopped'] is not True or evidence['resource_quiescent'] is not True:
            raise ValueError('Explicit stopped/quiescent observations are required')
        if evidence['method'] not in ('normal_exit', 'service_restart', 'software_reboot', 'physical_power_cycle', 'controlled_build_stop'):
            raise ValueError('Unknown recovery method')
        if not isinstance(evidence['observations'], list) or not evidence['observations']:
            raise ValueError('Observed evidence must be recorded')
        if evidence['method'] == 'physical_power_cycle' and not evidence.get('why_software_was_insufficient'):
            raise ValueError('Explain why software recovery was insufficient')
        request = next((r for r in self._load()['requests'] if r['request_id'] == request_id), None)
        if request is None:
            raise ValueError('Unknown request')
        if evidence['method'] == 'controlled_build_stop':
            if request['resource'] != 'full_vivado':
                raise ValueError('Controlled build stop cannot release a board request')
            if evidence.get('build_processes_absent') is not True or not evidence.get('build_directory'):
                raise ValueError('Controlled build stop requires build identity and absent-process observation')
        with ResourceLock(self.locks, request['resource']):
            with self.edit() as data:
                item = next(r for r in data['requests'] if r['request_id'] == request_id)
                if item['state'] not in ('active', 'needs_reconciliation', 'finished_pending_quiescence'):
                    raise ValueError('Only interrupted/unresolved execution needs reconciliation')
                item.update(state='reconciled', reconciliation_path=str(Path(evidence_path).resolve()),
                    reconciliation_sha256=sha(evidence_path), reconciled_utc=now())
        return item
