"""One bounded single-GPU job charged to an explicitly authorized round ledger.

This is a process wrapper, not a scheduler. It preserves existing job records,
reserves capacity under a short flock transaction, and pins the child to the
requested physical GPU (the child must use cuda:0). A Linux child subreaper and
PID/start-time checks cover grandchildren that create new sessions, including
the training/sampling wrappers' own workers.
"""
import argparse
from contextlib import contextmanager
import ctypes
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

SCHEMA = 'peerdream-direction-budget-1'
ACTIVE = {'reserved', 'running', 'stopping', 'cleanup_failed'}
TERMINAL = {'completed', 'complete', 'failed', 'stopped', 'timeout', 'launch_failed'}


class StopRequested(Exception):
    pass


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


@contextmanager
def ledger_transaction(path):
    path = Path(path)
    with Path(str(path) + '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(path.read_text())
        yield ledger
        atomic_json(path, ledger)


def finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid nonnegative ledger value: ' + name)
    return float(value)


def accounting(ledger, now):
    if ledger.get('schema') != SCHEMA:
        raise ValueError('Only the existing direction-budget schema is supported')
    cap = finite_nonnegative(ledger.get('gpu_hour_cap'), 'gpu_hour_cap')
    authorized = finite_nonnegative(ledger.get('authorized_gpu_hour_cap', 4.), 'authorized_gpu_hour_cap')
    if not 0 < cap <= authorized or (authorized > 4 and not ledger.get('authorization')):
        raise ValueError('Round cap exceeds its recorded authorization')
    used = finite_nonnegative(ledger.get('prior_region_probe_gpu_hours'), 'prior_region_probe_gpu_hours')
    reservation = 0.
    rows = ledger.get('jobs')
    if not isinstance(rows, list) or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Ledger jobs must be a list with unique IDs')
    for row in rows:
        if row.get('gpu_count') != 1 or len(row.get('gpus', [])) != 1:
            raise ValueError('Direction wrapper supports existing single-GPU jobs only')
        state = row.get('state')
        if state in ACTIVE:
            started = finite_nonnegative(row.get('started_unix'), 'started_unix')
            maximum = finite_nonnegative(row.get('maximum_wall_seconds'), 'maximum_wall_seconds')
            if maximum <= 0 or started > now + 5:
                raise ValueError('Invalid active reservation clock or duration')
            elapsed = max(0., now-started)
            used += elapsed/3600.
            reservation += max(0., maximum-elapsed)/3600.
        elif state in TERMINAL:
            used += finite_nonnegative(row.get('used_gpu_hours'), 'used_gpu_hours')
        else:
            raise ValueError('Unknown ledger job state: ' + str(state))
    return dict(used_gpu_hours=used, remaining_reserved_gpu_hours=reservation,
                committed_gpu_hours=used+reservation, gpu_hour_cap=cap)


def process_table():
    result = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            text = path.read_text()
            fields = text[text.rfind(')')+2:].split()
            pid = int(path.parent.name)
            result[pid] = dict(pid=pid, state=fields[0], ppid=int(fields[1]),
                               process_group=int(fields[2]), session=int(fields[3]),
                               process_start_ticks=fields[19])
        except (OSError, ValueError, IndexError):
            continue
    return result


def start_ticks(pid):
    try:
        text = Path('/proc/%d/stat' % pid).read_text()
        return text[text.rfind(')')+2:].split()[19]
    except (OSError, IndexError):
        return None


def register(ledger_path, job_id, gpu, maximum, cwd, log, command, owner, now=None):
    now = time.time() if now is None else now
    if not 0 < finite_nonnegative(maximum, 'new_maximum_wall_seconds') <= 14400:
        raise ValueError('A positive bounded reservation <=14400 seconds is required')
    with ledger_transaction(ledger_path) as ledger:
        usage = accounting(ledger, now)
        if any(row['id'] == job_id for row in ledger['jobs']):
            raise ValueError('Job ID already exists; never overwrite a prior attempt')
        if any(row['state'] in ACTIVE and gpu in row['gpus'] for row in ledger['jobs']):
            raise ValueError('Requested GPU is already reserved; close the prior job explicitly')
        if usage['committed_gpu_hours'] + maximum/3600. > usage['gpu_hour_cap'] + 1e-12:
            raise ValueError('New reservation exceeds the remaining authorized round budget')
        row = dict(id=job_id, owner=owner, state='reserved', gpu_count=1, gpus=[gpu],
                   pid=None, process_start_ticks=None, wrapper_pid=os.getpid(),
                   wrapper_start_ticks=start_ticks(os.getpid()), started_unix=now,
                   maximum_wall_seconds=maximum, command=command, cwd=str(cwd), log=str(log),
                   cuda_visible_devices=str(gpu), child_logical_device='cuda:0',
                   execution='single_GPU_process_tree_with_subreaper',
                   used_gpu_hours=0., reservation_gpu_hours=maximum/3600.)
        ledger['jobs'].append(row)
        return row


def update_owned(ledger_path, job_id, owner, changes):
    with ledger_transaction(ledger_path) as ledger:
        rows = [row for row in ledger['jobs'] if row['id'] == job_id]
        if len(rows) != 1 or rows[0].get('owner') != owner:
            raise ValueError('Only the registering wrapper may update this job')
        if {'id', 'owner', 'maximum_wall_seconds', 'started_unix', 'gpus', 'gpu_count'} & set(changes):
            raise ValueError('Original reservation identity is immutable')
        rows[0].update(changes, updated_unix=time.time())


def enable_subreaper():
    # PR_SET_CHILD_SUBREAPER: orphaned descendants, including setsid workers,
    # are adopted by this dedicated wrapper rather than lost under PID 1.
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, 'prctl') or libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Linux child subreaper is required for GPU worker cleanup')


def observe_tree(wrapper_pid, known, table=None):
    table = process_table() if table is None else table
    owned = {wrapper_pid}
    # Include tracked live processes even if a parent/session changed.
    owned.update(pid for pid, ticks in known.items()
                 if pid in table and table[pid]['process_start_ticks'] == ticks)
    changed = True
    while changed:
        new = {pid for pid, row in table.items() if row['ppid'] in owned}
        changed = not new <= owned
        owned |= new
    for pid in owned - {wrapper_pid}:
        if pid in table:
            known[pid] = table[pid]['process_start_ticks']
    return {pid: table[pid] for pid in owned - {wrapper_pid} if pid in table
            and known.get(pid) == table[pid]['process_start_ticks']}


def signal_tree(rows, sig):
    # Signal each matching PID, not just one process group: a worker can have
    # called setsid. Recheck the start tick immediately before each signal.
    for pid, row in rows.items():
        if row['state'] != 'Z' and start_ticks(pid) == row['process_start_ticks']:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


def reap_adopted(rows, main_pid):
    for pid, row in rows.items():
        if pid == main_pid or row['ppid'] != os.getpid():
            continue
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass


def monitor(child, maximum, on_stop=None):
    """Soft TERM then hard tree cleanup, including orphaned/new-session workers."""
    began = time.monotonic()
    deadline = began + maximum
    # Cleanup is inside the reservation, with a small OS reaping allowance.
    hard_at = deadline - min(2., maximum*.05)
    soft_at = deadline - min(30., max(.02, maximum*.10))
    requested = [False]
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    for sig in old:
        signal.signal(sig, lambda *_: requested.__setitem__(0, True))
    known = {}
    ticks = start_ticks(child.pid)
    if ticks is not None:
        known[child.pid] = ticks
    soft = forced = orphaned = False
    stop_reason = None
    live = {}
    try:
        while True:
            rows = observe_tree(os.getpid(), known)
            code = child.poll()
            live = {pid: row for pid, row in rows.items() if row['state'] != 'Z'}
            reap_adopted(rows, child.pid)
            now = time.monotonic()
            if code is not None:
                # Re-scan after wait/poll: the child may have exited and had a
                # new-session worker adopted between the previous scan and poll.
                rows = observe_tree(os.getpid(), known)
                reap_adopted(rows, child.pid)
                live = {pid: row for pid, row in rows.items() if row['state'] != 'Z'}
                live.pop(child.pid, None)
                if not live:
                    break
                orphaned = True
                signal_tree(live, signal.SIGKILL)
            if (requested[0] or now >= soft_at) and not soft:
                soft = True
                stop_reason = 'signal' if requested[0] else 'wall_budget'
                signal_tree(rows, signal.SIGTERM)
                if on_stop is not None:
                    on_stop()
                if requested[0]:
                    hard_at = min(hard_at, now + min(30., maximum*.10))
            if now >= hard_at:
                forced = True
                signal_tree(rows, signal.SIGKILL)
            if now >= deadline:
                rows = observe_tree(os.getpid(), known)
                signal_tree(rows, signal.SIGKILL)
                live = {pid: row for pid, row in rows.items() if row['state'] != 'Z'}
                # A short bounded reap does not authorize further GPU work.
                try:
                    child.wait(timeout=.25)
                except subprocess.TimeoutExpired:
                    pass
                final = observe_tree(os.getpid(), known)
                reap_adopted(final, child.pid)
                live = {pid: row for pid, row in final.items() if row['state'] != 'Z'}
                break
            time.sleep(min(.1, max(.005, hard_at-now if not forced else deadline-now)))
        code = child.poll()
        return dict(exit_code=code, soft_stop_requested=soft, hard_cleanup=forced,
                    stop_reason=stop_reason,
                    orphaned_descendants_detected=orphaned,
                    remaining_processes=list(live.values()),
                    observed_processes=[dict(pid=pid, process_start_ticks=ticks) for pid, ticks in sorted(known.items())],
                    monitored_wall_seconds=time.monotonic()-began)
    finally:
        # Any wrapper exception still cleans matching descendants, not a broad
        # pkill by command name. Preserve the PID/start-time identity checks.
        rows = observe_tree(os.getpid(), known)
        signal_tree(rows, signal.SIGKILL)
        try:
            child.wait(timeout=.25)
        except subprocess.TimeoutExpired:
            pass
        reap_adopted(observe_tree(os.getpid(), known), child.pid)
        for sig, handler in old.items():
            signal.signal(sig, handler)


def run(args):
    enable_subreaper()
    cwd, log = Path(args.cwd).resolve(), Path(args.log).resolve()
    if not cwd.is_dir() or log.exists():
        raise ValueError('Existing cwd and a new log path are required')
    log.parent.mkdir(parents=True, exist_ok=True)
    owner = uuid.uuid4().hex
    began_wall, began_mono = time.time(), time.monotonic()
    register(args.ledger, args.id, args.gpu, args.max_seconds, cwd, log, args.command, owner, began_wall)
    child = None
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    def startup_stop(signum, _frame):
        raise StopRequested('Wrapper stop signal ' + str(signum))
    try:
        for sig in previous_handlers:
            signal.signal(sig, startup_stop)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), WORLD_SIZE='1', RANK='0', LOCAL_RANK='0')
        with log.open('xb') as stream:
            child = subprocess.Popen(args.command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                     env=environment, start_new_session=True)
            ticks = start_ticks(child.pid)
            if ticks is None:
                raise RuntimeError('Unable to pin child process start time')
            update_owned(args.ledger, args.id, owner, dict(state='running', pid=child.pid, process_start_ticks=ticks))
            remaining = args.max_seconds - (time.monotonic()-began_mono)
            if remaining <= 0:
                os.killpg(child.pid, signal.SIGKILL)
                raise TimeoutError('Reservation expired during process startup')
            result = monitor(child, remaining,
                             on_stop=lambda: update_owned(args.ledger, args.id, owner, dict(state='stopping')))
        if result['remaining_processes']:
            state = 'cleanup_failed'  # Remains active/reserved until explicitly reconciled.
        elif result['hard_cleanup']:
            state = 'timeout'
        elif result['orphaned_descendants_detected']:
            state = 'failed'
        elif result['soft_stop_requested']:
            state = 'stopped'
        else:
            state = 'completed' if result['exit_code'] == 0 else 'failed'
        update_owned(args.ledger, args.id, owner,
                     dict(state=state, finished_unix=time.time(), used_gpu_hours=(time.monotonic()-began_mono)/3600.,
                          **result))
        return 0 if state == 'completed' else 1
    except BaseException as error:
        if child is not None:
            known = {}
            tick = start_ticks(child.pid)
            if tick is not None:
                known[child.pid] = tick
            rows = observe_tree(os.getpid(), known)
            signal_tree(rows, signal.SIGKILL)
            try:
                child.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                pass
        update_owned(args.ledger, args.id, owner,
                     dict(state='stopped' if isinstance(error, StopRequested) else ('launch_failed' if child is None else 'failed'),
                          finished_unix=time.time(), used_gpu_hours=(time.monotonic()-began_mono)/3600.,
                          error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('ledger', 'id', 'cwd', 'log'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    if args.command and args.command[0] == '--':
        args.command = args.command[1:]
    if (not args.command or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}', args.id)
            or args.gpu < 0 or not 0 < args.max_seconds <= 14400):
        p.error('Require a command, unique simple ID, nonnegative physical GPU and positive cap <=14400s')
    return args


if __name__ == '__main__':
    raise SystemExit(run(arguments()))
