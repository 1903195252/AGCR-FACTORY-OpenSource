"""File-locked shared queue. A paired camera group is the indivisible job.

Locks remain held for the complete pair and are released by the OS on exit.
Completion markers, not lease timeouts, determine whether work is complete.
"""
import fcntl
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from paper_dataset.common import atomic_json


class Claim:
    def __init__(self, queue, key, plans, handle):
        self.queue, self.key, self.plans, self.handle = queue, key, plans, handle

    def fail(self, error, scope='scene'):
        atomic_json(self.queue.failure_path(self.key), dict(group=list(self.key), plans=self.plans,
                    error=str(error), scope=scope, time=time.time(), pid=os.getpid()))

    def close(self):
        if self.handle:
            self.handle.close()
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class TaskQueue:
    def __init__(self, root, path, run_id):
        self.root, self.path, self.run_id = Path(root), Path(path), run_id
        self.locks = self.path.parent / 'claims'
        self.failures = self.path.parent / 'failures' / run_id
        self.locks.mkdir(parents=True, exist_ok=True)
        self.failures.mkdir(parents=True, exist_ok=True)

    def failure_path(self, key):
        return self.failures / ('l%02d_p%04d.json' % key)

    def document(self):
        doc = json.loads(self.path.read_text())
        if doc['run_id'] != self.run_id:
            raise RuntimeError('Queue controller changed; old worker must exit')
        return doc

    def finished(self, plans):
        return all((self.root / 'episodes' / p['name'] / 'complete.json').is_file() for p in plans)

    def take(self, preferred_layout=None):
        doc = self.document()
        if doc['sealed']:
            return None
        groups = defaultdict(list)
        for p in doc['tasks']:
            groups[p['layout'], p['path']].append(p)
        # Reuse a loaded scene when possible; every remaining group is eligible.
        for key in sorted(groups, key=lambda k: (k[0] != preferred_layout, k)):
            plans = groups[key]
            if self.finished(plans) or self.failure_path(key).exists():
                continue
            handle = (self.locks / ('l%02d_p%04d.lock' % key)).open('a')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            # Recheck after acquisition: another worker may have just finished.
            if self.finished(plans) or self.failure_path(key).exists():
                handle.close()
                continue
            if self.document()['sealed']:
                handle.close()
                return None
            return Claim(self, key, plans, handle)
        return None


def live_worker_pids(root, queue_path):
    """Find workers from an earlier controller without trusting stale PID files."""
    found = []
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
            if (b'--worker' in args and b'--queue-file' in args and
                    str(root).encode() in args and str(queue_path).encode() in args):
                found.append(int(p.name))
        except OSError:
            continue
    return found
