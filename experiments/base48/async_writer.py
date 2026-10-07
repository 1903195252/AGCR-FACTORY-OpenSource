"""Bounded CPU publication; callers retain commit-after-validation semantics."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor


class BoundedWriter:
    def __init__(self, max_pending=2):
        if not 1 <= max_pending <= 2:
            raise ValueError('At most two pending publications are allowed')
        self.limit = max_pending
        self.pending = deque()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='lossless-npz')

    def wait_for_slot(self):
        # Call before decoding the next CPU payload, not only after allocating it.
        while self.pending and (self.pending[0].done() or len(self.pending) >= self.limit):
            self.pending.popleft().result()

    def submit(self, function, *args):
        self.wait_for_slot()
        future = self.pool.submit(function, *args)
        self.pending.append(future)
        return future

    def flush(self):
        while self.pending:
            self.pending.popleft().result()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            self.flush()
        finally:
            self.pool.shutdown(wait=True)
        return False
