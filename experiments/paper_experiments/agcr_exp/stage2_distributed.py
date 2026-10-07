"""DDP utilities retaining whole paired clips and the serial sample stream."""
import copy
import hashlib
import multiprocessing
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from .selection import validation_mean
from .stage2_data import load_window
from .stage2_system import tensors
from .train_dynamics_geometry import sample_loss


def local_batch_size(global_batch, world_size):
    if world_size < 1 or global_batch < world_size or global_batch % world_size:
        raise ValueError('Global batch must be positive and divisible by world size')
    return global_batch // world_size


class StripedSampler:
    """All ranks advance the same RNG; each loads its own position in a stripe."""
    def __init__(self, count, state, rank, world_size):
        if count < 1 or not 0 <= rank < world_size:
            raise ValueError('Invalid sampler size/rank')
        self.count, self.rank, self.world_size = count, rank, world_size
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = copy.deepcopy(state)

    def next(self):
        selected = None
        for owner in range(self.world_size):
            pair = (int(self.rng.integers(self.count)), int(self.rng.integers(16)))
            if owner == self.rank:
                selected = pair
        return (*selected, copy.deepcopy(self.rng.bit_generator.state))


class DistributedPrefetch:
    def __init__(self, dataset, state, rank, world_size, workers, ahead):
        self.dataset = dataset
        self.sampler = StripedSampler(len(dataset), state, rank, world_size)
        self.pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'))
        self.pending = deque()
        for _ in range(ahead):
            self.enqueue()

    def enqueue(self):
        index, auxiliary, state = self.sampler.next()
        job = (self.dataset.root, self.dataset.rows[index], auxiliary)
        self.pending.append((self.pool.submit(load_window, job), state, index, auxiliary))

    def next(self):
        future, state, index, auxiliary = self.pending.popleft()
        sample = future.result()
        self.enqueue()
        return sample, state, index, auxiliary

    def close(self):
        for future, *_ in self.pending:
            future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)


class ObjectiveModule(nn.Module):
    """The objective must run through DDP.forward, not ddp.module.objective."""
    def __init__(self, model, geometry_chunk):
        super().__init__()
        self.model, self.geometry_chunk = model, geometry_chunk

    def forward(self, batch, encoded, auxiliary):
        loss, parts, diagnostics = self.model.objective(batch, encoded, auxiliary, self.geometry_chunk)
        return loss, {k:v.detach() for k,v in parts.items()}, diagnostics


def accumulation_context(wrapped, microstep, local_batch):
    # Includes both the forward and backward call; only the last microstep syncs.
    return wrapped.no_sync() if hasattr(wrapped, 'no_sync') and microstep < local_batch-1 else nullcontext()


def gather_objects(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def stop_on_any_rank(requested, device):
    flag = torch.tensor(int(requested), device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def capture_rng(device):
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(saved, device):
    torch.set_rng_state(saved['torch'].cpu())
    torch.cuda.set_rng_state(saved['cuda'].cpu(), device)
    np.random.set_state(saved['numpy'])
    random.setstate(saved['python'])


def state_digest(model):
    checksum = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        checksum.update(name.encode())
        checksum.update(value.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def merge_training_records(records):
    """Restore serial sample order, preserving even chance duplicate draws."""
    ordered = sorted((sample for rank in records for sample in rank['samples']), key=lambda x:x['position'])
    expected = list(range(len(ordered)))
    if [x['position'] for x in ordered] != expected:
        raise ValueError('Missing/duplicated global batch positions')
    keys = {key for sample in ordered for key in sample['parts']}
    return dict(loss=sum(x['loss'] for x in ordered)/len(ordered),
                loss_components={k:sum(x['parts'].get(k,0.) for x in ordered)/len(ordered) for k in sorted(keys)},
                sample_indices=[x['selection'] for x in ordered],
                training_diagnostics=[x['diagnostics'] for x in ordered],
                data_wait_seconds=max(r['data_wait_seconds'] for r in records),
                rank_data_wait_seconds=[r['data_wait_seconds'] for r in records],
                rank_peak_gib=[r['peak_gib'] for r in records])


@torch.no_grad()
def distributed_validation(model, encoders, dataset, rows, device, geometry_chunk, rank, world_size, progress=None):
    model.eval()
    results = []
    try:
        for position in range(rank, len(rows), world_size):
            row = rows[position]
            sample = load_window((dataset.root, row, 5))
            loss, parts, metrics = sample_loss(model, encoders, sample, device, geometry_chunk)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite validation: '+row['uid'])
            results.append(dict(position=position, loss=float(loss),
                                parts={k:float(v) for k,v in parts.items()}, diagnostics=dict(uid=row['uid'], **metrics)))
            if progress:
                progress(len(results), len(range(rank,len(rows),world_size)))
    finally:
        model.train()
    results = sorted((item for block in gather_objects(results) for item in block), key=lambda x:x['position'])
    if [x['position'] for x in results] != list(range(len(rows))):
        raise RuntimeError('Validation panel must be covered exactly once')
    values = [x['loss'] for x in results]
    mean, by_layout = validation_mean(rows, values)
    return dict(validation_loss=mean, validation_by_layout=by_layout, window_losses=values,
                components=[x['parts'] for x in results], diagnostics=[x['diagnostics'] for x in results])
