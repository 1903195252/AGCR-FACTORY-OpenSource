"""Batched execution of the unchanged paired observation objective."""
import copy
import multiprocessing as mp
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch.nn import functional as F

from .data import read_observation_pair
from .train_observation import frame_tensors
from .observation import observation_loss


def worker_init():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def prepare_pair(args):
    root, row, offset = args
    # Reduce the native masks to the SAME 24x36 area targets before IPC/H2D.
    sample = read_observation_pair(root, row, offset)
    _, inputs, target = frame_tensors(sample, 0, torch.device('cpu'))
    # RGB interpolation remains on CUDA: CPU interpolation can change features.
    rgb = torch.as_tensor(sample['inputs']['rgb'][0]).permute(0,3,1,2)
    return rgb, inputs, target


def prepare_batch(specs):
    return [prepare_pair(spec) for spec in specs]


class OrderedPrefetch:
    """Future batches never advance the checkpoint's consumed RNG state."""
    def __init__(self, root, rows, rng_state, workers=4, prefetch=3, batch=16):
        if workers < 0 or prefetch < 1:
            raise ValueError('Invalid worker/prefetch count')
        self.root, self.rows, self.batch = root, rows, batch
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = copy.deepcopy(rng_state)
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                        initializer=worker_init) if workers else None
        self.queue = deque()
        for _ in range(prefetch if self.pool else 1):
            self.enqueue()

    def enqueue(self):
        indices = [(int(self.rng.integers(len(self.rows))), int(self.rng.integers(6))) for _ in range(self.batch)]
        specs = [(self.root, self.rows[index], offset) for index, offset in indices]
        state = copy.deepcopy(self.rng.bit_generator.state)
        future = self.pool.submit(prepare_batch, specs) if self.pool else specs
        self.queue.append((future, state, indices))

    def next(self):
        future, state, indices = self.queue.popleft()
        result = future.result() if self.pool else prepare_batch(future)
        self.enqueue()
        return result, state, indices

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        if device.type == 'cuda' and value.device.type == 'cpu':
            value = value.contiguous().pin_memory()
        return value.to(device, non_blocking=True)
    return {key: to_device(item, device) for key, item in value.items()}


def collate_pairs(pairs, device):
    rgb = torch.cat([pair[0] for pair in pairs])
    inputs = {key: torch.cat([pair[1][key] for pair in pairs]) for key in pairs[0][1]}
    targets = [to_device(pair[2], device) for pair in pairs]
    rgb = to_device(rgb, device).float().contiguous(memory_format=torch.channels_last)/255
    rgb = F.interpolate(rgb, size=(336,504), mode='bilinear', align_corners=False)
    rgb = (rgb-rgb.new_tensor([.485,.456,.406])[None,:,None,None])/rgb.new_tensor([.229,.224,.225])[None,:,None,None]
    return rgb, to_device(inputs, device), targets


def pair_outputs(output, index):
    return {key: value[index*2:index*2+2] for key, value in output.items()}


def batched_forward(model, dino, pairs, device, microbatch=8, rgb_only=False, backward=False):
    """One loss per A/G pair, equal pair weighting; association never crosses pairs."""
    if microbatch not in (1, 2, 4, 8, 16):
        raise ValueError('Microbatch must divide the fixed 16-pair effective batch')
    total, components = 0., dict(observation=0., association=0.)
    for start in range(0, len(pairs), microbatch):
        batch = pairs[start:start+microbatch]
        rgb, inputs, targets = collate_pairs(batch, device)
        with torch.no_grad():
            features = dino.forward_features(rgb)['x_norm_patchtokens']
        outputs = model(features, **inputs, rgb_only=rgb_only)
        losses = []
        for i, target in enumerate(targets):
            loss, parts = observation_loss(model, pair_outputs(outputs, i), target)
            losses.append(loss)
            for key in components:
                components[key] += parts[key] / len(pairs)
        loss = torch.stack(losses).sum() / len(pairs)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite batched observation loss')
        if backward:
            loss.backward()
        total += float(loss.detach())
    return total, components


def original_update(model, dino, root, specs, device, backward=True, profile=False):
    """Original on-device preprocessing and 16 sequential paired forwards."""
    totals = dict(read=0., transfer_prepare=0., dino=0., head=0., loss_backward=0.)
    losses, components = [], dict(observation=0., association=0.)
    def stamp():
        if profile and device.type == 'cuda':
            torch.cuda.synchronize(device)
        return time.perf_counter()
    for row, offset in specs:
        t = stamp(); sample = read_observation_pair(root, row, offset); t1 = stamp()
        rgb, inputs, target = frame_tensors(sample, 0, device); t2 = stamp()
        with torch.no_grad():
            features = dino.forward_features(rgb)['x_norm_patchtokens']
        t3 = stamp(); output = model(features, **inputs); t4 = stamp()
        loss, parts = observation_loss(model, output, target)
        if backward:
            (loss/len(specs)).backward()
        losses.append(float(loss.detach()))
        for key in components:
            components[key] += parts[key] / len(specs)
        t5 = stamp()
        for key, value in zip(totals, [t1-t, t2-t1, t3-t2, t4-t3, t5-t4]):
            totals[key] += value
    return float(np.mean(losses)), components, totals
