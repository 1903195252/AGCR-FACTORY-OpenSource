"""Short-lived exact memoization of deterministic, fixed video conditions.

Only the condition encoder and geometric key/value projections are cached.
Queries, attention, appearance weighting, all DiT blocks and every denoising
step execute normally. This wrapper changes no trained module source or keys.
"""
from contextlib import contextmanager

import torch
from torch import nn


def fingerprint(value):
    """Metadata/version checks only: no device copies, hashes or synchronization."""
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            return None
        try:
            version = value._version
        except RuntimeError:
            # Inference tensors have no version counter; caching them cannot
            # safely detect in-place writes, so the caller takes the slow path.
            return None
        # Euler deliberately creates new detach() aliases in its no-grad
        # prefix. They retain the same storage, offset and version counter.
        # Use that identity instead of the transient Python tensor id.
        storage = value.untyped_storage()
        return ('tensor', storage.data_ptr(), storage.nbytes(), value.storage_offset(),
                version, tuple(value.shape), tuple(value.stride()),
                str(value.dtype), str(value.device), value.layout)
    if isinstance(value, dict):
        values = [(key, fingerprint(item)) for key, item in value.items()]
        return None if any(item is None for _, item in values) else ('dict', tuple(values))
    if isinstance(value, (tuple, list)):
        items = tuple(fingerprint(item) for item in value)
        return None if any(item is None for item in items) else (type(value).__name__, items)
    if value is None or isinstance(value, (str, bool, int, float)):
        return ('scalar', type(value).__name__, value)
    return None


def autocast_state():
    states = []
    for device in ('cpu', 'cuda'):
        try:
            enabled = torch.is_autocast_enabled(device)
        except TypeError:
            enabled = torch.is_autocast_cpu_enabled() if device == 'cpu' else torch.is_autocast_enabled()
        dtype = None
        if enabled:
            try:
                dtype = torch.get_autocast_dtype(device)
            except AttributeError:
                dtype = torch.get_autocast_cpu_dtype() if device == 'cpu' else torch.get_autocast_gpu_dtype()
        states.append((device, enabled, str(dtype)))
    return tuple(states)


@contextmanager
def inference_condition_cache(wan):
    """Cache one cell's fixed inputs; invalidate on input/weight/output writes.

    A fresh context must surround each complete 50-step cell. Strong references
    prevent storage address reuse; both input and parameter versions are checked on
    every call. Context exit (including exceptions) restores the original bound
    forwards and releases cached tensors. Unversioned inference tensors bypass
    caching. Hooks still execute through the normal nn.Module call boundary.
    """
    if torch.is_grad_enabled():
        raise RuntimeError('Condition caching is restricted to no-grad inference')
    stats = dict(enabled=False, hits=0, misses=0)
    # Minimal sampling stand-ins need no caching; the scientific sampler remains
    # usable without requiring a fake implementation of the native architecture.
    if not isinstance(wan, nn.Module):
        yield stats
        return
    if wan.training or any(module.training for module in wan.modules()):
        raise RuntimeError('Condition caching requires every video module in eval mode')
    if getattr(wan, '_base_condition_cache_active', False):
        raise RuntimeError('Nested condition cache contexts are not supported')
    modules = [wan.encoder]
    for adapter in wan.adapters:
        modules.extend([adapter.geo_key, adapter.geo_value])
    if len({id(module) for module in modules}) != len(modules):
        raise ValueError('Condition projections must have distinct module identities')
    restored, entries = [], []
    wan._base_condition_cache_active = True
    stats['enabled'] = True
    def install(module):
        original = module.forward
        own = module.__dict__.get('forward')
        had_own = 'forward' in module.__dict__
        cache = {}
        entries.append(cache)
        def forward(*args, **kwargs):
            if torch.is_grad_enabled() or any(child.training for child in module.modules()):
                raise RuntimeError('A cached condition projection left no-grad/eval mode')
            parameters = tuple(module.parameters())+tuple(module.buffers())
            inputs = fingerprint((args, tuple(kwargs.items())))
            weights = fingerprint(parameters)
            key = (inputs, weights, autocast_state())
            if inputs is not None and weights is not None and cache.get('key') == key:
                if fingerprint(cache['output']) == cache['output_fingerprint']:
                    stats['hits'] += 1
                    return cache['output']
            result = original(*args, **kwargs)
            stats['misses'] += 1
            cache.clear()
            output_fingerprint = fingerprint(result)
            if inputs is not None and weights is not None and output_fingerprint is not None:
                cache.update(key=key, output=result, output_fingerprint=output_fingerprint,
                             retained_inputs=(args, kwargs), retained_weights=parameters)
            return result
        module.forward = forward
        restored.append((module, had_own, own))
    try:
        for module in modules:
            install(module)
        yield stats
    finally:
        for module, had_own, own in reversed(restored):
            if had_own:
                module.forward = own
            else:
                del module.forward
        for entry in entries:
            entry.clear()
        del wan._base_condition_cache_active
