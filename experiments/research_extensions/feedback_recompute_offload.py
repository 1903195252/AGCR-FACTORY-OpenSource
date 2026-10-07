"""Bounded same-dtype activation residency covering reentrant recomputation.

The per-view function re-enters saved-tensor hooks when autograd calls it during
backward. Wrapping only model.decode misses that call, since a checkpoint keeps
the nested function rather than invoking model.decode again. Numerical kernels,
temporal order, randomness, trainable parameters and loss terms are unchanged.
"""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch

from .decode_activation_offload import ActivationStore, container_memory
from .feedback_checkpoint_boundaries import BOUNDARIES
from .feedback_reentrant_boundaries import tensor_reentrant


@contextmanager
def recompute_saved_tensors_on_cpu(model, max_gib=128., *, _test_cpu=False,
                                  _module=None):
    if not 0 < max_gib <= 128:
        raise ValueError('Saved activations and native features share a maximum128GiB CPU allowance')
    before = None if _test_cpu else container_memory()
    cap = int(max_gib * 2**30)
    if before is not None and before['available_after_clean_cache_reclaim'] - 16*2**30 < cap:
        raise MemoryError('CPU offload needs its full allowance plus16GiB reserve')
    if _module is None:
        from experiments.paired_interventions import system as _module
    original = _module.checkpoint
    store = ActivationStore(model, cap, minimum_bytes=0 if _test_cpu else 1 << 20,
                            copy_cpu_for_test=_test_cpu)
    record = dict(policy='reentrant_view_with_inner_saved_tensor_offload',
                  boundary_calls={name: 0 for name in sorted(BOUNDARIES)},
                  region_evaluations=0, per_view_no_grad_calls=0,
                  per_view_grad_calls=0, per_view_grad_offloaded_saves=0,
                  other_module_calls_preserved=0,
                  leaf_checkpoints_preserved=True, inputs_detached=False,
                  sentinel_changes_values=False)

    def dispatch(function, *args, **kwargs):
        name = getattr(function, '__qualname__', '')
        if name not in BOUNDARIES:
            record['other_module_calls_preserved'] += 1
            return original(function, *args, **kwargs)
        if kwargs != dict(use_reentrant=False):
            raise ValueError('Composite checkpoint contract changed')
        record['boundary_calls'][name] += 1
        if name == 'JointSystem.feedback.<locals>.generated_readout':
            return function(*args)

        def offloaded_view(*values):
            # This scope must be inside the closure saved by checkpoint, not
            # merely around dispatch/checkpoint's original forward call.
            grad_enabled = torch.is_grad_enabled()
            key = 'per_view_grad_calls' if grad_enabled else 'per_view_no_grad_calls'
            record[key] += 1
            prior_saves = store.offloaded_saves
            try:
                with torch.autograd.graph.saved_tensors_hooks(store.pack, store.unpack):
                    return function(*values)
            finally:
                if grad_enabled:
                    record['per_view_grad_offloaded_saves'] += store.offloaded_saves-prior_saves

        return tensor_reentrant(offloaded_view, args, record)

    with ExitStack() as stack:
        for name in ('decode', 'video_readout'):
            method = getattr(model, name)

            def wrapped(*args, _method=method, **kwargs):
                with torch.autograd.graph.saved_tensors_hooks(store.pack, store.unpack):
                    return _method(*args, **kwargs)

            stack.enter_context(patch.object(model, name, wrapped))
        stack.enter_context(patch.object(_module, 'checkpoint', dispatch))
        try:
            yield store, record
        finally:
            store.container_before = before
            store.container_after = None if _test_cpu else container_memory()
