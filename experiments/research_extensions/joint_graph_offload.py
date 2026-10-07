"""Same bounded store for historical geometry and decoded-feedback graphs.

The two32GiB admission reached geometry allocation before entering the previous
decode-only hooks. This scope includes all forward graphs accumulated by one
joint_backward call, including both camera geometries, evidence, and replay.
The inner per-view hook remains active during checkpoint recomputation.
"""
from contextlib import contextmanager

import torch

from .feedback_recompute_offload import recompute_saved_tensors_on_cpu


@contextmanager
def joint_graph_saved_tensors_on_cpu(model, max_gib=128., *, _test_cpu=False,
                                   _module=None):
    with recompute_saved_tensors_on_cpu(model, max_gib, _test_cpu=_test_cpu,
                                      _module=_module) as (store, record):
        record['joint_forward_scope'] = True
        record['joint_forward_operations_unchanged'] = True
        with torch.autograd.graph.saved_tensors_hooks(store.pack, store.unpack):
            yield store, record
