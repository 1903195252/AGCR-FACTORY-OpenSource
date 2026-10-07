"""Same saved-tensor offload kernel with an independently bounded 64 GiB cap.

The prior 32 GiB admission is preserved. Only its conservative allocation
ceiling changes; the original ActivationStore/pack/unpack implementation is
reused. Container clean-cache reclaim is checked with an extra 16 GiB reserve.
"""
from contextlib import ExitStack,contextmanager
from unittest.mock import patch

import torch

from .decode_activation_offload import ActivationStore,container_memory


@contextmanager
def decode_saved_tensors_on_cpu64(model,max_gib=64.,*,_test_cpu=False):
    if not 0<max_gib<=64:raise ValueError('This admission allows at most64GiB saved CPU activations')
    before=None if _test_cpu else container_memory()
    cap=int(max_gib*2**30)
    if before is not None and before['available_after_clean_cache_reclaim']-16*2**30<cap:
        raise MemoryError('Container RAM lacks the requested offload allowance plus16GiB reserve')
    store=ActivationStore(model,cap,minimum_bytes=0 if _test_cpu else 1<<20,copy_cpu_for_test=_test_cpu)
    with ExitStack()as stack:
        for name in('decode','video_readout'):
            original=getattr(model,name)
            def wrapped(*args,_original=original,**kwargs):
                with torch.autograd.graph.saved_tensors_hooks(store.pack,store.unpack):
                    return _original(*args,**kwargs)
            stack.enter_context(patch.object(model,name,wrapped))
        try:yield store
        finally:
            store.container_before=before
            store.container_after=None if _test_cpu else container_memory()
