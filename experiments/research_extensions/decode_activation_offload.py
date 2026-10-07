"""Lossless, bounded CPU residency of saved decode/readout activations only.

Model parameters/buffers remain on their original device, including aliased
views. No precision, video dimensions, noise, timestep or objective changes.
Off by default: callers explicitly enter the scope for an independent candidate.
"""
from contextlib import ExitStack,contextmanager
from pathlib import Path
import weakref
from unittest.mock import patch

import torch


def container_memory():
    root=Path('/sys/fs/cgroup')
    if not(root/'memory.max').exists():
        raise RuntimeError('A finite cgroup-v2 memory limit is required for CPU offload')
    raw=(root/'memory.max').read_text().strip()
    if raw=='max':raise RuntimeError('CPU offload needs an explicit container RAM limit')
    stats={key:int(value)for key,value in(line.split()for line in(root/'memory.stat').read_text().splitlines())}
    current=int((root/'memory.current').read_text())
    # Clean mapped file cache may be reclaimed by the kernel. Do not count
    # anonymous tensors, shared memory, dirty pages or writeback as available.
    reclaimable=max(0,stats.get('file',0)-stats.get('shmem',0)-stats.get('file_dirty',0)-stats.get('file_writeback',0))
    return dict(limit_bytes=int(raw),current_bytes=current,reclaimable_file_bytes=reclaimable,
                anonymous_bytes=stats.get('anon',0),available_after_clean_cache_reclaim=int(raw)-current+reclaimable)


def storage_key(tensor):
    return (tensor.device.type,tensor.device.index,tensor.untyped_storage().data_ptr())


class SavedActivation:
    def __init__(self,tensor,owner):
        self.owner=owner;self.device=tensor.device;self.nbytes=tensor.numel()*tensor.element_size()
        self.source_ref=weakref.ref(tensor)
        if owner.live_bytes+self.nbytes>owner.max_bytes:
            raise MemoryError('Bounded CPU saved-activation allowance exhausted before copying')
        self.cpu=tensor.detach().to(device='cpu',copy=True,non_blocking=False)
        owner.live_bytes+=self.nbytes;owner.peak_bytes=max(owner.peak_bytes,owner.live_bytes)
        owner.offloaded_saves+=1;owner.transferred_bytes+=self.nbytes
    def __del__(self):
        if hasattr(self,'cpu'):self.owner.live_bytes-=self.nbytes


class ActivationStore:
    def __init__(self,model,max_bytes,minimum_bytes=1<<20,copy_cpu_for_test=False):
        self.max_bytes=int(max_bytes);self.minimum_bytes=int(minimum_bytes)
        if self.max_bytes<=0:raise ValueError('Positive activation allowance required')
        self.fixed={storage_key(p)for p in list(model.parameters())+list(model.buffers())}
        self.live_bytes=self.peak_bytes=self.transferred_bytes=0
        self.offloaded_saves=self.skipped_parameter_saves=self.reused_saves=0
        self.copy_cpu_for_test=copy_cpu_for_test
        self.cache=weakref.WeakValueDictionary()
    def pack(self,tensor):
        key=storage_key(tensor)
        if key in self.fixed:
            self.skipped_parameter_saves+=1
            return tensor
        if (tensor.device.type!='cuda'and not self.copy_cpu_for_test)or tensor.numel()*tensor.element_size()<self.minimum_bytes:
            return tensor
        version=tensor._version
        exact=(key,version,tensor.storage_offset(),tuple(tensor.shape),tuple(tensor.stride()),tensor.dtype)
        saved=self.cache.get(exact)
        if saved is not None and saved.source_ref()is tensor:
            self.reused_saves+=1
            return saved
        saved=SavedActivation(tensor,self);self.cache[exact]=saved
        return saved
    @staticmethod
    def unpack(value):
        if isinstance(value,SavedActivation):
            return value.cpu.to(device=value.device,non_blocking=False)
        return value
    def report(self):
        return dict(cpu_activation_live_bytes=self.live_bytes,cpu_activation_peak_bytes=self.peak_bytes,
            cpu_activation_limit_bytes=self.max_bytes,device_to_cpu_bytes=self.transferred_bytes,
            offloaded_saves=self.offloaded_saves,reused_saves=self.reused_saves,
            skipped_parameter_or_buffer_saves=self.skipped_parameter_saves,
            dtype_changed=False,parameters_moved=False,scope='model.decode_and_video_readout_saved_tensors_only')


@contextmanager
def decode_saved_tensors_on_cpu(model,max_gib=32.,*,_test_cpu=False):
    if not 0<max_gib<=32:raise ValueError('This admission allows at most32GiB saved CPU activations')
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
