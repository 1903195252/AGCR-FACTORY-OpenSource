"""Lossless bounded activation snapshots preserving native tensor layout.

Snapshots include the storage prefix needed to reproduce the original offset.
The CPU allowance counts that actual prefix/span, including unused padding,
not just logical tensor.numel(). Existing FeatureStore accounting is retained.
"""
import weakref

import torch

from .decode_activation_offload import ActivationStore,storage_key

POLICY='exact_storage_span_shape_stride_offset_v1'


class SavedLayoutActivation:
    def __init__(self,tensor,owner):
        if tensor.layout!=torch.strided or tensor.is_quantized or tensor.is_conj()or tensor.is_neg():
            raise ValueError('Layout snapshots require ordinary strided tensors')
        self.shape=tuple(tensor.shape);self.stride=tuple(tensor.stride());self.offset=tensor.storage_offset()
        if any(value<0 for value in self.stride)or self.offset<0 or not tensor.numel():
            raise ValueError('Layout snapshots require a nonempty, nonnegative storage span')
        self.elements=self.offset+1+sum((size-1)*stride for size,stride in zip(self.shape,self.stride))
        self.nbytes=self.elements*tensor.element_size()
        if self.nbytes>tensor.untyped_storage().nbytes():raise ValueError('Tensor span exceeds its actual storage')
        if owner.live_bytes+self.nbytes>owner.max_bytes:
            raise MemoryError('Bounded CPU storage-span allowance exhausted before copying')
        self.owner=owner;self.device=tensor.device;self.dtype=tensor.dtype
        self.source_ref=weakref.ref(tensor)
        # detach is solely a saved-tensor representation: autograd restores the
        # original graph edge when unpack returns this tensor's exact data.
        flat=tensor.detach().as_strided((self.elements,),(1,),storage_offset=0)
        self.cpu=flat.to(device='cpu',copy=True,non_blocking=False)
        owner.live_bytes+=self.nbytes;owner.peak_bytes=max(owner.peak_bytes,owner.live_bytes)
        owner.transferred_bytes+=self.nbytes;owner.offloaded_saves+=1
        owner.layout_offloaded_saves+=1
        owner.noncontiguous_offloaded_saves+=int(not tensor.is_contiguous())
        owner.storage_prefix_copied_bytes+=self.offset*tensor.element_size()
        owner.logical_copied_bytes+=tensor.numel()*tensor.element_size()

    def __del__(self):
        if hasattr(self,'cpu'):self.owner.live_bytes-=self.nbytes


class LayoutActivationStore(ActivationStore):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.layout_offloaded_saves=self.layout_restored_tensors=self.noncontiguous_offloaded_saves=0
        self.storage_prefix_copied_bytes=self.logical_copied_bytes=0

    def pack(self,tensor):
        key=storage_key(tensor)
        if key in self.fixed:
            self.skipped_parameter_saves+=1
            return tensor
        if (not tensor.numel()or(tensor.device.type!='cuda'and not self.copy_cpu_for_test)
                or tensor.numel()*tensor.element_size()<self.minimum_bytes):return tensor
        exact=(key,tensor._version,tensor.storage_offset(),tuple(tensor.shape),tuple(tensor.stride()),tensor.dtype)
        saved=self.cache.get(exact)
        if saved is not None and saved.source_ref()is tensor:
            self.reused_saves+=1
            return saved
        saved=SavedLayoutActivation(tensor,self);self.cache[exact]=saved
        return saved

    @staticmethod
    def unpack(value):
        if not isinstance(value,SavedLayoutActivation):return value
        flat=value.cpu.to(device=value.device,non_blocking=False)
        restored=flat.as_strided(value.shape,value.stride,storage_offset=value.offset)
        if (tuple(restored.shape)!=value.shape or tuple(restored.stride())!=value.stride
                or restored.storage_offset()!=value.offset or restored.dtype!=value.dtype):
            raise RuntimeError('Saved-tensor layout restoration changed its contract')
        value.owner.layout_restored_tensors+=1
        return restored

    def report(self):
        result=super().report()
        result.update(saved_tensor_layout_policy=POLICY,
            layout_offloaded_saves=self.layout_offloaded_saves,
            layout_restored_tensors=self.layout_restored_tensors,
            noncontiguous_offloaded_saves=self.noncontiguous_offloaded_saves,
            stride_preserved=True,storage_offset_preserved=True,span_accounted=True,
            storage_span_cpu_bytes=self.live_bytes,storage_span_peak_bytes=self.peak_bytes,
            storage_span_copied_bytes=self.transferred_bytes,
            storage_prefix_copied_bytes=self.storage_prefix_copied_bytes,
            logical_copied_bytes=self.logical_copied_bytes)
        return result
