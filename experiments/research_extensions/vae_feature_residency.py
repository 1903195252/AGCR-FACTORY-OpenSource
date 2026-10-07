"""Bounded, differentiable CPU residency for native VAE depth-feature hooks.

The native decoder and depth layers are unchanged. Five lists of native hook
outputs otherwise remain GPU-resident until all temporal concatenations finish;
these Python references are not affected by saved-tensor hooks. This explicit
scope moves only those outputs, preserves dtype and temporal order, and supplies
one feature level at a time to the original depth projection.
"""
from contextlib import contextmanager
import weakref
from unittest.mock import patch

import torch


class FeatureStore:
    def __init__(self, activation_store):
        self.activations=activation_store
        self.limit=activation_store.max_bytes
        self.live=self.peak=self.copied=self.concatenated=self.levels=0

    def _release(self, size):
        self.live-=size
        self.activations.max_bytes=self.limit-self.live

    def allocate(self, size, operation):
        if self.live+self.activations.live_bytes+size>self.limit:
            raise MemoryError('Combined saved-activation/native-feature CPU allowance exhausted')
        self.live+=size;self.peak=max(self.peak,self.live)
        self.activations.max_bytes=self.limit-self.live
        try: value=operation()
        except BaseException:
            self._release(size);raise
        weakref.finalize(value,self._release,size)
        return value

    def copy(self, value):
        size=value.numel()*value.element_size()
        result=self.allocate(size,lambda:value.to(device='cpu',copy=True,non_blocking=False))
        self.copied+=size
        return result

    def concatenate(self, parts):
        size=sum(v.numel()*v.element_size()for v in parts)
        result=self.allocate(size,lambda:torch.cat(parts,2))
        self.concatenated+=size;self.levels+=1
        return result

    def report(self):
        return dict(feature_cpu_live_bytes=self.live,feature_cpu_peak_bytes=self.peak,
            feature_device_to_cpu_bytes=self.copied,feature_concat_bytes=self.concatenated,
            feature_levels=self.levels,combined_cpu_limit_bytes=self.limit,
            dtype_changed=False,gradient_detached=False,temporal_order_changed=False,
            depth_execution='unchanged forward with lazy per-level device copy')


class DeviceFeatures:
    def __init__(self, features, device):self.features,self.device=features,device
    def __iter__(self):
        for feature in self.features:
            yield feature.to(device=self.device,non_blocking=False)


@contextmanager
def native_features_on_cpu(model,activation_store,*,_expected_channels=None):
    store=FeatureStore(activation_store)
    wrapper=model.vae
    expected=[1024,1024,1024,512,256]if _expected_channels is None else _expected_channels
    def decode(z):
        features=[[]for _ in range(5)];handles=[]
        layers=[wrapper.vae.decoder.mid_block,*wrapper.vae.decoder.up_blocks]
        if len(layers)!=5:raise ValueError('Native decoder must have exactly five feature levels')
        for i,layer in enumerate(layers):
            def collect(module,args,value,i=i):features[i].append(store.copy(value))
            handles.append(layer.register_forward_hook(collect))
        try:
            rgb=wrapper.vae.decode(z*wrapper.std+wrapper.mean).sample
            # Native causal caches are owned/cleared by the original decode;
            # retain exactly that lifecycle, including exception cleanup.
            values=[]
            for parts in features:
                values.append(store.concatenate(parts))
                parts.clear()
            if [value.shape[1]for value in values]!=expected:
                raise ValueError('Unexpected official VAE decoder feature contract')
            return(rgb,*values)
        finally:
            for handle in handles:handle.remove()
            wrapper.vae.clear_cache()
    original_depth=model.depth.forward
    def depth(z,features,geometry,view):
        return original_depth(z,DeviceFeatures(features,z.device),geometry,view)
    with patch.object(wrapper,'_decode',decode),patch.object(model.depth,'forward',depth):
        yield store
