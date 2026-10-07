"""Invocation-local native Wan decoding for nested checkpoint replay.

The official temporal loop, layers, first-chunk flag, patching and clamp are
unchanged. Instance-wide mutable causal-cache objects are replaced by a local
list for each decode invocation. Feature hooks accept only their own invocation.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import sys
import threading
from unittest.mock import patch

import torch

from .vae_feature_residency import FeatureStore,DeviceFeatures


def decode_with_local_cache(vae,z,return_dict,decoder_output,unpatchify):
    if vae.use_tiling:raise ValueError('Native full-frame decoding must have tiling disabled')
    # Same count and initial values as official clear_cache(), but no write to
    # shared _feat_map/_conv_idx. Each explicit decoder call owns its index too.
    cache=[None]*vae._cached_conv_counts['decoder']
    x=vae.post_quant_conv(z)
    for i in range(z.shape[2]):
        value=vae.decoder(x[:,:,i:i+1],feat_cache=cache,feat_idx=[0],first_chunk=(i==0))
        out=value if i==0 else torch.cat([out,value],2)
    if vae.config.patch_size is not None:out=unpatchify(out,patch_size=vae.config.patch_size)
    out=torch.clamp(out,min=-1.,max=1.)
    return decoder_output(sample=out)if return_dict else(out,)


@contextmanager
def reentrant_native_features(model,activation_store,*,_expected_channels=None,_output_factory=None):
    wrapper=model.vae;vae=wrapper.vae
    source=sys.modules[type(vae).__module__]
    factory=getattr(source,'DecoderOutput',None)if _output_factory is None else _output_factory
    unpatch=getattr(source,'unpatchify',None)
    if factory is None or(vae.config.patch_size is not None and unpatch is None):
        raise ValueError('Official native decoder output/patch contract unavailable')
    store=FeatureStore(activation_store)
    expected=[1024,1024,1024,512,256]if _expected_channels is None else _expected_channels
    active=ContextVar('native_decode_invocation',default=())
    lock=threading.Lock()
    record=dict(decodes=0,active_calls=0,max_active_calls=0,max_nested_depth=0,
                local_cache=True,feature_hooks_invocation_scoped=True)
    def official(z,return_dict=True):
        return decode_with_local_cache(vae,z,return_dict,factory,unpatch)
    def decode(z):
        marker=object();previous=active.get();token=active.set((*previous,marker))
        with lock:
            record['decodes']+=1;record['active_calls']+=1
            record['max_active_calls']=max(record['max_active_calls'],record['active_calls'])
            record['max_nested_depth']=max(record['max_nested_depth'],len(previous)+1)
        features=[[]for _ in range(5)];handles=[]
        layers=[vae.decoder.mid_block,*vae.decoder.up_blocks]
        try:
            if len(layers)!=5:raise ValueError('Native decoder must have five feature levels')
            for i,layer in enumerate(layers):
                def collect(module,args,value,i=i):
                    if active.get()and active.get()[-1]is marker:
                        features[i].append(store.copy(value))
                handles.append(layer.register_forward_hook(collect))
            rgb=vae.decode(z*wrapper.std+wrapper.mean).sample
            values=[]
            for parts in features:
                values.append(store.concatenate(parts));parts.clear()
            if [v.shape[1]for v in values]!=expected:raise ValueError('Native feature channels changed')
            return(rgb,*values)
        finally:
            for handle in handles:handle.remove()
            active.reset(token)
            with lock:record['active_calls']-=1
            # No shared cache is used or cleared here: clearing during another
            # decode would reintroduce exactly the instance-state hazard.
    depth=model.depth.forward
    def placed(z,features,geometry,view):return depth(z,DeviceFeatures(features,z.device),geometry,view)
    with patch.object(vae,'_decode',official),patch.object(wrapper,'_decode',decode),patch.object(model.depth,'forward',placed):
        yield store,record
