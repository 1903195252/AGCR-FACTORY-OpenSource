"""Exact frozen-depth model placement on a separately allocated CUDA device.

Only tensor residency changes. Copies of latent, geometric conditions and depth
outputs remain differentiable; no reference or loss is added. This module is
off by default and requires a controller to reserve/account for both devices.
"""
from contextlib import contextmanager
from unittest.mock import patch

import torch

from .execution import _move_data


@contextmanager
def frozen_depth_on_device(model,device,*,_test_cpu=False):
    device=torch.device(device)
    if device.type!='cuda'and not _test_cpu:
        raise ValueError('Native depth residency requires a separately reserved CUDA device')
    module=model.depth;params=list(module.parameters())
    if any(p.requires_grad or p.grad is not None for p in params):
        raise ValueError('Depth placement is restricted to frozen depth weights without gradients')
    tensors=list(dict.fromkeys(params+list(module.buffers())))
    originals=[(t,t.device,t.dtype)for t in tensors]
    ids=tuple(map(id,params));forward=module.forward
    record=dict(depth_device=str(device),parameters=len(params),forward_calls=0,
                input_tensor_bytes=0,output_tensor_bytes=0,parameter_objects_preserved=False,
                dtype_changed=False,gradients_detached=False)
    def transfer(value,target):
        return value.to(device=target,copy=_test_cpu,non_blocking=False)
    class Features:
        def __init__(self,source):self.source=source
        def __iter__(self):
            for value in self.source:
                record['input_tensor_bytes']+=value.numel()*value.element_size()
                yield transfer(value,device)
    def placed(z,features,geometry,view):
        primary=z.device
        record['forward_calls']+=1
        record['input_tensor_bytes']+=sum(v.numel()*v.element_size()for v in(z,geometry))
        output=forward(transfer(z,device),Features(features),transfer(geometry,device),view)
        if not isinstance(output,dict)or any(not isinstance(v,torch.Tensor)for v in output.values()):
            raise TypeError('Original depth output must be a tensor dictionary')
        record['output_tensor_bytes']+=sum(v.numel()*v.element_size()for v in output.values())
        return{k:transfer(v,primary)for k,v in output.items()}
    try:
        for tensor,_,_ in originals:_move_data(tensor,device)
        with patch.object(module,'forward',placed):yield record
    finally:
        for tensor,origin,dtype in originals:
            _move_data(tensor,origin)
            if tensor.dtype!=dtype:raise RuntimeError('Depth placement changed precision')
        record['parameter_objects_preserved']=tuple(map(id,module.parameters()))==ids
        if not record['parameter_objects_preserved']:
            raise RuntimeError('Depth placement replaced a Parameter object')
