"""Place only the frozen native VAE decoder on a separately reserved device.

The latent reaches this boundary after the original mean/std transformation.
The encoder, quantization layer, and normalization tensors remain on the main
device. Existing decoder hooks still stream depth features into their bounded
CPU store; only the RGB sample returns to the latent's original device.
"""
from contextlib import contextmanager
from unittest.mock import patch

import torch

from .execution import _move_data


@contextmanager
def frozen_vae_decoder_on_device(model,device,*,_test_cpu=False):
    device=torch.device(device)
    if device.type!='cuda'and not _test_cpu:
        raise ValueError('Native VAE decoding needs a separately reserved CUDA device')
    wrapper=model.vae;vae=wrapper.vae
    if vae is not model.encoders.vae:
        raise ValueError('Native wrapper and historical encoder must share the same official VAE')
    modules=(vae.decoder,vae.post_quant_conv)
    parameters=list(dict.fromkeys(p for module in modules for p in module.parameters()))
    tensors=list(dict.fromkeys(parameters+[b for module in modules for b in module.buffers()]))
    if not parameters or any(p.requires_grad or p.grad is not None for p in parameters):
        raise ValueError('VAE residency requires frozen decoder/post-quant weights without gradients')
    # Reentrant checkpoints snapshot input-device RNG, not a newly introduced
    # remote generator. Official decoding runs in eval mode and uses no RNG.
    if any(module.training for module in modules)or any(
            isinstance(module,torch.nn.modules.dropout._DropoutNd)and module.training and module.p>0
            for root in modules for module in root.modules()):
        raise ValueError('Native frozen VAE decoder must retain its existing evaluation mode')
    protected=list(dict.fromkeys([p for module in(vae.encoder,vae.quant_conv)
                                  for p in list(module.parameters())+list(module.buffers())]))
    protected += [wrapper.mean,wrapper.std,model.encoders.mean,model.encoders.std]
    if {id(v)for v in tensors}&{id(v)for v in protected}:
        raise ValueError('Decoder residency would also relocate historical encoder tensors')
    before=[(tensor,tensor.device,tensor.dtype,tensor.requires_grad)for tensor in tensors]
    protected_before=[(tensor,tensor.device,tensor.dtype,tensor.requires_grad)for tensor in protected]
    ids=tuple(map(id,parameters));decode=vae.decode
    record=dict(decoder_device=str(device),scope='frozen_vae_decoder_and_post_quant_only',
        parameter_objects_preserved=False,encoder_and_normalization_unchanged=False,
        dtype_changed=False,gradients_detached=False,decode_calls=0,
        input_tensor_bytes=0,output_tensor_bytes=0,
        parameter_bytes=sum(p.numel()*p.element_size()for p in parameters),
        feature_output_policy='existing_native_hooks_to_bounded_cpu_store',
        normalization_before_device_copy=True,random_operations_added=False)

    def transfer(value,target):
        return value.to(device=target,copy=_test_cpu,non_blocking=False)

    def placed(z,*args,**kwargs):
        if not isinstance(z,torch.Tensor):raise TypeError('Official VAE decode requires a latent tensor')
        source=z.device
        record['decode_calls']+=1
        record['input_tensor_bytes']+=z.numel()*z.element_size()
        output=decode(transfer(z,device),*args,**kwargs)
        if isinstance(output,tuple):
            if len(output)!=1 or not isinstance(output[0],torch.Tensor):
                raise TypeError('Official VAE tuple output contract changed')
            sample=output[0]
            result=(transfer(sample,source),)
        elif isinstance(getattr(output,'sample',None),torch.Tensor):
            sample=output.sample
            result=type(output)(sample=transfer(sample,source))
        else:raise TypeError('Official VAE DecoderOutput contract changed')
        record['output_tensor_bytes']+=sample.numel()*sample.element_size()
        return result

    try:
        for tensor,_,_,_ in before:_move_data(tensor,device)
        with patch.object(vae,'decode',placed):yield record
    finally:
        for tensor,origin,dtype,requires_grad in before:
            _move_data(tensor,origin)
            if tensor.dtype!=dtype or tensor.requires_grad!=requires_grad:
                raise RuntimeError('VAE residency changed precision or gradient flags')
        record['parameter_objects_preserved']=tuple(map(id,dict.fromkeys(
            p for module in modules for p in module.parameters())))==ids
        record['encoder_and_normalization_unchanged']=all(
            tensor.device==origin and tensor.dtype==dtype and tensor.requires_grad==requires_grad
            for tensor,origin,dtype,requires_grad in protected_before)
        if not record['parameter_objects_preserved']or not record['encoder_and_normalization_unchanged']:
            raise RuntimeError('VAE residency changed decoder identity or historical encoder residency')
