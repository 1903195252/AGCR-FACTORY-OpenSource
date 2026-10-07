"""Place only the last frozen VAE block and output head on the primary GPU.

Enter inside the whole-decoder residency scope; retain both scopes throughout
backward. Block2's residuals and final upsampling remain on the VAE-prefix GPU.
The original decoder forward, per-stage cache slots and feature hooks remain
unchanged. One differentiable copy enters block3 for each temporal chunk.
"""
from contextlib import contextmanager
from unittest.mock import patch

import torch

from . import execution

POLICY = 'frozen_vae_up_block_3_and_output_head_on_primary_v1'


@contextmanager
def frozen_vae_final_block_on_device(model, device, *, _test_cpu=False):
    target = torch.device(device)
    if target.type != 'cuda' and not _test_cpu:
        raise ValueError('VAE final block requires an explicitly reserved CUDA device')
    wrapper = model.vae; vae = wrapper.vae; decoder = vae.decoder
    if vae is not model.encoders.vae:
        raise ValueError('Native wrapper and history encoder must share one VAE')
    if len(decoder.up_blocks) != 4:
        raise ValueError('Final-block split requires four native up-blocks')
    boundary = decoder.up_blocks[3]
    if (getattr(decoder.up_blocks[2], 'avg_shortcut', None) is None
            or getattr(boundary, 'avg_shortcut', None) is not None):
        raise ValueError('Block2 must retain final upsampling and block3 must have no shortcut upsample')
    modules = (boundary, decoder.norm_out, decoder.nonlinearity, decoder.conv_out)
    params = list(dict.fromkeys(p for module in modules for p in module.parameters()))
    tensors = list(dict.fromkeys(params + [b for module in modules for b in module.buffers()]))
    if not params or any(p.requires_grad or p.grad is not None for p in params):
        raise ValueError('Final-block residency requires frozen VAE parameters without gradients')
    if decoder.training or any(
            isinstance(m, torch.nn.modules.dropout._DropoutNd) and m.training and m.p > 0
            for module in modules for m in module.modules()):
        raise ValueError('Native VAE must retain deterministic evaluation mode')
    origins = {t.device for t in tensors}
    if len(origins) != 1:
        raise ValueError('Enter final-block placement after whole-decoder placement')
    source = next(iter(origins))
    if not _test_cpu and (source.type != 'cuda' or source == target):
        raise ValueError('VAE prefix and final block require distinct reserved CUDA devices')
    protected_modules = (vae.encoder, vae.quant_conv, vae.post_quant_conv, decoder.conv_in,
                         decoder.mid_block, *decoder.up_blocks[:3])
    protected = list(dict.fromkeys(t for module in protected_modules
                                   for t in list(module.parameters()) + list(module.buffers())))
    protected += list(dict.fromkeys((wrapper.mean, wrapper.std, model.encoders.mean, model.encoders.std)))
    if {id(t) for t in tensors} & {id(t) for t in protected}:
        raise ValueError('Final-block tensors alias protected prefix/encoder/scaling tensors')
    before = [(t, t.device, t.dtype, t.requires_grad) for t in tensors]
    protected_before = [(t, t.device, t.dtype, t.requires_grad) for t in protected]
    ids = tuple(map(id, params)); original = boundary.forward
    record = dict(enabled=True, policy=POLICY, first_tail_block=3,
        tail_device=str(target), source_device=str(source),
        prefix_block_indices=[0, 1, 2], moved_block_indices=[3],
        boundary_calls=0, input_tensor_bytes=0, boundary_input_shapes=[],
        tail_parameter_bytes=sum(p.numel() * p.element_size() for p in params),
        parameter_objects_preserved=False, prefix_encoder_scaling_unchanged=False,
        cache_objects_preserved=True, cache_index_starts=[], dtype_changed=False,
        gradients_detached=False, random_operations_added=False,
        original_decoder_forward_retained=True, feature_hooks_retained=True)

    def placed(value, *args, **kwargs):
        if not isinstance(value, torch.Tensor):
            raise TypeError('Native block3 boundary must receive a tensor')
        if not _test_cpu and value.device != source:
            raise ValueError('Block3 must receive the unchanged block2 output on the prefix GPU')
        cache = kwargs.get('feat_cache', args[0] if len(args) > 0 else None)
        index = kwargs.get('feat_idx', args[1] if len(args) > 1 else None)
        if index is not None and int(index[0]) not in record['cache_index_starts']:
            record['cache_index_starts'].append(int(index[0]))
        shape = list(value.shape)
        if shape not in record['boundary_input_shapes']:
            record['boundary_input_shapes'].append(shape)
        record['boundary_calls'] += 1
        record['input_tensor_bytes'] += value.numel() * value.element_size()
        transferred = value.to(device=target, copy=_test_cpu, non_blocking=False)
        if transferred.dtype != value.dtype:
            raise RuntimeError('Block3 copy changed activation precision')
        # Each block retains its own entries in the original shared cache list.
        # Prefix caches stay on source; block3/head caches are written on target.
        output = original(transferred, *args, **kwargs)
        if not isinstance(output, torch.Tensor) or output.device != target:
            raise RuntimeError('Native block3 output is on the wrong assigned device')
        if (cache is not kwargs.get('feat_cache', args[0] if len(args) > 0 else None)
                or index is not kwargs.get('feat_idx', args[1] if len(args) > 1 else None)):
            record['cache_objects_preserved'] = False
            raise RuntimeError('Final-block placement replaced a causal-cache/index container')
        return output

    try:
        # Dynamic lookup preserves the fixed-storage registry's move callbacks.
        for tensor, _, _, _ in before:
            execution._move_data(tensor, target)
        with patch.object(boundary, 'forward', placed):
            yield record
    finally:
        for tensor, origin, dtype, requires_grad in before:
            execution._move_data(tensor, origin)
            if tensor.dtype != dtype or tensor.requires_grad != requires_grad:
                raise RuntimeError('Final-block placement changed tensor dtype or gradient flags')
        record['parameter_objects_preserved'] = tuple(map(id, dict.fromkeys(
            p for module in modules for p in module.parameters()))) == ids
        record['prefix_encoder_scaling_unchanged'] = all(
            tensor.device == origin and tensor.dtype == dtype and tensor.requires_grad == requires_grad
            for tensor, origin, dtype, requires_grad in protected_before)
        if not record['parameter_objects_preserved'] or not record['prefix_encoder_scaling_unchanged']:
            raise RuntimeError('Final-block placement changed identities or protected placement')
