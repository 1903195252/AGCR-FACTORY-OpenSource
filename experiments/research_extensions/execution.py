"""Depth-then-flow backward with temporary CPU residency of the Wan backbone.

This changes execution/memory residency only. Both original loss terms retain
their resolution, precision, normalization, stochastic inputs and weights. The
caller zeroes gradients first, then clips and steps exactly once after return.
Splitting backward can change floating-point accumulation order; numerical
equivalence, rather than bitwise-identical gradients, is the contract.
"""
from contextlib import contextmanager
import time

import torch


POLICY = 'depth_then_flow_backbone_cpu'


def _move_data(tensor, device):
    """Keep Parameter objects and optimizer references intact across devices."""
    if tensor.device != torch.device(device):
        with torch.no_grad():
            tensor.data = tensor.detach().to(device=device, non_blocking=False)


def _synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def _empty_cache(device):
    if torch.device(device).type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


@contextmanager
def offload_backbone(backbone, device, *, stage_callback=None, timings=None, move_data=None):
    """Temporarily move only backbone tensors, restoring even after an error.

    A direct data transfer avoids Module._apply policies that may replace a
    Parameter object. No optimizer tensors are moved. LoRA parameters are in
    the backbone and preserve their object identity, dtype, value and flags.
    Enter only before this step has computed any backbone gradients.
    """
    move = _move_data if move_data is None else move_data
    timings = {} if timings is None else timings
    parameters = list(backbone.parameters())
    if any(parameter.grad is not None for parameter in parameters):
        raise ValueError('Zero backbone gradients with set_to_none=True before staged backward')
    tensors = list(dict.fromkeys(parameters + list(backbone.buffers())))
    original = [(tensor, tensor.device, tensor.dtype) for tensor in tensors]
    parameter_ids = tuple(id(parameter) for parameter in parameters)
    try:
        if stage_callback is not None:
            stage_callback('offload_backbone')
        _synchronize(device)
        started = time.monotonic()
        for tensor, _, _ in original:
            move(tensor, torch.device('cpu'))
        _empty_cache(device)
        timings['offload_backbone'] = time.monotonic() - started
        yield
    finally:
        # Restoration must run even if offload, forward, backward or a status
        # callback failed. A restore error propagates with the original error
        # chained by Python; a partially executed optimizer step never occurs.
        started = time.monotonic()
        try:
            if stage_callback is not None:
                stage_callback('restore_backbone')
        finally:
            for tensor, original_device, original_dtype in original:
                move(tensor, original_device)
                if tensor.dtype != original_dtype:
                    raise RuntimeError('Backbone residency transfer changed precision')
            _synchronize(device)
            timings['restore_backbone'] = time.monotonic() - started
            if tuple(id(parameter) for parameter in backbone.parameters()) != parameter_ids:
                raise RuntimeError('Backbone Parameter objects were replaced during transfer')


def _finite_scalar(value, label):
    if not isinstance(value, torch.Tensor) or value.ndim != 0 or not bool(torch.isfinite(value)):
        raise FloatingPointError('Expected finite scalar ' + label)


def _frozen(module, label):
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError('Staged video execution requires frozen ' + label)


def staged_video_backward(model, branch, generator, *, stage_callback=None):
    """Accumulate complete original video gradients without an optimizer step.

    Returns detached Python metrics, including loss, flow, depth, predicted
    tracks and timings. Do not backward the returned loss again. Historical
    inference and reference-latent encoding retain the original order. The
    depth graph is consumed before the backbone returns to the GPU and before
    constructing the flow graph. Shared condition parameters accumulate both
    gradients; the historical frontend and VAE remain frozen.
    """
    if model.base_phase != 'video':
        raise ValueError('This execution policy is restricted to the Base48 video stage')
    _frozen(model.upstream, 'historical frontend')
    _frozen(model.encoders.vae, 'VAE')
    backbone = model.wan.backbone
    if any(parameter.grad is not None for parameter in backbone.parameters()):
        raise ValueError('Zero backbone gradients with set_to_none=True before staged backward')
    if {id(p) for p in backbone.parameters()} & {id(p) for p in model.condition.parameters()}:
        raise ValueError('The condition encoder must remain outside the offloaded backbone')
    device = branch['inputs']['rgb'].device
    timings = {}
    model.profile_times = timings

    @contextmanager
    def stage(name):
        if stage_callback is not None:
            stage_callback(name)
        _synchronize(device)
        started = time.monotonic()
        try:
            yield
        finally:
            _synchronize(device)
            timings[name] = time.monotonic() - started

    with stage('history_condition'):
        _, history, _, condition = model.infer_condition(branch['inputs'], generator)
    predicted_tracks = len(history['result']['tracks'])
    # The full history dictionary is not used by either video loss. Releasing
    # its references cannot change the already constructed condition tensors.
    del history
    with stage('reference_vae_encode'):
        rgb = torch.cat([branch['inputs']['rgb'][-1:], branch['supervision']['rgb']], 0)
        latent = model.vae.encode(rgb)
    del rgb
    if latent.shape != (2, 48, 5, 30, 46):
        raise ValueError('Base48 requires unchanged native [2,48,5,30,46] video latents')
    if latent.requires_grad:
        raise ValueError('Reference VAE latent must remain frozen')

    with offload_backbone(backbone, device, stage_callback=stage_callback, timings=timings):
        with stage('clean_depth_forward'):
            depth = model.clean_depth_objective(branch, condition, latent)
            _finite_scalar(depth, 'depth objective')
            weighted_depth = float(model.variant['depth_weight']) * depth
        with stage('depth_backward'):
            weighted_depth.backward()
        depth_value = float(depth.detach())
        depth_weighted_value = float(weighted_depth.detach())
        del depth, weighted_depth
        _empty_cache(device)

    with stage('flow_forward'):
        flow = model.flow_objective(branch, condition, latent, generator)
        _finite_scalar(flow, 'flow objective')
        weighted_flow = float(model.variant['flow_weight']) * flow
    with stage('flow_backward'):
        weighted_flow.backward()
    flow_value = float(flow.detach())
    flow_weighted_value = float(weighted_flow.detach())
    del flow, weighted_flow, latent, condition
    return dict(loss=depth_weighted_value + flow_weighted_value,
                flow=flow_value, depth=depth_value, predicted_tracks=predicted_tracks,
                timings=dict(timings), execution_policy=POLICY,
                precision='unchanged_original', gradients_accumulated=True,
                optimizer_steps=0)
