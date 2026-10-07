"""Tensor-explicit reentrant per-view decoding; direct camera feedback.

Dictionary inputs/outputs are flattened and reconstructed without detaching.
This reinstates per-view recomputation without nesting the original
non-reentrant saved-tensor counters. Camera feedback executes directly, avoiding
another 100 Euler evaluations. Per-layer checkpoints are unchanged.
CPU-prepared only: native success and memory efficacy require real admission.
"""
from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch.utils._pytree import tree_flatten,tree_unflatten
from torch.utils.checkpoint import checkpoint

from .feedback_checkpoint_boundaries import BOUNDARIES


def tensor_reentrant(function,args,record):
    leaves,spec=tree_flatten(args)
    positions=[i for i,value in enumerate(leaves)if isinstance(value,torch.Tensor)]
    tensors=[leaves[i]for i in positions]
    if not tensors:raise ValueError('Composite feedback requires tensor inputs')
    floating=next((value for value in tensors if value.is_floating_point()),None)
    if floating is None:raise ValueError('Composite feedback needs a floating input')
    # Detached-control conditions still need renderer-parameter gradients.
    # This unused scalar activates reentrant backward without changing values.
    sentinel=torch.zeros((),device=floating.device,dtype=floating.dtype,requires_grad=True)
    output_contract=[]
    def calculate(*values):
        restored=list(leaves)
        for i,value in zip(positions,values[:-1]):restored[i]=value
        output=function(*tree_unflatten(restored,spec))
        flat,out_spec=tree_flatten(output)
        if not flat or any(not isinstance(value,torch.Tensor)for value in flat):
            raise TypeError('Composite feedback output must contain only tensors')
        metadata=[(tuple(value.shape),value.dtype,value.device)for value in flat]
        if not output_contract:output_contract.append((out_spec,metadata))
        elif output_contract[0]!=(out_spec,metadata):
            raise ValueError('Composite feedback output contract changed on recomputation')
        record['region_evaluations']+=1
        return tuple(flat)
    result=checkpoint(calculate,*tensors,sentinel,use_reentrant=True,preserve_rng_state=True)
    return tree_unflatten(list(result),output_contract[0][0])


@contextmanager
def reentrant_feedback_boundaries(*,_module=None):
    if _module is None:
        from experiments.paired_interventions import system as _module
    original=_module.checkpoint
    record=dict(policy='tensor_explicit_reentrant_view_direct_camera',
                boundary_calls={name:0 for name in sorted(BOUNDARIES)},region_evaluations=0,
                leaf_checkpoints_preserved=True,inputs_detached=False,
                sentinel_changes_values=False,other_module_calls_preserved=0)
    def dispatch(function,*args,**kwargs):
        name=getattr(function,'__qualname__','')
        if name not in BOUNDARIES:
            record['other_module_calls_preserved']+=1
            return original(function,*args,**kwargs)
        if kwargs!=dict(use_reentrant=False):raise ValueError('Composite checkpoint contract changed')
        record['boundary_calls'][name]+=1
        if name=='JointSystem.feedback.<locals>.generated_readout':
            return function(*args)
        return tensor_reentrant(function,args,record)
    with patch.object(_module,'checkpoint',dispatch):yield record
