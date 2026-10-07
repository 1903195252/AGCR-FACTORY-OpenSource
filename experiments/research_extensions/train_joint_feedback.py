"""Bounded matched native feedback training, admitted only after a real backward.

Detached/full arms use identical weights, TRAIN-family stream, 100 Euler calls,
image losses, paired evidence and state replay. Only the decoded loss's upstream
condition/score-function gradient differs. This job is never queued on import.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager, nullcontext

import numpy as np
# Versioned memory allocation only; this standalone worker never shares CUDA
# state with the preserved adapter jobs or original training processes.
os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
os.environ.pop('PYTORCH_CUDA_ALLOC_CONF',None)
import torch

from .joint_feedback_candidate import (SCHEMA,ARMS,load_joint_step_gate,parameter_scope,
    joint_backward,validate_joint_payload,validate_joint_tensors)
from .region_probe import atomic_json,sha256,validate_renderer_weights,install_state_overlay
from .train_condition_renderer import fixed_future,forward_scopes
from .train_candidate import (locked_train_branches,branch_index,atomic_checkpoint,
    capture_rng,restore_rng,gradient_metrics,parent_read_only_guard,supervise)


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('paths','checkpoint','state-checkpoint','future-checkpoint','panel',
                 'native-probe-report','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--steps',type=int,choices=(8,16),required=True)
    p.add_argument('--seed',type=int,default=17033)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--depth-device')
    p.add_argument('--vae-device')
    p.add_argument('--max-seconds',type=float,required=True)
    p.add_argument('--memory-fraction',type=float,default=.9)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--stop-file');p.add_argument('--resume',action='store_true')
    p.add_argument('--worker-report',help=argparse.SUPPRESS)
    p.add_argument('--watchdog-stop',help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if (a.seed!=17033 or not 60<=a.max_seconds<=3600 or not 0<a.memory_fraction<=.97
            or not 1<=a.threads<=8 or not a.device.startswith('cuda:') or not a.device[5:].isdigit()
            or bool(a.worker_report)!=bool(a.watchdog_stop)):
        p.error('Fixed seed17033, one explicit CUDA device, <=3600s, <=.97 memory, <=8threads required')
    if a.depth_device and (a.depth_device==a.device or not a.depth_device.startswith('cuda:')or not a.depth_device[5:].isdigit()):
        p.error('Depth device must be a different, explicitly reserved CUDA device')
    if a.vae_device and (not a.depth_device or a.vae_device in(a.device,a.depth_device)
            or not a.vae_device.startswith('cuda:')or not a.vae_device[5:].isdigit()):
        p.error('VAE decoder requires a third, distinct, explicitly reserved CUDA device')
    return a


def training_families(a,paths):
    panel=locked_train_branches(a.panel,paths['families'])
    names={r['family_id'] for r in panel['branches'] if r['cohort']=='evidence'}
    index=json.loads(Path(paths['families']).read_text())
    families=sorted((r for r in index['families'] if r['family_id'] in names),key=lambda r:r['family_id'])
    if not families or any(r['split']!='train' or r['cohort']!='evidence' for r in families):
        raise ValueError('Paired decoded-evidence diagnostic requires informative TRAIN families')
    return panel,index,families


def validate_resume(payload,identity):
    step=payload.get('step');target=identity['steps']
    if (payload.get('schema')!=SCHEMA or payload.get('identity')!=identity
            or type(step)is not int or not 0<=step<=target or payload.get('cursor')!=step
            or payload.get('target_steps')!=target or payload.get('completed')is not(step==target)
            or not {'model','optimizer','rng','baseline'}<=set(payload)):
        raise ValueError('Joint resume configuration, provenance or committed step changed')


RECOMPUTE_SCOPE='decode_video_readout_and_per_view_backward_recomputation'
JOINT_GRAPH_SCOPE='joint_forward_decode_readout_and_per_view_backward'
JOINT_GRAPH_POLICY='joint_forward_and_reentrant_view_saved_tensor_offload'
RECOMPUTE_CAPACITY='shared_128GiB_including_features_with16GiB_reserve'
CAPACITY192_POLICY='shared_192GiB_including_features_with32GiB_reserve'
RECOMPUTE_SOURCES=('probe_joint_adam_recompute_offload.py','feedback_recompute_offload.py',
    'feedback_checkpoint_boundaries.py','feedback_reentrant_boundaries.py',
    'vae_decode_local_cache.py','depth_device_residency.py',
    'decode_activation_offload.py','vae_feature_residency.py')
JOINT_GRAPH_SOURCES=tuple(name for name in RECOMPUTE_SOURCES
    if name!='probe_joint_adam_recompute_offload.py')+('joint_graph_offload.py','probe_joint_adam_joint_graph_offload.py')
THREE_DEVICE_SOURCES=tuple(name for name in JOINT_GRAPH_SOURCES
    if name!='probe_joint_adam_joint_graph_offload.py')+('probe_joint_adam_three_device.py','vae_decoder_device_residency.py')
SAVED_LAYOUT_POLICY='exact_storage_span_shape_stride_offset_v1'
FIXED_STORAGE_POLICY='registered_and_plain_encoder_storages_refreshed_after_moves'
THREE_DEVICE_LAYOUT_SOURCES=tuple(name for name in THREE_DEVICE_SOURCES
    if name!='probe_joint_adam_three_device.py')+('probe_joint_adam_three_device_layout.py','layout_saved_activation.py','residency_fixed_storage.py')
THREE_DEVICE_LAYOUT192_SOURCES=tuple(name for name in THREE_DEVICE_LAYOUT_SOURCES
    if name not in('probe_joint_adam_three_device_layout.py','joint_graph_offload.py','feedback_recompute_offload.py'))+(
        'probe_joint_adam_three_device_layout192.py','feedback_capacity192.py')


def execution_recipe(native):
    """Resolve the execution recipe after the full Adam/receipt/source gate.

    A small-device allowance is specific to the successfully measured inner
    recomputation hook. A legacy checkpoint-policy name alone never grants it.
    """
    offload=native.get('execution_activation_offload',dict(enabled=False))
    depth=native.get('execution_depth_residency',dict(enabled=False))
    vae=native.get('execution_vae_residency',dict(enabled=False))
    three_device=vae.get('enabled')is True
    boundaries=native.get('execution_feedback_checkpoints',{})
    saved_layout=offload.get('saved_tensor_layout')
    if saved_layout is not None and (saved_layout!=SAVED_LAYOUT_POLICY or not three_device):
        raise ValueError('Saved activation layout requires its measured three-device recipe')
    fixed_registry=offload.get('saved_parameter_registry')
    if (saved_layout and fixed_registry!=FIXED_STORAGE_POLICY)or(not saved_layout and fixed_registry is not None):
        raise ValueError('Saved-layout execution requires its matching fixed-storage registry')
    capacity192=offload.get('cpu_capacity_policy')==CAPACITY192_POLICY
    if capacity192 and (not saved_layout or type(offload.get('max_gib'))not in(int,float)or offload['max_gib']!=192):
        raise ValueError('The independent192GiB policy requires exact layout and its full capacity')
    joint_graph=offload.get('scope')==JOINT_GRAPH_SCOPE
    recompute=joint_graph or offload.get('scope')==RECOMPUTE_SCOPE
    offload64=offload.get('cpu_capacity_policy')=='independent_64GiB_same_kernel'
    if depth.get('enabled') and (depth.get('scope')!='frozen_depth_only'
            or depth.get('gpu_count')!=(3 if three_device else 2) or depth.get('dtype_preserved')is not True
            or depth.get('input_gradients_preserved')is not True
            or not str(depth.get('device','')).startswith('cuda:')):
        raise ValueError('Unknown secondary depth execution admission')
    if three_device:
        expected_vae=dict(enabled=True,device=vae.get('device'),gpu_count=3,
            scope='frozen_decoder_and_post_quant_only',dtype_preserved=True,
            input_gradients_preserved=True,encoder_device_unchanged=True,
            scaling_device_unchanged=True,output_device='original_latent_device')
        vae_device=vae.get('device')
        if (not joint_graph or depth.get('enabled')is not True
                or any(type(vae.get(k))is not type(v)or vae.get(k)!=v for k,v in expected_vae.items())
                or not isinstance(vae_device,str)or not vae_device.startswith('cuda:')
                or not vae_device[5:].isdigit()or vae_device==depth.get('device')):
            raise ValueError('Three-device execution needs the exact frozen VAE decoder placement')
        records=native.get('vae_residency_records',[])
        expected_record=dict(decoder_device=vae_device,scope='frozen_vae_decoder_and_post_quant_only',
            parameter_objects_preserved=True,encoder_and_normalization_unchanged=True,
            dtype_changed=False,gradients_detached=False)
        if (not isinstance(records,list)or not records or any(
                not isinstance(r,dict)or any(type(r.get(k))is not type(v)or r.get(k)!=v for k,v in expected_record.items())
                or type(r.get('decode_calls'))is not int or r['decode_calls']<=0 for r in records)):
            raise ValueError('Native report did not execute and restore the frozen VAE decoder placement')
        if saved_layout:
            records=native.get('cpu_activation_records',[])
            expected_layout=dict(saved_tensor_layout_policy=SAVED_LAYOUT_POLICY,
                stride_preserved=True,storage_offset_preserved=True,span_accounted=True,dtype_changed=False)
            if (not isinstance(records,list)or not records or any(
                    not isinstance(r,dict)
                    or any(type(r.get(k))is not type(v)or r.get(k)!=v for k,v in expected_layout.items())
                    or any(type(r.get(k))is not int or r[k]<=0 for k in
                        ('layout_offloaded_saves','layout_restored_tensors'))
                    or type(r.get('noncontiguous_offloaded_saves'))is not int
                    or not 0<=r['noncontiguous_offloaded_saves']<=r['layout_offloaded_saves']
                    for r in records)
                    or not any(r['noncontiguous_offloaded_saves']>0 for r in records)):
                raise ValueError('Native report lacks executed exact-layout activation saving and restoration')
            if capacity192 and any(type(r.get('container_before',{}).get('available_after_clean_cache_reclaim'))is not int
                    or r['container_before']['available_after_clean_cache_reclaim']<224*2**30 for r in records):
                raise ValueError('Native192GiB execution did not demonstrate its32GiB container reserve')
            records=native.get('fixed_storage_records',[])
            expected_fixed=dict(policy=FIXED_STORAGE_POLICY,activation_tensors_registered=False,
                gradients_detached=False,weights_modified=False,move_hooks_restored=True)
            if (not isinstance(records,list)or not records or any(
                    not isinstance(r,dict)
                    or any(type(r.get(k))is not type(v)or r.get(k)!=v for k,v in expected_fixed.items())
                    or any(type(r.get(k))is not int or r[k]<=0 for k in
                        ('refreshes','move_calls','tracked_tensors','unique_storages'))
                    or type(r.get('move_errors'))is not int or r['move_errors']!=0
                    for r in records)):
                raise ValueError('Native report lacks executed and restored fixed-storage registry refreshes')
    elif vae!=dict(enabled=False):
        raise ValueError('Unknown VAE decoder execution admission')
    if boundaries:
        policy=boundaries.get('policy')
        if boundaries!=dict(policy=policy,leaf_checkpoints_preserved=True) or policy not in(
                'leaf_checkpoints_without_composite_feedback_boundaries',
                'tensor_explicit_reentrant_view_direct_camera',JOINT_GRAPH_POLICY):
            raise ValueError('Unknown feedback checkpoint execution admission')
    if recompute:
        expected=dict(enabled=True,scope=JOINT_GRAPH_SCOPE if joint_graph else RECOMPUTE_SCOPE,dtype_preserved=True,
            model_parameters_retained_on_gpu=True,native_feature_residency=True,
            native_feature_depth_transfer='per_level',cpu_limit_includes_native_features=True,
            invocation_local_vae_cache=True,invocation_scoped_feature_hooks=True,
            cpu_capacity_policy=CAPACITY192_POLICY if capacity192 else RECOMPUTE_CAPACITY,
            recomputation_hook_installed_inside_per_view=True)
        if joint_graph:expected['joint_forward_scope']=True
        admission=dict(minimum_device_gib=30,original_entrypoint_default_minimum_device_gib=64,
            physical_properties_unmodified=True,expected_physical_gpu_count=3 if three_device else 2)
        if (any(type(offload.get(k))is not type(v)or offload.get(k)!=v for k,v in expected.items())
                or type(offload.get('max_gib'))not in(int,float)or not 0<offload['max_gib']<=(192 if capacity192 else 128)
                or native.get('execution_device_admission')!=admission
                or depth.get('enabled')is not True
                or boundaries!=dict(policy=JOINT_GRAPH_POLICY if joint_graph else'tensor_explicit_reentrant_view_direct_camera',leaf_checkpoints_preserved=True)):
            raise ValueError('Recomputation admission recipe is incomplete or inconsistent')
        # Require a real successful one-step report even if this helper is used
        # independently of load_joint_step_gate in a future entrypoint.
        if any(native.get(k)is not True for k in('completed','execution_completed','gradient_route_passed',
                'all_parameter_gradients_finite','frozen_parameter_versions_unchanged')) or type(native.get('optimizer_steps'))is not int or native['optimizer_steps']!=1:
            raise ValueError('Recomputation recipe needs a successful actual Adam update')
        groups=native.get('optimizer_gradient',{}).get('groups',{})
        if any(type(groups.get(name,{}).get('l2'))not in(int,float)
                or not math.isfinite(groups[name]['l2'])or groups[name]['l2']<=0
                for name in('upstream','new')):
            raise ValueError('Both joint parameter groups need finite nonzero native gradients')
        required_sources=(THREE_DEVICE_LAYOUT192_SOURCES if capacity192 else THREE_DEVICE_LAYOUT_SOURCES if saved_layout else THREE_DEVICE_SOURCES)if three_device else JOINT_GRAPH_SOURCES if joint_graph else RECOMPUTE_SOURCES
        if not set(required_sources)<=set(native.get('source_sha256',{})):
            raise ValueError('Recomputation admission implementation provenance incomplete')
        records=native.get('feedback_checkpoint_records',[])
        if not records or any(r.get('policy')!='reentrant_view_with_inner_saved_tensor_offload'
                or r.get('leaf_checkpoints_preserved')is not True or r.get('inputs_detached')is not False
                or r.get('sentinel_changes_values')is not False or r.get('per_view_grad_calls',0)<=0
                or r.get('per_view_no_grad_calls',0)<=0 or r.get('per_view_grad_offloaded_saves',0)<=0 for r in records):
            raise ValueError('Native backward did not demonstrate the admitted inner saved-tensor hook')
        if joint_graph and any(r.get('joint_forward_scope')is not True
                or r.get('joint_forward_operations_unchanged')is not True for r in records):
            raise ValueError('Native report did not demonstrate the admitted joint forward scope')
        minimum=30
    else:
        if boundaries.get('policy')==JOINT_GRAPH_POLICY:
            raise ValueError('Joint graph checkpoint policy requires its paired offload scope')
        if native.get('execution_device_admission'):
            raise ValueError('Small-device admission is restricted to the measured recomputation recipe')
        if offload.get('enabled') and (offload.get('scope')!='decode_and_video_readout_only'
                or type(offload.get('max_gib'))not in(int,float)
                or not 0<offload['max_gib']<=(64 if offload64 else 32)
                or offload.get('dtype_preserved')is not True):
            raise ValueError('Unknown native activation execution admission')
        minimum=64
    if offload.get('native_feature_residency') and (not offload.get('enabled')
            or offload.get('native_feature_depth_transfer')!='per_level'
            or offload.get('cpu_limit_includes_native_features')is not True):
        raise ValueError('Unknown native feature execution admission')
    if offload.get('invocation_local_vae_cache') and (not offload.get('native_feature_residency')
            or offload.get('invocation_scoped_feature_hooks')is not True):
        raise ValueError('Invocation-local VAE execution needs isolated native feature hooks')
    sources=set()
    if recompute:sources.update((THREE_DEVICE_LAYOUT192_SOURCES if capacity192 else THREE_DEVICE_LAYOUT_SOURCES if saved_layout else THREE_DEVICE_SOURCES)if three_device else JOINT_GRAPH_SOURCES if joint_graph else RECOMPUTE_SOURCES)
    else:
        if offload.get('enabled'):
            sources.add('decode_activation_offload.py')
            if offload64:sources.add('decode_activation_offload64.py')
        if boundaries:
            sources.add('feedback_checkpoint_boundaries.py')
            if boundaries['policy']=='tensor_explicit_reentrant_view_direct_camera':sources.add('feedback_reentrant_boundaries.py')
    if depth.get('enabled'):sources.add('depth_device_residency.py')
    if offload.get('native_feature_residency'):sources.add('vae_feature_residency.py')
    if offload.get('invocation_local_vae_cache'):sources.add('vae_decode_local_cache.py')
    return dict(offload=offload,depth=depth,vae=vae,three_device=three_device,saved_layout=saved_layout,capacity192=capacity192,
                boundaries=boundaries,recompute=recompute,joint_graph=joint_graph,
                offload64=offload64,minimum_device_gib=minimum,sources=sorted(sources))


def execution_devices(a,recipe):
    """Bind logical CLI devices to the measured recipe before resource use."""
    depth,vae=recipe['depth'],recipe.get('vae',dict(enabled=False))
    depth_device=getattr(a,'depth_device',None);vae_device=getattr(a,'vae_device',None)
    if ((depth.get('enabled')and depth_device!=depth.get('device'))
            or (not depth.get('enabled')and depth_device)
            or (vae.get('enabled')and vae_device!=vae.get('device'))
            or (not vae.get('enabled')and vae_device)):
        raise ValueError('Explicit device placement differs from native execution admission')
    devices=[a.device]+([depth_device]if depth.get('enabled')else[])+([vae_device]if vae.get('enabled')else[])
    if len(set(devices))!=len(devices)or any(not isinstance(d,str)or not d.startswith('cuda:')or not d[5:].isdigit()for d in devices):
        raise ValueError('Distinct, explicitly reserved logical CUDA devices are required')
    if len({int(d[5:])for d in devices})!=len(devices):
        raise ValueError('Logical CUDA aliases cannot reserve the same device twice')
    return devices


@contextmanager
def joint_execution_scope(model,recipe,depth_device,vae_device=None,*,_test_cpu=False,_module=None):
    """One boundary dispatcher; no legacy reentrant patch can shadow the new hook."""
    offload=recipe['offload'];boundaries=recipe['boundaries']
    if recipe['recompute']:
        if recipe.get('capacity192'):
            from .feedback_capacity192 import joint_graph_saved_tensors_on_cpu192 as activate
        elif recipe['joint_graph']:
            from .joint_graph_offload import joint_graph_saved_tensors_on_cpu as activate
        else:
            from .feedback_recompute_offload import recompute_saved_tensors_on_cpu as activate
        activation_scope=activate(model,offload['max_gib'],_test_cpu=_test_cpu,_module=_module)
    elif offload.get('enabled'):
        if recipe['offload64']:
            from .decode_activation_offload64 import decode_saved_tensors_on_cpu64 as activate
        else:
            from .decode_activation_offload import decode_saved_tensors_on_cpu as activate
        activation_scope=activate(model,offload['max_gib'])
    else:activation_scope=nullcontext(None)
    if recipe.get('saved_layout'):
        from unittest.mock import patch
        if recipe.get('capacity192'):
            from . import feedback_capacity192 as offload_module
        else:
            from . import feedback_recompute_offload as offload_module
        from .layout_saved_activation import LayoutActivationStore
        layout_scope=patch.object(offload_module,'ActivationStore',LayoutActivationStore)
    else:layout_scope=nullcontext()
    with layout_scope,activation_scope as value,ExitStack()as fixed_scope:
        store,inner_record=value if recipe['recompute']else(value,None)
        if recipe.get('saved_layout'):
            from .residency_fixed_storage import track_fixed_parameter_storages
            fixed_record=fixed_scope.enter_context(track_fixed_parameter_storages(model,store))
        else:fixed_record=None
        if offload.get('invocation_local_vae_cache'):
            from .vae_decode_local_cache import reentrant_native_features
            feature_scope=reentrant_native_features(model,store)
        elif offload.get('native_feature_residency'):
            from .vae_feature_residency import native_features_on_cpu
            feature_scope=native_features_on_cpu(model,store)
        else:feature_scope=nullcontext(None)
        with feature_scope as feature_value:
            features,cache_record=feature_value if offload.get('invocation_local_vae_cache')else(feature_value,None)
            if recipe['depth'].get('enabled'):
                from .depth_device_residency import frozen_depth_on_device
                depth_scope=frozen_depth_on_device(model,depth_device)
            else:depth_scope=nullcontext(None)
            with depth_scope as placed:
                if recipe.get('vae',{}).get('enabled'):
                    from .vae_decoder_device_residency import frozen_vae_decoder_on_device
                    vae_scope=frozen_vae_decoder_on_device(model,vae_device)
                else:vae_scope=nullcontext(None)
                with vae_scope as vae_placed:
                    if store is not None and not recipe.get('saved_layout'):
                        from .decode_activation_offload import storage_key
                        store.fixed={storage_key(p)for p in list(model.parameters())+list(model.buffers())}
                    if boundaries and not recipe['recompute']:
                        if boundaries['policy']=='tensor_explicit_reentrant_view_direct_camera':
                            from .feedback_reentrant_boundaries import reentrant_feedback_boundaries as boundary_scope
                        else:
                            from .feedback_checkpoint_boundaries import flat_feedback_boundaries as boundary_scope
                        boundary_scope=boundary_scope()
                    else:boundary_scope=nullcontext(inner_record)
                    with boundary_scope as boundary_record:
                        yield dict(store=store,features=features,cache_record=cache_record,
                                   placed=placed,vae_placed=vae_placed,boundary_record=boundary_record,
                                   fixed_record=fixed_record)


def worker(a):
    from experiments.base48.model import Base48System
    from experiments.paired_interventions.data import load_branch,to_device
    from experiments.paired_interventions.train import import_weights,export_weights
    root=Path(a.output);stop=[False];step=cursor=0;baseline=0.
    coverage=dict(active_steps=0,hook_nonzero_steps={k:0 for k in ('predicted_future_state',
        'history_scene','history_track_tokens','history_initial_state')})
    report=dict(state='initializing',completed=False,diagnostic_only=True,step=0,target_steps=a.steps,
                arm=a.arm,started_unix=time.time(),formal_experiment_result=False)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def status(stage,**kw):
        report.update(stage=stage,updated_unix=time.time(),**kw)
        atomic_json(root/'status.json',report);atomic_json(a.worker_report,report)
    def stopped():
        return stop[0] or Path(a.watchdog_stop).exists() or (a.stop_file and Path(a.stop_file).exists())
    try:
        torch.set_num_threads(a.threads);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
        device=torch.device(a.device);torch.cuda.set_device(device)
        paths=json.loads(Path(a.paths).read_text());data,index,families=training_families(a,paths)
        weights={k:sha256(v)for k,v in dict(renderer_checkpoint=a.checkpoint,
            state_checkpoint=a.state_checkpoint,future_checkpoint=a.future_checkpoint).items()}
        native=load_joint_step_gate(a.native_probe_report,weights)
        recipe=execution_recipe(native)
        execution_devices(a,recipe)
        depth_residency=recipe['depth'];offload=recipe['offload'];boundaries=recipe['boundaries']
        minimum=recipe['minimum_device_gib']
        if torch.cuda.get_device_properties(device).total_memory<minimum*2**30:
            raise ValueError('Primary device is below the successfully admitted execution floor')
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction,device)
        if depth_residency.get('enabled'):
            if a.depth_device!=depth_residency.get('device'):
                raise ValueError('Joint execution needs the same explicitly reserved secondary depth device')
            if torch.cuda.get_device_properties(a.depth_device).total_memory<minimum*2**30:
                raise ValueError('Secondary depth device is below the admitted execution floor')
            torch.cuda.set_per_process_memory_fraction(a.memory_fraction,a.depth_device)
        elif a.depth_device:
            raise ValueError('Secondary device was not admitted by the native gate')
        if recipe['vae'].get('enabled'):
            if torch.cuda.get_device_properties(a.vae_device).total_memory<minimum*2**30:
                raise ValueError('VAE decoder device is below the admitted execution floor')
            torch.cuda.set_per_process_memory_fraction(a.memory_fraction,a.vae_device)
        sources={n:sha256(Path(__file__).with_name(n))for n in ('train_joint_feedback.py',
            'joint_feedback_candidate.py','joint_parameter_scope.py','history_feedback_bridge.py','execution.py',
            'train_condition_renderer.py','condition_omega_switches.py','geometry_alignment_repair.py')}
        identity=dict(schema=SCHEMA+'-training',arm=a.arm,steps=a.steps,seed=a.seed,
            ancestry={k+'_sha256':v for k,v in weights.items()},index_sha256=data['index_sha256'],panel_sha256=data['panel_sha256'],
            native_gate_sha256=sha256(a.native_probe_report),source_sha256=sources,
            families=[r['family_id'] for r in families],family_stream='seeded_train_evidence_epoch_permutation',
            feedback_world_policy='update_modulo_2',
            flow_depth_unchanged=True,state_replay=True,paired_evidence_weight=.1,
            upstream_trainable=True,future_truth_video_conditions=False,
            neutralize_geometry_omega=True,neutralize_renderer_omega=True,
            decoded_feedback_gradient='stopped'if a.arm=='detached_feedback'else'history_and_state',
            losses_identical_across_arms=True,sampling_steps=50,camera_paths=2,gradient_tail_steps=2,
            parameter_scope='upstream_condition_adapters_gate',frozen_lora_depth_reader=True,
            lr_upstream=1e-5,lr_new=1e-4,weight_decay=.01,betas=[.9,.999],clip_norm=1.,
            execution='sequential_graphs',precision='unchanged_native',test_used=False,
            diagnostic_only=True,formal_experiment_result=False)
        identity['execution_activation_offload']=offload
        identity['execution_depth_residency']=depth_residency
        if recipe['vae'].get('enabled'):
            identity['execution_vae_residency']=recipe['vae']
        identity['execution_feedback_checkpoints']=boundaries
        identity['execution_device_admission']=native.get('execution_device_admission',
            dict(minimum_device_gib=64,expected_physical_gpu_count=2 if depth_residency.get('enabled')else 1))
        for name in recipe['sources']:
            identity['source_sha256'][name]=sha256(Path(__file__).with_name(name))
        status('loading_native_model')
        payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        if not payload.get('completed')or payload['identity']['phase']!='video':
            raise ValueError('Completed native renderer required')
        model=Base48System(paths,device,'video',payload['identity'].get('variant_options'))
        identity['evidence_configuration']={key:model.variant[key]for key in
            ('evidence_score_reduction','evidence_loss_scales','evidence_bandwidth','evidence_draws')}
        validate_renderer_weights(model,payload);import_weights(model,payload['model']);del payload
        install_state_overlay(model,a.state_checkpoint,expected_index_sha256=data['index_sha256'])
        fixed_future(a.future_checkpoint,a.state_checkpoint,data['index_sha256'],model.upstream,device)
        generator=torch.Generator(device=device).manual_seed(a.seed+100003)
        with parameter_scope(model)as groups,forward_scopes(True,True):
            # Report moment storage as well as the measured real-Adam native
            # admission; each matched arm initializes its own fresh optimizer.
            optimizer=torch.optim.AdamW(groups,betas=(.9,.999),weight_decay=.01)
            report['optimizer_moments_estimated_gib']=sum(p.numel()*p.element_size()*2 for g in groups for p in g['params'])/2**30
            report['trainable_parameter_counts']={g['name']:sum(p.numel()for p in g['params'])for g in groups}
            frozen={name:p._version for name,p in model.named_parameters()if not p.requires_grad}
            if a.resume:
                previous=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
                validate_resume(previous,identity)
                if json.loads((root/'identity.json').read_text())!=identity:raise ValueError('Identity changed')
                # Initializer/refiner structure was built before constructing
                # optimizer groups. Reinstalling it now would replace module
                # Parameters and orphan their optimizer references.
                validate_joint_payload(previous,a.future_checkpoint,a.state_checkpoint,data['index_sha256'],allow_partial=True,renderer_sha=weights['renderer_checkpoint'])
                validate_joint_tensors(model,previous);import_weights(model,previous['model'])
                optimizer.load_state_dict(previous['optimizer']);restore_rng(previous['rng'],generator,device)
                step=cursor=previous['step'];baseline=previous['baseline'];coverage=previous['feedback_coverage'];del previous
                frozen={name:p._version for name,p in model.named_parameters()if not p.requires_grad}
            else:atomic_json(root/'identity.json',identity)
            def save(reason):
                if frozen!={n:p._version for n,p in model.named_parameters()if not p.requires_grad}:
                    raise RuntimeError('Frozen renderer prior/reader changed')
                atomic_checkpoint(root/'last.pt',dict(schema=SCHEMA,identity=identity,
                    model=export_weights(model),optimizer=optimizer.state_dict(),
                    rng=capture_rng(generator,device),step=step,cursor=cursor,target_steps=a.steps,
                    completed=step==a.steps,baseline=baseline,feedback_coverage=coverage,
                    feedback_mechanism_admitted=coverage['active_steps']>0 and (
                        all(v>0 for v in coverage['hook_nonzero_steps'].values())if a.arm=='joint_feedback'
                        else all(v==0 for v in coverage['hook_nonzero_steps'].values())),save_reason=reason,
                    saved_unix=time.time(),diagnostic_only=True,formal_experiment_result=False))
                report['checkpoint_step']=step
            if not a.resume:save('initialized')
            status('training',state='running')
            while step<a.steps and not stopped():
                row=families[branch_index(cursor,len(families),a.seed)]
                status('loading_complete_train_family',family_id=row['family_id'],step=step)
                cells={}
                for record in row['records']:
                    path=Path(index['root'])/record['path']
                    if sha256(path)!=record['sha256']:raise ValueError('Audited TRAIN branch changed')
                    cells[(record['world'],record['camera'])]=load_branch(path)
                family=dict(metadata={k:v for k,v in row.items()if k!='records'},branches=to_device(cells,device))
                branch=family['branches'][(step%2,(step//2)%2)];branch['cohort']=row['cohort']
                optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device);started=time.monotonic()
                if a.depth_device:torch.cuda.reset_peak_memory_stats(a.depth_device)
                if a.vae_device:torch.cuda.reset_peak_memory_stats(a.vae_device)
                with joint_execution_scope(model,recipe,a.depth_device,a.vae_device)as execution:
                    parts=joint_backward(model,branch,family,generator,step,baseline,a.arm,
                        stage_callback=lambda name:status(name,step=step))
                if execution['store']is not None:parts['cpu_activation']=execution['store'].report()
                if execution['features']is not None:parts['native_features']=execution['features'].report()
                if execution['cache_record']is not None:parts['native_cache']=execution['cache_record']
                if execution['placed']is not None:parts['depth_residency']=execution['placed']
                if execution['vae_placed']is not None:parts['vae_residency']=execution['vae_placed']
                if execution['boundary_record']is not None:parts['feedback_checkpoints']=execution['boundary_record']
                if execution['fixed_record']is not None:parts['fixed_storage_registry']=execution['fixed_record']
                grads=gradient_metrics(groups);status('optimizer_step',step=step);optimizer.step()
                baseline=.95*baseline+.05*parts['baseline_observation']
                coverage['active_steps']+=int(parts['feedback']['active_feedback_supervision'])
                for name,value in parts['feedback_gradient']['hook_nonzero'].items():
                    coverage['hook_nonzero_steps'][name]+=int(value)
                torch.cuda.synchronize(device)
                if a.depth_device:torch.cuda.synchronize(a.depth_device)
                if a.vae_device:torch.cuda.synchronize(a.vae_device)
                elapsed=time.monotonic()-started
                step+=1;cursor+=1
                metrics=dict(step=step,family_id=row['family_id'],split='train',arm=a.arm,
                    losses=parts,gradients=grads,step_seconds=elapsed,
                    cumulative_feedback_coverage=dict(active_steps=coverage['active_steps'],
                        hook_nonzero_steps=dict(coverage['hook_nonzero_steps'])),
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                    updated_unix=time.time())
                if a.depth_device:
                    metrics['secondary_gpu_peak']=dict(device=a.depth_device,
                        peak_allocated_gib=torch.cuda.max_memory_allocated(a.depth_device)/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved(a.depth_device)/2**30)
                if a.vae_device:
                    metrics['vae_gpu_peak']=dict(device=a.vae_device,
                        peak_allocated_gib=torch.cuda.max_memory_allocated(a.vae_device)/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved(a.vae_device)/2**30)
                with(root/'updates.jsonl').open('a')as stream:
                    stream.write(json.dumps(metrics,allow_nan=False)+'\n');stream.flush()
                del family,cells,branch,parts
                status('saving_checkpoint',step=step,metrics=metrics);save('complete'if step==a.steps else'committed_update')
            status('complete'if step==a.steps else'paused',state='completed'if step==a.steps else'paused',
                completed=step==a.steps,step=step,finished_unix=time.time())
            return 0
    except Exception as error:
        status('failed',state='out_of_memory'if isinstance(error,torch.cuda.OutOfMemoryError)else'failed',
            completed=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            finished_unix=time.time())
        return 1


def account_joint_attempt(output,attempt,gpu_count):
    """Correct the generic one-GPU supervisor's local usage annotation only."""
    if gpu_count not in(1,2,3):raise ValueError('Unexpected admitted joint device count')
    output,attempt=Path(output),Path(attempt)
    path=attempt/'report.json';report=json.loads(path.read_text())
    report.update(allocated_gpus=gpu_count,allocated_gpu_hours=report['wall_seconds']*gpu_count/3600.,
        gpu_accounting='admitted_devices_times_worker_wall; outer shared GPU ledger is authoritative')
    atomic_json(path,report);atomic_json(output/'status.json',report)
    previous=[json.loads(p.read_text())for p in(output/'attempts').glob('*/report.json')]
    atomic_json(output/'usage.json',dict(allocated_gpu_hours=sum(p.get('allocated_gpu_hours',0.)for p in previous),
        attempts=len(previous),accounting='admitted_GPU_count_times_worker_wall_including_load_save',
        outer_shared_GPU_ledger_authoritative=True))


def main(argv=None):
    cli=list(sys.argv[1:]if argv is None else argv);a=arguments(cli)
    if a.worker_report:return worker(a)
    verified=load_joint_step_gate(a.native_probe_report,{key:sha256(path)for key,path in
        dict(renderer_checkpoint=a.checkpoint,state_checkpoint=a.state_checkpoint,future_checkpoint=a.future_checkpoint).items()})
    recipe=execution_recipe(verified)
    gpu_count=len(execution_devices(a,recipe))
    out=Path(a.output).resolve();paths=json.loads(Path(a.paths).read_text())
    parent_read_only_guard(out,paths,a.checkpoint,a.state_checkpoint,a.panel)
    for source in (Path(a.future_checkpoint).resolve().parent,Path(a.native_probe_report).resolve().parent):
        if out==source or source in out.parents or out in source.parents:raise ValueError('Output overlaps initializer/admission')
    out.mkdir(parents=True,exist_ok=True)
    with(out/'.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.resume:
            if not(out/'last.pt').is_file():raise FileNotFoundError('No resumable joint candidate')
        elif any(p.name!='.lock'for p in out.iterdir()):raise FileExistsError('Fresh joint output required')
        attempt=out/'attempts'/('%d_%d'%(time.time_ns(),os.getpid()));attempt.mkdir(parents=True)
        command=[sys.executable,'-m','experiments.research_extensions.train_joint_feedback',*cli,
            '--worker-report',str(attempt/'report.json'),'--watchdog-stop',str(attempt/'STOP')]
        code=supervise(command,out,attempt,a.max_seconds,120,a.stop_file)
        account_joint_attempt(out,attempt,gpu_count)
        return code


if __name__=='__main__':raise SystemExit(main())
