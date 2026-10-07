"""History-to-video joint candidate, separate from frozen-upstream adapters.

This module supplies native training steps and typed checkpoint loading. It is
not an automatically queued job. A passed native feedback report is required
before its optimizer is admitted. Future labels supply losses only.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch

from .history_feedback_bridge import feedback_gradient_report
from .region_probe import sha256
from .train_condition_renderer import fixed_future


SCHEMA = 'agcr-native-history-feedback-candidate-1'
ARMS = ('detached_feedback', 'joint_feedback')
REQUIRED_HOOKS = ('predicted_future_state', 'history_scene',
                  'history_track_tokens', 'history_initial_state')


def validate_native_gate(report, weight_sha256):
    """Execution completion alone is not an admission criterion."""
    exact = dict(completed=True, execution_completed=True, gradient_route_passed=True,
                 all_parameter_gradients_finite=True, optimizer_steps=0,
                 weight_files_written=False, parameter_versions_unchanged=True,
                 split='train', seed=17031, sampling_steps=50, gradient_tail_steps=2,
                 camera_paths=2, future_truth_conditions=False)
    if any(type(report.get(k)) is not type(v) or report.get(k) != v for k,v in exact.items()):
        raise ValueError('Native pure-noise feedback admission did not pass')
    if report.get('weight_sha256') != weight_sha256:
        raise ValueError('Native admission used different initialization weights')
    hooks = report.get('gradients', {}).get('tensor_hooks', {})
    if any(not isinstance(hooks.get(k), (int,float)) or not math.isfinite(hooks[k])
           or hooks[k] <= 0 for k in REQUIRED_HOOKS):
        raise ValueError('A decoded-feedback history/state hook is missing or invalid')
    return report


def load_native_gate(path,weight_sha256):
    """Admit only the atomic successful report and its completion receipt."""
    path=Path(path)
    report=json.loads(path.read_text())
    receipt=json.loads(path.with_name('completion.json').read_text())
    if (receipt.get('completed')is not True or receipt.get('execution_completed')is not True
            or receipt.get('gradient_route_passed')is not True
            or receipt.get('report_sha256')!=sha256(path)):
        raise ValueError('Native report lacks a matching successful completion receipt')
    return validate_native_gate(report,weight_sha256)


def load_joint_step_gate(path,weight_sha256):
    """Separate admission schema for a real optimizer update, never posing as 0."""
    path=Path(path);report=json.loads(path.read_text())
    receipt=json.loads(path.with_name('completion.json').read_text())
    expected=dict(schema='agcr-native-joint-step-admission-1',completed=True,
        execution_completed=True,gradient_route_passed=True,all_parameter_gradients_finite=True,
        optimizer_steps=1,weight_files_written=False,parameter_versions_unchanged=False,
        frozen_parameter_versions_unchanged=True,full_base_flow_depth_and_supervised_replay=True,
        split='train',seed=17031,sampling_steps=50,gradient_tail_steps=2,camera_paths=2,
        future_truth_conditions=False,
        neutralize_geometry_omega=True,neutralize_renderer_omega=True,
        trainable_scope='upstream_condition_adapters_gate_frozen_lora_depth_reader')
    if any(type(report.get(k))is not type(v)or report.get(k)!=v for k,v in expected.items()):
        raise ValueError('Actual complete native joint optimizer admission did not pass')
    if(report.get('weight_sha256')!=weight_sha256 or receipt.get('report_sha256')!=sha256(path)
            or receipt.get('schema')!=expected['schema']or receipt.get('completed')is not True
            or receipt.get('optimizer_steps')!=1):
        raise ValueError('Joint optimizer admission receipt/parent mismatch')
    changed=report.get('trainable_parameter_versions_changed',{})
    hooks=report.get('gradients',{}).get('tensor_hooks',{})
    if(not all(changed.get(k,0)>0 for k in('upstream','condition_adapters_gate'))
            or any(not math.isfinite(hooks.get(k,0.))or hooks.get(k,0.)<=0 for k in REQUIRED_HOOKS)):
        raise ValueError('Joint optimizer failed intended state/renderer/feedback updates')
    sources=report.get('source_sha256',{})
    if not {'probe_joint_adam_native.py','joint_feedback_candidate.py','joint_parameter_scope.py'}<=set(sources):
        raise ValueError('Native joint admission implementation provenance incomplete')
    for name,digest in sources.items():
        if Path(name).name!=name or sha256(Path(__file__).with_name(name))!=digest:
            raise ValueError('Native joint admission source changed: '+name)
    return report


from .joint_parameter_scope import parameter_scope


@contextmanager
def frozen_upstream_for_base(model):
    """Ordinary historical-conditioning flow/depth precedes the real feedback.

    Required by the existing memory-safe staged backward. This temporary scope
    affects only the base objective; the following feedback/replay are enabled.
    """
    original = [(p,p.requires_grad) for p in model.upstream.parameters()]
    try:
        model.upstream.requires_grad_(False)
        yield
    finally:
        for p,flag in original:
            p.requires_grad_(flag)


def state_replay_objective(model, branch, generator, update):
    """Existing observation/filter/dynamics/geometry and presence supervision.

    Reference states used inside this auxiliary supervision are never consumed
    by the separately computed pure-noise feedback or ordinary video conditions.
    """
    from experiments.base48.model import free_prefix_presence_objective
    from experiments.paired_interventions.bridge import historical_posterior
    from experiments.paired_interventions.data import replay_sample, historical_labels
    from experiments.paired_interventions.world import query_labels
    encoded = model.encoders.history(branch['inputs']['rgb'])
    history = historical_posterior(model.upstream, branch['inputs'], encoded)
    replay = replay_sample(branch, update%16)
    ids = query_labels(history, historical_labels(branch))
    auxiliary = model.encoders.features(replay['auxiliary']['rgb'])
    cached = (history['result'],history['observations'],history['shared'],history['separate'])
    loss,parts,_ = model.upstream.objective(replay,encoded,auxiliary,geometry_chunk=2,history_result=cached)
    if branch.get('cohort') == 'ambiguous' and model.variant['ambiguous_nll_only']:
        loss = loss-.1*parts['rollout_l1']
    presence,counts = free_prefix_presence_objective(model.upstream.dynamics,history,ids,
        replay['labels']['present'],replay['labels']['history_eligible'],generator)
    return loss+float(model.variant['free_prefix_presence_weight'])*presence, dict(parts,**counts,
        free_prefix_presence_BCE=presence)


def finite_backward(value, name):
    if value.ndim != 0 or not bool(torch.isfinite(value)):
        raise FloatingPointError('Nonfinite scalar '+name)
    value.backward()


def paired_feedback_objective(model,family,generator,update,baseline,arm):
    """Identical generated losses; only their upstream condition gradient differs."""
    from experiments.paired_interventions import system as original
    from experiments.paired_interventions.video_losses import reappearance_window
    JointSystem=original.JointSystem
    if arm not in ARMS:
        raise ValueError('Unregistered decoded-feedback gradient control')
    references=[]
    reference_function=original.reference_for_queries
    def loss_reference(branch,ids):
        value=reference_function(branch,ids)
        visible=value['visible'].detach();known=value['known'].detach()[None,None].expand_as(visible)
        window=reappearance_window(visible)&known
        references.append(dict(visible=visible,window_count=int(window.sum()),
                               visible_count=int((window&visible).sum())))
        return value
    with patch.dict(model.variant,dict(detach_decoded_feedback_state=arm=='detached_feedback',camera=True)),\
            patch.object(original,'reference_for_queries',loss_reference):
        value,parts=JointSystem.feedback(model,family,generator,update%2,baseline)
    counts=dict(loss_reference_camera_count=len(references),
        reappearance_mask_terms=sum(r['window_count']for r in references),
        reappearance_visible_terms=sum(r['visible_count']for r in references),
        camera_consistency_terms=int((references[0]['visible']&references[1]['visible']).sum())if len(references)==2 else 0)
    counts['active_feedback_supervision']=bool(counts['reappearance_mask_terms']or counts['camera_consistency_terms'])
    return value,dict(parts,
        feedback_upstream_detached=arm=='detached_feedback',sampling_steps=50,
        camera_paths=2,gradient_tail_steps=2,**counts)


def joint_backward(model, branch, family, generator, update, baseline,
                   arm,stage_callback=lambda _:None):
    """Accumulate one full joint update, never calling optimizer.step().

    Base flow/depth, decoded feedback/evidence, and supervised replay use
    separate graphs consumed sequentially. All gradients accumulate before the
    caller's single clipping/optimizer step. Every feedback uses both planned
    camera paths, unchanged pure noise/50 steps and the last-two-step gradient.
    """
    from .execution import staged_video_backward
    if family['metadata']['split'] != 'train' or set(family['branches']) != {(0,0),(0,1),(1,0),(1,1)}:
        raise ValueError('A complete TRAIN family is required')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Caller must zero all gradients before a complete joint update')
    with frozen_upstream_for_base(model):
        base = staged_video_backward(model,branch,generator,stage_callback=stage_callback)
    stage_callback('pure_noise_history_feedback')
    # Calling JointSystem.forward would use reference_condition in its base
    # objective. Reuse only the dedicated historical feedback method instead.
    feedback,details = paired_feedback_objective(model,family,generator,update,baseline,arm)
    finite_backward(feedback,'decoded feedback')
    report = feedback_gradient_report(model)
    hooks = report['tensor_hooks']
    if any(not math.isfinite(hooks.get(k,0.)) for k in REQUIRED_HOOKS):
        raise RuntimeError('Nonfinite native decoded-feedback history/state derivative')
    if (arm=='joint_feedback' and details['active_feedback_supervision']
            and all(k not in hooks for k in REQUIRED_HOOKS)):
        raise RuntimeError('Active decoded supervision did not touch any historical-state graph hook')
    if arm=='detached_feedback' and any(hooks.get(k,0.)!=0 for k in REQUIRED_HOOKS):
        raise RuntimeError('Detached control unexpectedly propagated decoded feedback upstream')
    report['hook_triggered']={k:k in hooks for k in REQUIRED_HOOKS}
    report['hook_nonzero']={k:hooks.get(k,0.)>0 for k in REQUIRED_HOOKS}
    feedback_value = float(feedback.detach())
    observation = float(details['baseline_observation'])
    details = {k:float(v.detach()) if isinstance(v,torch.Tensor) and v.numel()==1 else v
               for k,v in details.items() if not isinstance(v,torch.Tensor) or v.numel()==1}
    del feedback
    # Native v1 OOM occurred while constructing evidence with the completed
    # 100-Euler feedback graph still resident. Consume graphs in original RNG
    # order; both losses' gradients still accumulate into one optimizer update.
    stage_callback('paired_evidence')
    evidence,coverage=model.evidence(family,generator)
    finite_backward(.1*evidence,'paired evidence')
    evidence_value=float(evidence.detach())
    coverage={k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in coverage.items()}
    del evidence
    stage_callback('supervised_state_replay')
    replay,terms = state_replay_objective(model,branch,generator,update)
    finite_backward(replay,'supervised state replay')
    replay_value = float(replay.detach())
    terms = {k:float(v.detach()) if isinstance(v,torch.Tensor) and v.numel()==1 else v
             for k,v in terms.items() if not isinstance(v,torch.Tensor) or v.numel()==1}
    del replay
    return dict(loss=base['loss']+feedback_value+.1*evidence_value+replay_value,base=base,
        feedback=details,feedback_loss=feedback_value,state_replay_loss=replay_value,
        paired_evidence=evidence_value,evidence_coverage=coverage,
        replay=terms,feedback_gradient=report,baseline_observation=observation,
        optimizer_steps=0,future_truth_video_conditions=False,
        execution='sequential_base_flow_depth_then_native_decoded_feedback_then_state_replay')


def validate_joint_payload(payload,future_path,state_path,index_sha,allow_partial=False,renderer_sha=None):
    if payload.get('schema') != SCHEMA or (not allow_partial and payload.get('completed') is not True):
        raise ValueError('Completed typed joint candidate required for sampling')
    identity=payload.get('identity',{})
    ancestry=identity.get('ancestry',{})
    step,target=payload.get('step'),payload.get('target_steps')
    if (identity.get('arm')not in ARMS or identity.get('steps')not in(8,16)
            or identity.get('seed')!=17033 or type(step)is not int or target!=identity.get('steps')
            or not 0<=step<=target or payload.get('completed')is not(step==target)
            or identity.get('sampling_steps')!=50 or identity.get('camera_paths')!=2
            or identity.get('gradient_tail_steps')!=2):
        raise ValueError('Joint candidate update, draw or ablation contract changed')
    if (renderer_sha is None or ancestry.get('renderer_checkpoint_sha256')!=renderer_sha
            or ancestry.get('future_checkpoint_sha256') != sha256(future_path)
            or ancestry.get('state_checkpoint_sha256') != sha256(state_path)
            or identity.get('index_sha256') != index_sha
            or identity.get('future_truth_video_conditions') is not False
            or identity.get('upstream_trainable') is not True):
        raise ValueError('Joint candidate ancestry/information boundary mismatch')
    sources=identity.get('source_sha256',{})
    if not sources or any(Path(name).name!=name or sha256(Path(__file__).with_name(name))!=digest
                          for name,digest in sources.items()):
        raise ValueError('Joint candidate implementation changed')
    return identity


def validate_joint_tensors(model,payload):
    current=model.state_dict();saved=payload.get('model',{})
    expected={k for k in current if not k.startswith('wan.backbone.') or 'lora_' in k}
    if set(saved)!=expected:
        raise ValueError('Joint candidate learned-key set mismatch')
    for name,value in saved.items():
        if not isinstance(value,torch.Tensor) or value.shape!=current[name].shape or value.dtype!=current[name].dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid joint candidate tensor: '+name)
        frozen=(name.startswith(('reader.','encoders.','depth.')) or '.scale.' in name
                or (name.startswith('wan.backbone.') and 'lora_' in name))
        if frozen and not torch.equal(value.cpu(),current[name].detach().cpu()):
            raise ValueError('Joint candidate altered a frozen tensor: '+name)


def install_joint_candidate(model, payload, future_path, state_path, index_sha, allow_partial=False,renderer_sha=None):
    """Build initial structure, then import all jointly learned state weights.

    The caller loads the immutable parent renderer and state overlay first.
    There is NO fixed_future reload after the full joint import.
    """
    from experiments.paired_interventions.train import import_weights
    identity=validate_joint_payload(payload,future_path,state_path,index_sha,allow_partial,renderer_sha)
    fixed_future(future_path,state_path,index_sha,model.upstream,next(model.parameters()).device)
    validate_joint_tensors(model,payload)
    saved=payload['model']
    import_weights(model,saved)
    return identity
