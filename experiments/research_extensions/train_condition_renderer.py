"""Independent bounded TRAIN adaptation to the existing step2048 conditions.

Prepared candidate; not attached to any queue. Conditions originate from past
RGB-D. The two arms share weights/data/noise/optimizer and optional omega scopes;
only vacated-history loss support differs. No original run is resumed/rewritten.
"""
import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

from . import train_candidate as native
from . import region_objective
from .condition_omega_switches import renderer_condition
from .geometry_alignment_repair import instantaneous_geometry
from .region_probe import sha256
from .train_history_future import load_future_checkpoint
from .vacated_flow_objective import VacatedFlowSystem
from .sample_candidate import (validate_candidate as legacy_validate_candidate,
                               validate_candidate_tensors as legacy_validate_tensors)

SCHEMA = 'agcr-condition-renderer-candidate-1'
IDENTITY_SCHEMA = 'agcr-condition-renderer-training-1'
ARMS = ('condition_control', 'vacated_support')
SOURCES = ('train_condition_renderer.py', 'sample_condition_renderer.py', 'vacated_flow_objective.py')


def fixed_future(path, parent, index_sha, upstream=None, device='cpu'):
    value = load_future_checkpoint(path, parent, index_sha, upstream=upstream, device=device)
    if (value['arm'] != 'paired_geometry' or value['step'] != 2048
            or value['historical']['arm'] != 'coarse_full'):
        raise ValueError('Use the registered coarse paired-geometry step2048 overlay')
    return value


@contextmanager
def forward_scopes(geometry=False, renderer=False):
    from experiments.paper_experiments.agcr_exp.future_prediction import SurfaceGeometry
    from experiments.paired_interventions.video_modules import ConditionEncoder
    with ExitStack() as stack:
        if geometry:
            original = SurfaceGeometry.forward
            def geometry_forward(module, *a, **kw):
                with instantaneous_geometry(module):
                    return original(module, *a, **kw)
            stack.enter_context(patch.object(SurfaceGeometry, 'forward', geometry_forward))
        if renderer:
            original_encoder = ConditionEncoder.forward
            def encoder_forward(module, condition):
                return original_encoder(module, renderer_condition(condition, neutralize_omega=True))
            stack.enter_context(patch.object(ConditionEncoder, 'forward', encoder_forward))
        yield


def arguments(argv=None):
    cli = list(sys.argv[1:] if argv is None else argv)
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--future-checkpoint', required=True)
    extra.add_argument('--neutralize-geometry-omega', action='store_true')
    extra.add_argument('--neutralize-renderer-omega', action='store_true')
    supplied, remaining = extra.parse_known_args(cli)
    if '--arm' not in remaining:
        raise ValueError('Explicit candidate arm required')
    pos = remaining.index('--arm')+1
    arm = remaining[pos]
    if arm not in ARMS:
        raise ValueError('Unknown condition-renderer arm')
    remaining[pos] = 'control'
    args = native.arguments(remaining)
    args.arm = arm
    vars(args).update(vars(supplied))
    if args.steps > 128 or args.seed != 17029 or args.max_seconds > 7200:
        raise ValueError('Bounded diagnostic: <=128 updates, seed17029, <=7200s')
    return args, cli


@contextmanager
def callbacks(args):
    old_install, old_config, old_provenance = native.install_state_overlay, native.configuration, native.provenance
    old_groups = native.training_groups
    def install(model, path, expected_index_sha256=None):
        parent = old_install(model, path, expected_index_sha256)
        future = fixed_future(args.future_checkpoint, path, expected_index_sha256, model.upstream, args.device)
        model.vacated_enabled = args.arm == 'vacated_support'
        return dict(parent, future_conditioning=future, mode='frozen_history_future2048_renderer_adaptation')
    def configuration(current):
        proxy = argparse.Namespace(**vars(current)); proxy.arm = 'control'
        value = old_config(proxy)
        value.update(arm=current.arm, history_condition_unchanged=False, history_condition_source='coarse_paired_geometry2048',
            future_checkpoint_sha256=sha256(args.future_checkpoint), future_truth_conditions=False,
            sampling_steps=50, neutralize_geometry_omega=args.neutralize_geometry_omega,
            neutralize_renderer_omega=args.neutralize_renderer_omega,
            vacated_support=current.arm=='vacated_support', maximum_native_flow_weight=3.,
            official_reader_used_for_training=False, extra_wan_forward=False, upstream_frozen=True,
            geometry_auxiliary_retraining=False, reappearance_extra_loss=False)
        value.update(parameter_scope='condition_adapters_gate_only', lora_frozen=True,
                     depth_parameters_frozen=True, depth_condition_gradient_preserved=True)
        return value
    def training_groups(model):
        model.depth.requires_grad_(False)
        for name,parameter in model.wan.backbone.named_parameters():
            if 'lora_' in name:
                parameter.requires_grad_(False)
        groups = old_groups(model)
        if any(g['name'] != 'new' for g in groups):
            raise ValueError('Only condition/adapters/gate may train in this fixed candidate')
        names = {id(p):name for name,p in model.named_parameters()}
        for group in groups:
            for parameter in group['params']:
                if not names[id(parameter)].startswith(('condition.','wan.adapters.','wan.gate.','wan.encoder.')):
                    raise ValueError('Unexpected parameter in fixed adapter-only candidate: '+names[id(parameter)])
        return groups
    def provenance(*a, **kw):
        value = old_provenance(*a, **kw)
        for name in SOURCES + ('condition_omega_switches.py','geometry_alignment_repair.py'):
            path = Path(__file__).with_name(name)
            value['sources'][str(path.resolve())] = sha256(path)
        value['files']['future_checkpoint'] = dict(path=str(Path(args.future_checkpoint).resolve()), sha256=sha256(args.future_checkpoint))
        return value
    with ExitStack() as stack:
        for owner, name, value in ((native,'SCHEMA',SCHEMA),(native,'IDENTITY_SCHEMA',IDENTITY_SCHEMA),
                (native,'install_state_overlay',install),(native,'configuration',configuration),
                (native,'training_groups',training_groups),
                (native,'provenance',provenance),(region_objective,'ReappearanceSystem',VacatedFlowSystem)):
            stack.enter_context(patch.object(owner,name,value))
        stack.enter_context(forward_scopes(args.neutralize_geometry_omega,args.neutralize_renderer_omega))
        yield


def validate_candidate(payload, renderer_sha, state_sha, index_sha, future, parent, allow_partial=False):
    identity = payload.get('identity',{})
    config = identity.get('configuration',{})
    if (payload.get('schema') != SCHEMA or identity.get('schema') != IDENTITY_SCHEMA
            or config.get('arm') not in ARMS or config.get('future_truth_conditions') is not False
            or config.get('upstream_frozen') is not True or config.get('sampling_steps') != 50
            or config.get('parameter_scope') != 'condition_adapters_gate_only'
            or config.get('lora_frozen') is not True or config.get('depth_parameters_frozen') is not True
            or config.get('future_checkpoint_sha256') != sha256(future)):
        raise ValueError('Invalid condition-renderer ancestry/information boundary')
    proxy = dict(payload,schema='agcr-peerdream-direction-candidate-1',
                 identity=dict(identity,schema='agcr-peerdream-direction-training-1'))
    legacy_validate_candidate(proxy,renderer_sha,state_sha,index_sha,allow_partial)
    if identity['state_conditioning'].get('future_conditioning') != fixed_future(future,parent,index_sha):
        raise ValueError('Future overlay differs from renderer training')
    for name in SOURCES + ('condition_omega_switches.py','geometry_alignment_repair.py'):
        path = Path(__file__).with_name(name)
        if identity['provenance']['sources'].get(str(path.resolve())) != sha256(path):
            raise ValueError('Candidate source changed: '+name)
    return identity


def validate_adapter_tensors(model,payload):
    """Check frozen depth/LoRA byte values against the loaded parent renderer."""
    import torch
    legacy_validate_tensors(model,payload)
    current=model.state_dict()
    checked=0
    for name,value in payload['model'].items():
        if name.startswith('depth.') or (name.startswith('wan.backbone.') and 'lora_' in name):
            if not torch.equal(value.cpu(),current[name].detach().cpu()):
                raise ValueError('Adapter-only candidate changed frozen parent weights: '+name)
            checked+=1
    if not checked:
        raise ValueError('No frozen depth/LoRA tensors found for verification')


def main(argv=None):
    args, cli = arguments(argv)
    if args.worker_report:
        with callbacks(args):
            return native.worker(args)
    paths = json.loads(Path(args.paths).read_text())
    root = Path(args.output).resolve()
    native.parent_read_only_guard(root,paths,args.checkpoint,args.state_checkpoint,args.panel)
    source = Path(args.future_checkpoint).resolve().parent
    if root == source or source in root.parents or root in source.parents:
        raise ValueError('Candidate output overlaps frozen future checkpoint')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.resume:
            if not (root/'last.pt').is_file():
                raise FileNotFoundError('No candidate checkpoint to resume')
        elif any(p.name != '.lock' for p in root.iterdir()):
            raise FileExistsError('Use a fresh independent candidate output')
        attempt = root/'attempts'/('%d_%d'%(time.time_ns(),os.getpid()))
        attempt.mkdir(parents=True)
        command = [sys.executable,'-m','experiments.research_extensions.train_condition_renderer',*cli,
            '--worker-report',str(attempt/'report.json'),'--watchdog-stop',str(attempt/'STOP')]
        return native.supervise(command,root,attempt,args.max_seconds,args.checkpoint_grace_seconds,args.stop_file)


if __name__ == '__main__':
    raise SystemExit(main())
