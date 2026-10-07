"""One real update on the exact failed joint-training case; independent admission.

Not queued automatically. Requires a >=64 GiB device to avoid probing an
unmeasured full RGB-D feedback graph on the occupied 32 GiB sampling workers.
Uses a metadata-determined TRAIN family; no VAL/TEST selection or images.
"""
import argparse
import json
import os
import math
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('paths','renderer-checkpoint','state-checkpoint','future-checkpoint','panel','reference-initial-checkpoint','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-seconds',type=int,required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--memory-fraction',type=float,default=.9)
    p.add_argument('--neutralize-geometry-omega',action='store_true')
    p.add_argument('--neutralize-renderer-omega',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if not 60<=a.max_seconds<=1200 or not 1<=a.threads<=8 or not 0<a.memory_fraction<=.97:
        p.error('Bounded one-backward probe: 60..1200s, 1..8 threads, memory fraction <=.97')
    if not a.device.startswith('cuda:') or not a.device[5:].isdigit():
        p.error('An explicitly allocated cuda:N device is required')
    return a


def exact_tree(a,b):
    """Exact checkpoint/RNG equality without advancing any random stream."""
    import numpy as np
    import torch
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):
        return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(exact_tree(v,b[k])for k,v in a.items())
    if isinstance(a,(list,tuple)):
        return type(a)is type(b) and len(a)==len(b) and all(exact_tree(x,y)for x,y in zip(a,b))
    return type(a)is type(b) and a==b


def worker(a, *, minimum_device_gib=64):
    os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
    os.environ.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    import torch
    from experiments.base48.model import Base48System
    from experiments.paired_interventions.data import load_branch,to_device
    from experiments.paired_interventions.train import import_weights
    from .region_probe import sha256,validate_renderer_weights,install_state_overlay,atomic_json
    from .train_condition_renderer import fixed_future,forward_scopes
    from .history_feedback_bridge import feedback_gradient_report
    from .joint_parameter_scope import parameter_scope as feedback_parameter_scope
    from .joint_feedback_candidate import joint_backward
    from .train_candidate import gradient_metrics
    from experiments.paired_interventions.system import JointSystem
    out=Path(a.output);start=time.monotonic();report=dict(completed=False,optimizer_steps=0,weight_files_written=False,allocator='expandable_segments:True',memory_fraction=a.memory_fraction)
    try:
        import random
        import numpy as np
        torch.set_num_threads(a.threads);torch.manual_seed(17033);np.random.seed(17033);random.seed(17033)
        torch.cuda.set_device(a.device)
        properties=torch.cuda.get_device_properties(a.device)
        if not isinstance(minimum_device_gib, (int, float)) or not 30 <= minimum_device_gib <= 64:
            raise ValueError('Native admission device floor must be explicitly within30..64GiB')
        if properties.total_memory<minimum_device_gib*2**30:
            raise ValueError('Native full feedback preflight requires >='+str(minimum_device_gib)+'GiB')
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction,a.device)
        paths=json.loads(Path(a.paths).read_text());index=json.loads(Path(paths['families']).read_text())
        if index.get('audit_passed') is not True:
            raise ValueError('Audited paired source index required')
        from .train_joint_feedback import training_families
        from .train_candidate import branch_index
        data,index,families=training_families(a,paths)
        selected=families[branch_index(0,len(families),17033)]
        if {(r['world'],r['camera'])for r in selected['records']}!={(0,0),(0,1),(1,0),(1,1)}:
            raise ValueError('Complete TRAIN family required')
        branches={};sources={}
        for record in selected['records']:
            path=Path(index['root'])/record['path']
            if sha256(path)!=record['sha256']:raise ValueError('Audited TRAIN branch changed')
            sources[str(path)]=record['sha256']
            branches[(record['world'],record['camera'])]=load_branch(path)
        report.update(family_id=selected['family_id'],split='train',seed=17033,generator_seed=117036,
            training_cursor=0,family_stream='seeded_train_evidence_epoch_permutation',
            training_panel_sha256=data['panel_sha256'],index_sha256=data['index_sha256'],
            training_family_ids=[r['family_id']for r in families],sampling_steps=50,
            gradient_tail_steps=2,camera_paths=2,source_files=sources,
            weight_sha256={name:sha256(getattr(a,name))for name in ('renderer_checkpoint','state_checkpoint','future_checkpoint')},
            scope='One complete native base flow/depth + pure-noise feedback + evidence + state replay optimizer update; independent admission only',
            gpu=properties.name,device_memory_gib=properties.total_memory/2**30)
        payload=torch.load(a.renderer_checkpoint,map_location='cpu',weights_only=False)
        if not payload.get('completed') or payload['identity']['phase']!='video':
            raise ValueError('Completed original video renderer required')
        model=Base48System(paths,torch.device(a.device),'video',payload['identity'].get('variant_options'))
        validate_renderer_weights(model,payload);import_weights(model,payload['model']);del payload
        install_state_overlay(model,a.state_checkpoint,expected_index_sha256=sha256(paths['families']))
        fixed_future(a.future_checkpoint,a.state_checkpoint,sha256(paths['families']),model.upstream,a.device)
        from .joint_feedback_candidate import validate_joint_tensors
        reference=torch.load(a.reference_initial_checkpoint,map_location='cpu',weights_only=False)
        if(reference.get('step')!=0 or reference.get('completed')is not False
                or reference.get('identity',{}).get('seed')!=17033
                or reference['identity'].get('arm')!='joint_feedback'
                or reference['identity'].get('families')!=[r['family_id']for r in families]
                or reference['identity'].get('panel_sha256')!=data['panel_sha256']
                or reference['identity'].get('index_sha256')!=data['index_sha256']):
            raise ValueError('Exact failed initial TRAIN configuration required')
        validate_joint_tensors(model,reference)
        current=model.state_dict()
        if any(not torch.equal(value.detach().cpu(),reference['model'][name])
               for name,value in current.items()if name in reference['model']):
            raise ValueError('Native TRAIN-case initialization differs from failed run')
        report['reference_initial_checkpoint_sha256']=sha256(a.reference_initial_checkpoint)
        report['initial_model_exact_match']=True
        reference_rng=reference['rng']
        del reference,current
        family=dict(metadata={k:v for k,v in selected.items()if k!='records'},branches=to_device(branches,a.device))
        del branches
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.monotonic()
        with feedback_parameter_scope(model) as groups,forward_scopes(a.neutralize_geometry_omega,a.neutralize_renderer_omega):
            frozen={name:p._version for name,p in model.named_parameters()if not p.requires_grad}
            trainable={name:p._version for name,p in model.named_parameters()if p.requires_grad}
            report['schema']='agcr-native-joint-step-tail-admission-1'
            report['neutralize_geometry_omega']=a.neutralize_geometry_omega
            report['neutralize_renderer_omega']=a.neutralize_renderer_omega
            report['trainable_scope']='upstream_condition_adapters_gate_frozen_lora_depth_reader'
            report['trainable_parameter_counts']={g['name']:sum(p.numel()for p in g['params'])for g in groups}
            report['expected_adam_moments_gib']=sum(p.numel()*p.element_size()*2 for g in groups for p in g['params'])/2**30
            optimizer=torch.optim.AdamW(groups,betas=(.9,.999),weight_decay=.01)
            generator=torch.Generator(device=a.device).manual_seed(117036)
            from .train_candidate import capture_rng
            if not exact_tree(capture_rng(generator,a.device),reference_rng):
                raise ValueError('Native TRAIN-case RNG differs from the failed initial checkpoint')
            report['initial_rng_exact_match']=True
            del reference_rng
            branch=family['branches'][(0,0)];branch['cohort']=family['metadata']['cohort']
            def stage(name):
                report.update(stage=name,elapsed_seconds=time.monotonic()-start)
                atomic_json(out/'status.json',report)
            parts=joint_backward(model,branch,family,generator,0,0.,'joint_feedback',stage_callback=stage)
            gradients=gradient_metrics(groups)
            hooks=parts['feedback_gradient']['tensor_hooks']
            required=('predicted_future_state','history_scene','history_track_tokens','history_initial_state')
            route_passed=all(math.isfinite(hooks.get(k,0.))and hooks.get(k,0.)>0 for k in required)
            if not route_passed:raise RuntimeError('Controlled native admission requires four positive feedback hooks')
            report['optimizer_step_started']=True
            stage('optimizer_step');optimizer.step();torch.cuda.synchronize()
            report['optimizer_steps']=1
            current={name:p._version for name,p in model.named_parameters()}
            frozen_unchanged=all(current[name]==version for name,version in frozen.items())
            changed={name for name,version in trainable.items()if current[name]!=version}
            state_changed=sum(name.startswith('upstream.')for name in changed)
            renderer_changed=len(changed)-state_changed
            if not frozen_unchanged or not state_changed or not renderer_changed:
                raise RuntimeError('Actual joint optimizer did not preserve frozen modules/update both intended groups')
            report.update(completed=True,execution_completed=True,gradient_route_passed=True,
                all_parameter_gradients_finite=True,frozen_parameter_versions_unchanged=True,
                trainable_parameter_versions_changed=dict(upstream=state_changed,condition_adapters_gate=renderer_changed),
                parameter_versions_unchanged=False,weight_files_written=False,
                full_step_seconds=time.monotonic()-began,losses=parts,gradients=parts['feedback_gradient'],
                optimizer_gradient=gradients,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,future_truth_conditions=False,
                full_base_flow_depth_and_supervised_replay=True,optimizer_scope='actual joint parameter domain',
                source_sha256={name:sha256(Path(__file__).with_name(name))for name in
                    ('probe_joint_adam_training_case.py','joint_feedback_candidate.py','joint_parameter_scope.py','train_joint_feedback.py','train_candidate.py')},
                interpretation='One real optimizer admission only; weight discarded. Both later arms restart original identical parents. No generation improvement inferred.')
        atomic_json(out/'report.json',report)
        atomic_json(out/'completion.json',dict(schema=report['schema'],completed=True,execution_completed=True,
            gradient_route_passed=True,optimizer_steps=1,report_sha256=sha256(out/'report.json')))
        return 0
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
        atomic_json(out/'failure.json',report)
        raise


def main(argv=None):
    cli=list(sys.argv[1:]if argv is None else argv);a=arguments(cli);out=Path(a.output).resolve()
    for name in ('paths','renderer_checkpoint','state_checkpoint','future_checkpoint','panel','reference_initial_checkpoint'):
        source=Path(getattr(a,name)).resolve()
        if out==source or out in source.parents:raise ValueError('Output overlaps an immutable input')
    if a.worker:return worker(a)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a fresh diagnostic directory')
    out.mkdir(parents=True,exist_ok=True)
    from .sample_candidate import supervise
    return supervise([sys.executable,'-m','experiments.research_extensions.probe_joint_adam_training_case',*cli,'--worker'],out,a.max_seconds,None)


if __name__=='__main__':raise SystemExit(main())
