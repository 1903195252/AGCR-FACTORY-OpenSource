"""Whole-family DDP training, bounded runs and atomic exact-resume checkpoints.

Development runs are isolated and cannot erase the user's formal pause. Use
torchrun --nproc_per_node=2 -m experiments.paired_interventions.train ...
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .data import FamilyDataset,OrdinaryDataset,FamilySampler,to_device,file_sha256
from .system import JointSystem
from .validation import panel_indices,validate
from experiments.performance.execution_runtime import (
    execution_provenance, build_model, prepare_training_branch,
)


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def trainable_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode());h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def export_weights(model):
    # Frozen official assets are identified by provenance, never duplicated in
    # every checkpoint. All learned modules and normalization buffers are kept.
    return {k:v.detach().cpu() for k,v in model.state_dict().items()
            if not k.startswith('wan.backbone.') or 'lora_' in k}


def import_weights(model,weights):
    available=model.state_dict()
    extras=[k for k in weights if k not in available]
    if extras: raise ValueError('Unexpected learned checkpoint keys: '+str(extras[:8]))
    result=model.load_state_dict(weights,strict=False)
    missing=[k for k in result.missing_keys if not k.startswith('wan.backbone.') and not k.startswith('wan.adapters.')
             and not k.startswith('wan.gate.') and not k.startswith('wan.encoder.')]
    if missing: raise ValueError('Missing learned modules: '+str(missing[:8]))


class Objective(nn.Module):
    def __init__(self,model): super().__init__();self.model=model
    def forward(self,*args,**kwargs):
        loss,metrics=self.model(*args,**kwargs)
        return loss,{k:v.detach() if isinstance(v,torch.Tensor) else v for k,v in metrics.items()}


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',required=True);p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=['dynamics','reader','depth','video'],required=True)
    p.add_argument('--variant',default='full');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--steps',type=int);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--init');p.add_argument('--resume');p.add_argument('--development',action='store_true')
    p.add_argument('--families-only',action='store_true',help='Acceptance only; formal runs also use the original collection')
    p.add_argument('--gpu-hours',type=float,required=True);p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--force-feedback',action='store_true',help='Development gradient acceptance; never a paper schedule')
    p.add_argument('--stop-after',type=int,help='Stop without changing total scheduler length, for exact-resume checks')
    p.add_argument('--stop-file',help='Save at the next update boundary when this file exists')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--prefetch',type=int,default=2,help='Bounded CPU readers per rank; preserves committed sample order')
    p.add_argument('--validate-every',type=int,default=2000)
    p.add_argument('--validation-panel',type=int,default=64)
    p.add_argument('--execution-policy',choices=['standard','optimized_v5'],default='standard',
                   help='Versioned execution only; does not change the scientific protocol')
    return p.parse_args()


def main():
    a=arguments();start=time.monotonic()
    world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0))
    if a.batch<world or a.batch%world: raise ValueError('Batch must be a positive multiple of world size')
    if a.gpu_hours<=0: raise ValueError('Positive bounded GPU budget required')
    if a.prefetch<0 or a.prefetch>4: raise ValueError('Prefetch must be between 0 and 4 per rank')
    if (a.families_only or a.force_feedback) and not a.development: raise ValueError('Acceptance-only option in formal run')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    paths=json.loads(Path(a.paths).read_text());out=Path(a.output).resolve()
    protocol=json.loads(Path(__file__).with_name('protocol.json').read_text())
    variant=next((x for x in protocol['essential_variants'] if x['id']==a.variant),None)
    if variant is None: raise ValueError('Unknown registered variant')
    defaults=dict(dynamics=40000,reader=10000,depth=10000,video=80000)
    steps=a.steps or defaults[a.phase]
    if not a.development:
        if Path(paths['pause_marker']).exists(): raise RuntimeError('Formal training is user-paused; development checks do not resume it')
        if a.batch!=16 or steps!=defaults[a.phase]: raise ValueError('Formal schedule/batch differs from registered protocol')
    families=FamilyDataset(paths['families'],'train')
    validation=FamilyDataset(paths['families'],'val')
    panel=panel_indices(validation,a.validation_panel)
    informative=[i for i,row in enumerate(families.families) if row['cohort']=='evidence']
    if not informative: raise ValueError('No audited informative family for feedback')
    ordinary=None if a.families_only else OrdinaryDataset(paths['ordinary'])
    if not a.development and (families.index['development_only'] or ordinary.index['development_only']):
        raise ValueError('Development data cannot supply a formal training run')
    if not a.development and (len(panel)<64 or a.validate_every!=2000):
        raise ValueError('Formal training requires the fixed >=64-family panel every 2000 updates')
    if world>1: torch.distributed.init_process_group('nccl',timeout=timedelta(minutes=60))
    out.mkdir(parents=True,exist_ok=True)
    if not a.resume and (out/'last.pt').exists(): raise FileExistsError('Refusing to replace an existing run without --resume')
    if Path(paths['upstream']).resolve().parent==out: raise ValueError('Parent training run is read-only')
    source={p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')}
    identity=dict(schema='agcr-intervention-training-1',phase=a.phase,variant=a.variant,seed=a.seed,
        world_size=world,batch=a.batch,steps=steps,development_only=a.development,force_feedback=a.force_feedback,
        family_index_sha256=file_sha256(paths['families']),ordinary_index_sha256=file_sha256(paths['ordinary']) if ordinary else None,
        parent_sha256=file_sha256(paths['upstream']),statistics_sha256=file_sha256(paths['statistics']),
        protocol_sha256=file_sha256(Path(__file__).with_name('protocol.json')),
        validation_every=a.validate_every,validation_panel=[validation.families[i]['family_id'] for i in panel],
        empty_text_sha256=file_sha256(paths['empty_text']),source_sha256=source,
        precision='frozen_Wan_bf16; upstream_and_trainable_weights_fp32; depth_convolution_autocast_bf16',
        formal_experiment_results=False,paths=paths,**execution_provenance(a.execution_policy))
    previous=None
    if a.resume:
        previous=torch.load(a.resume,map_location='cpu',weights_only=False)
        if previous['identity']!=identity: raise ValueError('Resume contract changed: source, split, parameters or schedule')
    elif a.init:
        previous=torch.load(a.init,map_location='cpu',weights_only=False)
        if not a.development and (previous['identity']['development_only'] or not previous.get('completed')):
            raise ValueError('Formal training requires an accepted completed predecessor')
        if not a.development:
            expected={'depth':'dynamics','video':'depth'}.get(a.phase)
            if previous['identity']['phase']!=expected:raise ValueError('Wrong predecessor training phase')
            for key in ('variant','seed','family_index_sha256','ordinary_index_sha256','source_sha256','protocol_sha256','parent_sha256',
                        'execution_policy','execution_source_sha256','execution_acceptance_sha256'):
                if previous['identity'][key]!=identity[key]:raise ValueError('Predecessor provenance mismatch: '+key)
    if rank==0: atomic_json(out/'manifest.json',identity)
    pulse=dict(pid=os.getpid(),stage='loading_models',completed_steps=0,target_steps=steps)
    pulse_stop=threading.Event()
    def heartbeat():
        while not pulse_stop.is_set():
            atomic_json(out/'heartbeat.json',dict(pulse,updated_unix=time.time()))
            pulse_stop.wait(10)
    if rank==0: threading.Thread(target=heartbeat,daemon=True,name='training-heartbeat').start()
    model=build_model(paths,device,a.phase,variant,a.execution_policy)
    if previous: import_weights(model,previous['model'])
    groups=model.trainable_groups();optimizer=torch.optim.AdamW(groups,betas=(.9,.999),weight_decay=.01)
    fs=FamilySampler(len(families),a.seed+107);ds=FamilySampler(len(ordinary) if ordinary else len(families),a.seed+311)
    # Feedback runs only every eighth update. Reusing the ordinary update
    # cursor modulo this cohort aliases that cadence with the family count:
    # for 320 families and batch 16 it visits only 80 families. This stream
    # advances once per committed feedback batch, independently of base data.
    informative_sampler=FamilySampler(len(informative),a.seed+509)
    generator=torch.Generator(device=device).manual_seed(a.seed+100003*rank)
    step=0;baseline=0.;elapsed=0.;best={}
    if a.resume:
        optimizer.load_state_dict(previous['optimizer']);step=previous['step'];baseline=previous['baseline']
        fs=FamilySampler(**previous['family_sampler']);ds=FamilySampler(**previous['ordinary_sampler'])
        informative_sampler=FamilySampler(**previous['informative_sampler'])
        state=previous['rank_rng'][rank]
        torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'],device)
        generator.set_state(state['generator']);np.random.set_state(state['numpy']);random.setstate(state['python'])
        elapsed=previous['gpu_hours']
        best=previous.get('validation_best',{})
    objective=Objective(model)
    if world>1: objective=DDP(objective,device_ids=[local],find_unused_parameters=True,broadcast_buffers=False)
    stop=[False]
    def request_stop(*_): stop[0]=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    def checkpoint_run(completed=False,destination='last.pt'):
        rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device),generator=generator.get_state(),
                 numpy=np.random.get_state(),python=random.getstate())
        states=[None]*world
        if world>1: torch.distributed.all_gather_object(states,rng)
        else: states[0]=rng
        if rank==0:
            payload=dict(identity=identity,model=export_weights(model),optimizer=optimizer.state_dict(),step=step,
                completed=completed,baseline=baseline,validation_best=best,family_sampler=fs.state_dict(),ordinary_sampler=ds.state_dict(),
                informative_sampler=informative_sampler.state_dict(),
                rank_rng=states,gpu_hours=elapsed+world*(time.monotonic()-start)/3600,
                trainable_sha256=trainable_digest(model),
                execution_migrations=previous.get('execution_migrations',[]) if previous else [])
            temp=out/(destination+'.tmp');torch.save(payload,temp);temp.replace(out/destination)
        if world>1: torch.distributed.barrier()
    last_saved=-1
    loader=ThreadPoolExecutor(max_workers=max(a.prefetch,1),thread_name_prefix='native-data') if a.prefetch else None
    def load_micro(offset,feedback):
        family=None
        use_pair=a.families_only or (variant['paired_data'] and (ds.cursor+offset)%4==0)
        if use_pair:
            cell=(int((ds.cursor+offset)//4)%2,int((ds.cursor+offset)//8)%2)
            branch,metadata=families.branch(fs.index(offset),cell)
            family=dict(metadata=metadata)
        else: branch=ordinary[ds.index(offset)]
        if feedback:
            feedback_index=informative[informative_sampler.index(offset)]
            family=families[feedback_index]
        branch=prepare_training_branch(branch,step+1,a.phase,a.execution_policy)
        return branch,family
    try:
        while step<steps:
            requested=stop[0] or (a.stop_file is not None and Path(a.stop_file).exists()) or (time.monotonic()-start)*world/3600>=a.gpu_hours
            if a.stop_after is not None and step>=a.stop_after: requested=True
            stop_tensor=torch.tensor(int(requested),device=device)
            if world>1: torch.distributed.all_reduce(stop_tensor,op=torch.distributed.ReduceOp.MAX)
            if int(stop_tensor): break
            feedback=(a.force_feedback or (step>=40000 and (step+1)%8==0)) and a.phase=='video' and variant['paired_data']
            model.set_phase(a.phase,step>=40000 or a.force_feedback)
            rate=min((step+1)/1000,1.)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-1000)/max(steps-1000,1))))
            for group in optimizer.param_groups: group['lr']=(1e-4 if group['name']=='new' else 1e-5)*rate
            optimizer.zero_grad(set_to_none=True);metrics={};begin=time.monotonic()
            baseline_values=[]
            indices=list(range(rank,a.batch,world))
            waiting={}
            if loader:
                for micro,offset in enumerate(indices[:a.prefetch]):
                    waiting[micro]=loader.submit(load_micro,offset,feedback)
            for micro,offset in enumerate(indices):
                pulse.update(stage='feedback' if feedback else 'training',completed_steps=step,
                             microbatch=micro+1,microbatches=len(indices))
                branch,family=waiting.pop(micro).result() if loader else load_micro(offset,feedback)
                if loader and micro+a.prefetch<len(indices):
                    waiting[micro+a.prefetch]=loader.submit(load_micro,indices[micro+a.prefetch],feedback)
                branch=to_device(branch,device);family=to_device(family,device)
                sync=nullcontext() if world==1 or micro==len(indices)-1 else objective.no_sync()
                with sync:
                    loss,terms=objective(branch,generator,step+1,family,feedback,baseline)
                    finite=torch.isfinite(loss).int()
                    if world>1: torch.distributed.all_reduce(finite,op=torch.distributed.ReduceOp.MIN)
                    if not int(finite): raise FloatingPointError('Nonfinite objective; no optimizer update committed')
                    (loss/len(indices)).backward()
                for name,value in model.feedback_gradient_norms.items():
                    metrics['decoded_gradient_'+name]=metrics.get('decoded_gradient_'+name,0.)+value
                metrics['loss']=metrics.get('loss',0.)+float(loss.detach())/len(indices)
                for k,v in terms.items():
                    value=float(v)
                    if not math.isfinite(value): raise FloatingPointError('Nonfinite metric '+k)
                    metrics[k]=metrics.get(k,0.)+value/len(indices)
                if 'baseline_observation' in terms: baseline_values.append(float(terms['baseline_observation']))
                del loss,terms,branch,family
            parameters=[p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
            norm=nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            gradient_groups={name:float(sum((p.grad.float().square().sum() for p in group['params'] if p.grad is not None),torch.zeros((),device=device)).sqrt()) for name,group in ((g['name'],g) for g in optimizer.param_groups)}
            optimizer.step();step+=1;fs.advance(a.batch);ds.advance(a.batch)
            if feedback: informative_sampler.advance(a.batch)
            if baseline_values:
                value=torch.tensor([sum(baseline_values),len(baseline_values)],device=device)
                if world>1: torch.distributed.all_reduce(value)
                baseline=.99*baseline+.01*float(value[0]/value[1])
            reports=[None]*world
            if world>1: torch.distributed.all_gather_object(reports,metrics)
            else: reports[0]=metrics
            status=dict(step=step,total_steps=steps,phase=a.phase,development_only=a.development,feedback=feedback,
                execution_policy=a.execution_policy,
                pid=os.getpid(),updated_unix=time.time(),status='running',completed_steps=step,target_steps=steps,
                world_size=world,elapsed_training_seconds=(elapsed*3600/world)+(time.monotonic()-start),
                update_seconds=time.monotonic()-begin,gpu_hours=elapsed+world*(time.monotonic()-start)/3600,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(device),gradient_norm=float(norm),gradient_groups=gradient_groups,
                informative_family_draws=informative_sampler.cursor,
                metrics={k:sum(row.get(k,0) for row in reports)/world for k in set().union(*(row.keys() for row in reports))})
            if rank==0:
                atomic_json(out/'status.json',status)
                with (out/'metrics.jsonl').open('a') as log: log.write(json.dumps(status)+'\n')
                print(json.dumps(status),flush=True)
            if step%a.save_every==0:
                checkpoint_run(step==steps);last_saved=step
            if a.validate_every>0 and step%a.validate_every==0:
                pulse.update(stage='validating',completed_steps=step)
                report=validate(model,validation,panel,device,step,rank,world)
                if rank==0:
                    atomic_json(out/('validation_%06d.json'%step),report)
                    atomic_json(out/'validation_latest.json',report)
                scope=report['rows'][0]['scope']
                if scope!=best.get('scope') or report['mean']<best.get('value',float('inf')):
                    best=dict(scope=scope,value=report['mean'],step=step)
                    checkpoint_run(step==steps,'best.pt')
                    if rank==0:atomic_json(out/'best-selection.json',best)
                checkpoint_run(step==steps);last_saved=step
        if last_saved!=step: checkpoint_run(step==steps)
        if step==steps and rank==0 and (out/'best.pt').exists():
            # The selected parameters stay at their validation step. Mark the
            # enclosing training schedule complete without claiming these are
            # the last-update parameters or overwriting the last checkpoint.
            selected=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
            selected.update(completed=True,training_completed_step=steps)
            temp=out/'best.pt.tmp';torch.save(selected,temp);temp.replace(out/'best.pt')
        digest=trainable_digest(model);digests=[None]*world
        if world>1: torch.distributed.all_gather_object(digests,digest)
        else: digests[0]=digest
        if len(set(digests))!=1: raise AssertionError('DDP learned parameters diverged across ranks')
        if rank==0: atomic_json(out/'completion.json',dict(completed=step==steps,step=step,development_only=a.development,
            reason='complete' if step==steps else 'bounded_or_requested_stop',ddp_parameters_identical=True,trainable_sha256=digest))
    except Exception as error:
        if rank==0: atomic_json(out/'failure.json',dict(type=type(error).__name__,message=str(error),last_committed_step=step))
        raise
    finally:
        if loader: loader.shutdown(wait=True)
        pulse_stop.set()
        if world>1: torch.distributed.destroy_process_group()


if __name__=='__main__': main()
