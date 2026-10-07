"""Stage-two DDP with global-batch preservation and per-rank resumable RNG.

Use torch.distributed.run (one process per GPU). Single-rank execution is also
supported for numerical/throughput comparisons. Original runs are never edited.
"""
import argparse
import copy
from datetime import timedelta
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .data import Dataset, atomic_json, read_json
from .protocol import configuration
from .selection import validation_panel
from .stage2_distributed import (DistributedPrefetch, ObjectiveModule, accumulation_context,
    capture_rng, distributed_validation, gather_objects, local_batch_size, merge_training_records,
    restore_rng, state_digest, stop_on_any_rank)
from .stage2_migration import inherit_records
from .stage2_system import tensors
from .train_dynamics_geometry import build_models
from .train_observation import sha256


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'output', 'observation-checkpoint'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--statistics', default='artifacts/state_statistics.json')
    p.add_argument('--dino-code', default='third_party/dinov2')
    p.add_argument('--dino-weights', default='models/dinov2/dinov2_vitb14_pretrain.pth')
    p.add_argument('--vae', default='models/Wan2.2-TI2V-5B/vae')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=40000)
    p.add_argument('--global-batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=4, help='CPU loader workers per rank')
    p.add_argument('--prefetch', type=int, default=8, help='Queued whole clips per rank')
    p.add_argument('--geometry-chunk', type=int, default=8)
    p.add_argument('--execution-policy', choices=['standard','optimized_v1'], default='optimized_v1')
    p.add_argument('--init-checkpoint', help='Copy model/optimizer into a separate new run')
    p.add_argument('--migration-report', help='Required numerical proof for formal single-to-DDP migration')
    p.add_argument('--resume', help='Resume this DDP run, retaining world size and per-rank RNG')
    p.add_argument('--development', action='store_true')
    p.add_argument('--max-updates', type=int)
    p.add_argument('--validation-minimum', type=int, default=64)
    p.add_argument('--validation-every', type=int, default=2000)
    p.add_argument('--max-gpu-seconds', type=float, help='Session limit: wall seconds times GPU count')
    p.add_argument('--pause-file', default='artifacts/PAUSE')
    a = p.parse_args()
    if not a.development and (a.steps!=40000 or a.global_batch!=16 or a.validation_minimum!=64 or a.validation_every!=2000 or a.max_updates is not None):
        raise ValueError('Changed training protocol requires --development')
    if min(a.steps,a.global_batch,a.workers,a.prefetch,a.geometry_chunk,a.validation_minimum,a.validation_every)<1:
        raise ValueError('Nonpositive training settings')
    if a.max_updates is not None and a.max_updates < 1:
        raise ValueError('max-updates must be positive')
    if a.max_gpu_seconds is not None and a.max_gpu_seconds<=0:
        raise ValueError('GPU budget must be positive')
    if not a.development and Path(a.pause_file).exists():
        raise ValueError('Formal training is paused: '+a.pause_file)
    return a


def check_parent(previous, current, development):
    required = ['index_sha256','split_sha256','protocol_sha256','statistics_sha256',
                'dino_sha256','vae_sha256','dependency_sha256','precision']
    if not development:
        required += ['effective_batch_pairs','steps','validation_every','validation_panel']
        if previous.get('development_only',True):
            raise ValueError('Development weights cannot initialize formal training')
    for key in required:
        if previous[key] != current[key]:
            raise ValueError('Initialization changed '+key)
    if previous['execution'].get('policy','standard') != current['execution']['policy']:
        raise ValueError('DDP migration must retain the existing execution policy')


def check_migration_report(args, manifest):
    if not args.init_checkpoint or args.development:
        return
    if not args.migration_report:
        raise ValueError('Formal DDP migration requires its numerical verification report')
    proof = read_json(args.migration_report)
    required = {'global_batch16_equivalence','fixed_validation_equivalence','ddp_checkpoint_resume',
                'whole_pair_sample_order','rank_parameter_sync'}
    if not proof.get('passed') or proof.get('scope')!='stage2_ddp_migration' or not required.issubset(proof.get('verified_features',[])):
        raise ValueError('DDP numerical checks have not passed')
    if proof.get('origin_checkpoint_sha256')!=manifest['initialization']['checkpoint_sha256'] or proof.get('new_source_sha256')!=manifest['source_sha256']:
        raise ValueError('DDP migration report checked different weights or source')


def main():
    args = arguments()
    rank, size = int(os.environ.get('RANK','0')), int(os.environ.get('WORLD_SIZE','1'))
    local_rank = int(os.environ.get('LOCAL_RANK','0'))
    local_batch = local_batch_size(args.global_batch,size)
    if int(os.environ.get('LOCAL_WORLD_SIZE',str(size)))!=size:
        raise ValueError('This entrypoint supports one host with shared local storage')
    if not torch.cuda.is_available() or local_rank>=torch.cuda.device_count():
        raise ValueError('One available CUDA device is required per rank')
    torch.set_num_threads(4)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda',local_rank)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    started = time.monotonic()
    if size>1:
        dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    output = Path(args.output)
    prefetch = None
    try:
        output.mkdir(parents=True,exist_ok=True)
        lock = None
        if rank==0:
            lock = (output/'run.lock').open('a')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (output/'manifest.json').exists() and not args.resume:
                raise ValueError('Existing run: use --resume or a new output')
            if args.init_checkpoint and Path(args.init_checkpoint).resolve().parent==output.resolve():
                raise ValueError('Preserve the parent run; use a separate output')
        if size>1:
            dist.barrier()
        index, statistics = read_json(args.index), read_json(args.statistics)
        config, index_sha = configuration('full',args.seed), sha256(args.index)
        if index['development_only'] and not args.development:
            raise ValueError('Formal training requires the accepted index')
        if statistics.get('role')!='train' or statistics.get('index_sha256')!=index_sha or statistics.get('split_sha256')!=index['split_sha256']:
            raise ValueError('Normalization belongs to a different training split')
        train, val = Dataset(index,'train'), Dataset(index,'val')
        rows, panel = validation_panel(val.rows,args.validation_minimum,config['validation_panel_seed'])
        model, encoders, observation = build_models(args,statistics,device)
        if observation['manifest']['index_sha256']!=index_sha or observation['manifest']['protocol_sha256']!=config['protocol_sha256']:
            raise ValueError('Observation checkpoint split/protocol mismatch')
        if observation['manifest']['development_only'] and not args.development:
            raise ValueError('Development observation checkpoint')
        dino_sha = sha256(args.dino_weights)
        if observation['manifest']['dino_sha256']!=dino_sha:
            raise ValueError('Observation/DINO weight mismatch')
        manifest = dict(stage='dynamics_geometry',development_only=args.development,
            index_sha256=index_sha,split_sha256=index['split_sha256'],protocol_sha256=config['protocol_sha256'],
            config=config,steps=args.steps,dependency_sha256={'observation':sha256(args.observation_checkpoint)},
            observation_step=observation['step'],statistics_sha256=sha256(args.statistics),dino_sha256=dino_sha,
            vae_sha256={p.name:sha256(p) for p in sorted(Path(args.vae).glob('*')) if p.suffix in ('.json','.safetensors')},
            source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
            effective_batch_pairs=args.global_batch,precision='float32',train_windows=len(train),
            validation_windows=len(rows),validation_panel=panel,validation_every=args.validation_every,
            torch_version=torch.__version__,limitations=index.get('limitations',[]),
            execution=dict(world_size=size,backend='nccl' if size>1 else 'single',workers=args.workers,
                prefetch=args.prefetch,geometry_chunk=args.geometry_chunk,clip_microbatch=1,
                local_accumulation=local_batch,policy=args.execution_policy,
                voxel_chunk=65536 if args.execution_policy=='optimized_v1' else 8192,
                dynamics_checkpoint_above_tracks=model.dynamics.checkpoint_tracks_above,
                find_unused_parameters=True,gradient_as_bucket_view=True,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32))
        del observation
        origin = torch.load(args.init_checkpoint,map_location='cpu',weights_only=False) if args.init_checkpoint else None
        if origin is not None:
            check_parent(origin['manifest'],manifest,args.development)
            manifest['initialization'] = dict(checkpoint=str(Path(args.init_checkpoint).resolve()),
                checkpoint_sha256=sha256(args.init_checkpoint),step=origin['step'],
                prior_execution=origin['manifest']['execution'],prior_migration=origin['manifest'].get('migration'))
        check_migration_report(args,manifest)
        saved = torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else origin
        if args.resume and (saved['manifest']!=manifest or len(saved.get('rng_by_rank',[]))!=size):
            raise ValueError('Resume changed inputs/source/protocol/world size or lost rank RNG')
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=.01)
        step0, best_loss, best_step, prior_elapsed, prior_gpu = 0, None, None, 0., 0.
        consumed = copy.deepcopy(np.random.default_rng(args.seed).bit_generator.state)
        if saved is not None:
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            consumed = copy.deepcopy(saved['numpy_rng'])
            step0, best_loss, best_step = saved['step'],saved['best_validation_loss'],saved['best_step']
            prior_elapsed = saved['elapsed_training_seconds']
            prior_gpu = saved.get('elapsed_gpu_seconds',prior_elapsed)
        if step0>=args.steps:
            raise ValueError('Checkpoint already completed this stage')
        wrapped = ObjectiveModule(model,args.geometry_chunk)
        if size>1:
            wrapped = DistributedDataParallel(wrapped,device_ids=[local_rank],output_device=local_rank,
                find_unused_parameters=True,gradient_as_bucket_view=True,broadcast_buffers=False)
        # DDP initialization precedes RNG restoration. Rank zero inherits the old
        # stream; new ranks get documented distinct streams only on migration.
        if args.resume:
            restore_rng(saved['rng_by_rank'][rank],device)
        elif saved is not None and rank==0:
            torch.set_rng_state(saved['torch_rng'].cpu())
            if saved['cuda_rng'] is not None:
                torch.cuda.set_rng_state(saved['cuda_rng'][0].cpu(),device)
        else:
            torch.manual_seed(args.seed+1000003*rank)
            np.random.seed(args.seed+1000003*rank); random.seed(args.seed+1000003*rank)
        log = output/'training.jsonl'
        if rank==0:
            if origin is not None and not args.resume:
                inherit_records(args.init_checkpoint,output,manifest,origin)
            if log.exists():
                lines=[line for line in log.read_text().splitlines() if json.loads(line)['step']<=step0]
                log.write_text('\n'.join(lines)+'\n')
            frozen=output/'source/agcr_exp'
            if not frozen.exists():
                shutil.copytree(Path(__file__).parent,frozen,ignore=shutil.ignore_patterns('__pycache__'))
            for name, expected in manifest['source_sha256'].items():
                if sha256(frozen/name)!=expected:
                    raise ValueError('Frozen source mismatch: '+name)
            atomic_json(output/'manifest.json',manifest)
            atomic_json(output/'validation-panel.json',panel)
            atomic_json(output/'state-normalization.json',statistics)
        del saved, origin
        stop_requested = False
        def request_stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True
        for sig in (signal.SIGTERM,signal.SIGINT):
            signal.signal(sig,request_stop)
        def status(step,state,record=None,progress=None):
            if rank!=0:
                return
            elapsed=time.monotonic()-started
            atomic_json(output/'status.json',dict(status=state,pid=os.getpid(),complete=step==args.steps,
                completed_steps=step,target_steps=args.steps,best_step=best_step,best_validation_loss=best_loss,
                elapsed_seconds=prior_elapsed+elapsed,session_elapsed_seconds=elapsed,
                elapsed_gpu_seconds=prior_gpu+size*elapsed,session_gpu_seconds=size*elapsed,world_size=size,
                updated_unix=time.time(),development_only=args.development,last=record,validation_progress=progress,
                remaining_seconds=(args.steps-step)*elapsed/(step-step0) if step>step0 else None,
                cuda_peak_gib=torch.cuda.max_memory_allocated(device)/2**30))
        def save(step,name='last.pt'):
            states=gather_objects(dict(rng=capture_rng(device),sampler=consumed,digest=state_digest(model)))
            if any(s['sampler']!=consumed or s['digest']!=states[0]['digest'] for s in states):
                raise RuntimeError('Ranks diverged in parameters or consumed sample stream')
            if rank==0:
                elapsed=time.monotonic()-started
                temporary=output/(name+'.tmp')
                torch.save(dict(step=step,manifest=manifest,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    numpy_rng=consumed,torch_rng=states[0]['rng']['torch'],cuda_rng=[s['rng']['cuda'] for s in states],
                    rng_by_rank=[s['rng'] for s in states],rank_model_sha256=[s['digest'] for s in states],
                    best_validation_loss=best_loss,best_step=best_step,
                    elapsed_training_seconds=prior_elapsed+elapsed,elapsed_gpu_seconds=prior_gpu+size*elapsed),temporary)
                temporary.replace(output/name)
            if size>1:
                dist.barrier()
        model.train()
        save(step0)
        status(step0,'running')
        prefetch=DistributedPrefetch(train,consumed,rank,size,args.workers,args.prefetch)
        record, step, last_saved = None, step0, step0
        for step in range(step0+1,args.steps+1):
            tick, wait=time.monotonic(),0.
            optimizer.zero_grad(set_to_none=True)
            factor=min(step/1000,1.)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-1-1000)/max(1,args.steps-1000))))
            for group in optimizer.param_groups:
                group['lr']=1e-4*factor
            samples=[]
            for pair in range(local_batch):
                loading=time.monotonic()
                sample,next_state,index_number,auxiliary_offset=prefetch.next()
                wait+=time.monotonic()-loading
                batch=tensors(sample,device)
                encoded=encoders.history(batch['inputs']['rgb'])
                auxiliary=encoders.features(batch['auxiliary']['rgb'])
                with accumulation_context(wrapped,pair,local_batch):
                    loss,parts,metrics=wrapped(batch,encoded,auxiliary)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss: '+sample['uid'])
                    (loss/local_batch).backward()
                samples.append(dict(position=pair*size+rank,selection=[index_number,auxiliary_offset],
                    loss=float(loss.detach()),parts={k:float(v) for k,v in parts.items()},diagnostics=metrics))
                status(step-1,'running',dict(updating_step=step,local_accumulated_pairs=pair+1,
                                            local_target_pairs=local_batch,world_size=size))
                del batch,encoded,auxiliary,loss,parts,sample
            gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            consumed=next_state
            collected=gather_objects(dict(samples=samples,data_wait_seconds=wait,peak_gib=torch.cuda.max_memory_allocated(device)/2**30))
            torch.cuda.synchronize(device)
            record=dict(step=step,**merge_training_records(collected),gradient_norm=float(gradient),
                lr=optimizer.param_groups[0]['lr'],update_seconds=time.monotonic()-tick,
                elapsed_seconds=time.monotonic()-started,elapsed_training_seconds=prior_elapsed+time.monotonic()-started,
                elapsed_gpu_seconds=prior_gpu+size*(time.monotonic()-started),world_size=size,
                global_batch_pairs=args.global_batch,local_batch_pairs=local_batch)
            if step%args.validation_every==0 or step==args.steps:
                status(step,'validating',record)
                result=distributed_validation(model,encoders,val,rows,device,args.geometry_chunk,rank,size,
                    lambda done,count:status(step,'validating',record,dict(rank0_done=done,rank0_total=count,total=len(rows))))
                record.update(validation_loss=result['validation_loss'],validation_by_layout=result['validation_by_layout'],
                    validation_windows=len(rows),validation_panel_sha256=panel['panel_sha256'])
                if rank==0:
                    atomic_json(output/('validation-%06d.json'%step),result)
                if best_loss is None or result['validation_loss']<best_loss:
                    best_loss,best_step=result['validation_loss'],step
                    save(step,'best.pt')
            limited=args.max_updates is not None and step-step0>=args.max_updates
            timed_out=args.max_gpu_seconds is not None and size*(time.monotonic()-started)>=args.max_gpu_seconds
            user_paused=not args.development and Path(args.pause_file).exists()
            stopped=stop_on_any_rank(stop_requested or limited or timed_out or user_paused,device)
            if rank==0:
                with log.open('a') as stream:
                    stream.write(json.dumps(record,allow_nan=False)+'\n')
                print(json.dumps(record,allow_nan=False),flush=True)
            if step%10==0 or stopped or step==args.steps:
                save(step); last_saved=step
            status(step,'running',record)
            if stopped:
                break
        if last_saved!=step:
            save(step)
        status(step,'complete' if step==args.steps else 'checkpointed',record)
    except BaseException as exc:
        atomic_json(output/('error-rank-%d.json'%rank),dict(rank=rank,error=repr(exc),updated_unix=time.time()))
        if rank==0:
            current=read_json(output/'status.json') if (output/'status.json').exists() else {}
            if not current or current.get('pid')==os.getpid():
                atomic_json(output/'status.json',dict(current,status='failed',complete=False,error=repr(exc),updated_unix=time.time()))
        raise
    finally:
        if prefetch is not None:
            prefetch.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__=='__main__':
    main()
