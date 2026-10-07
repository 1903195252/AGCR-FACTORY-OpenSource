"""Independent Base48 SynCamMaster adaptation, native timing and RGB exports.

Uses the real upstream implementation through external_baselines.models.create.
An elapsed-time benchmark is never an adaptation or an effectiveness result.
"""
import argparse
from contextlib import nullcontext, contextmanager
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .train import BudgetLedger, SAMPLE_SEEDS, atomic_checkpoint, draw_plan
from experiments.paired_interventions.data import (
    FamilyDataset, OrdinaryDataset, FamilySampler, to_device, file_sha256,
)
from experiments.paired_interventions.train import atomic_json, trainable_digest
from experiments.paired_interventions.validation import panel_indices
from experiments.external_baselines.models import create
from experiments.external_baselines.official import LOCK
from experiments.external_baselines.run import Objective, learned, load_learned

METHOD = 'syncammaster'
SCHEMA = 'agcr-base48-external-training-1'


def settings(protocol, benchmark_only=False):
    if protocol.get('schema') != 'agcr-base48-protocol-1':
        raise ValueError('An independent Base48 protocol is required')
    cfg = dict(protocol.get('external', {}).get(METHOD, {}))
    for key in ('steps', 'batch', 'validate_every', 'panel'):
        if benchmark_only and cfg.get(key) is None:
            cfg[key] = None
        elif type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError('Unregistered positive external setting: ' + key)
    if not math.isfinite(float(cfg.get('lr', 0.))) or cfg['lr'] <= 0:
        raise ValueError('Register the external learning rate')
    cfg.setdefault('modality', 'rgbd')
    if cfg['modality'] not in ('rgb', 'rgbd'):
        raise ValueError('Conditioning modality must be rgb or rgbd')
    cfg.setdefault('weight_decay', .01)
    cfg.setdefault('clip_norm', 1.)
    cfg.setdefault('sampling_steps', 50)
    if cfg['sampling_steps'] != 50:
        raise ValueError('The shared native sampling protocol uses 50 steps')
    if protocol.get('paired_sample_seeds', SAMPLE_SEEDS) != SAMPLE_SEEDS:
        raise ValueError('The shared formal sample protocol requires all eight seeds')
    return cfg


def check_destination(output, ledger, source, backbone):
    for value in (Path(output).resolve(), Path(ledger).resolve()):
        if 'agcr-intervention-runs' in value.parts:
            raise ValueError('The preserved Pro run/budget cannot be modified')
        for parent in (Path(source).resolve(), Path(backbone).resolve()):
            if value == parent or parent in value.parents:
                raise ValueError('Do not write into official source or model assets')


def implementation_hashes(source, root=None):
    """Only this entry, its imported runtime and the native SynCamMaster files."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    source = Path(source)
    native = ('diffsynth/models/wan_video_dit.py', 'diffsynth/models/utils.py',
              'diffsynth/schedulers/flow_match.py')
    required = ('experiments/base48/__init__.py', 'experiments/base48/external.py',
        'experiments/base48/train.py', 'experiments/external_baselines/__init__.py',
        'experiments/external_baselines/models.py', 'experiments/external_baselines/conditions.py',
        'experiments/external_baselines/official.py', 'experiments/external_baselines/official_sources.json',
        'experiments/external_baselines/run.py', 'experiments/external_baselines/provenance.py',
        'experiments/performance/execution_runtime.py', 'experiments/performance/prefetched_replay.py')
    hashes = {name:file_sha256(root/name) for name in required}
    for directory in ('paired_interventions', 'paper_experiments/agcr_exp'):
        for path in sorted((root/'experiments'/directory).glob('*.py')):
            if not path.name.startswith(('test_', 'check_', 'benchmark')):
                hashes[str(path.relative_to(root))] = file_sha256(path)
    return dict(native_source_sha256={name:file_sha256(source/name) for name in native},
                source_hashes=hashes, source_hash_scope='base48_actual_runtime_and_native_files_v1')


def verify_sampling_implementation(identity, source, text, modality):
    """Base adapter identity; frozen Pro's broader hash contract stays unchanged."""
    if (identity.get('schema') != SCHEMA or identity.get('method') != METHOD
            or identity.get('phase') != 'video' or identity.get('modality') != modality):
        raise ValueError('Sampling requires the matching Base48 native video checkpoint')
    expected = implementation_hashes(source)
    expected['text_sha256'] = file_sha256(text)
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError('Native Base48 sampling implementation changed: ' + key)


def identity_for(a, cfg, panel, world):
    value = dict(schema=SCHEMA, method=METHOD, phase='video', modality=cfg['modality'],
        training_condition_source='native_rgb_supervision', source=LOCK[METHOD],
        seed=a.seed, batch=cfg['batch'], steps=cfg['steps'], learning_rate=cfg['lr'],
        configuration=cfg, world_size=world, development_only=False,
        families_sha256=file_sha256(a.families), ordinary_sha256=file_sha256(a.ordinary),
        text_sha256=file_sha256(a.text), forecaster_checkpoint_sha256=None,
        backbone=str(Path(a.backbone).resolve()),
        backbone_provenance_sha256=file_sha256(Path(a.backbone)/'source_manifest.json')
          if (Path(a.backbone)/'source_manifest.json').exists() else None,
        protocol_sha256=file_sha256(a.protocol), base_entry_sha256=file_sha256(__file__),
        training_runtime_sha256=file_sha256(Path(__file__).with_name('train.py')),
        validation_panel=panel, validation_scope='native_noise_objective_for_model_selection',
        validation_objective_draw_seeds=SAMPLE_SEEDS, output_modality='rgb',
        formal_experiment_results=False, input_frames=6, future_frames=16,
        resolution_wh=[720, 480], paired_views=['air', 'ground'])
    value.update(implementation_hashes(a.source))
    return value


@torch.no_grad()
def validation(model, dataset, panel, device, step, rank=0, world=1):
    """Fixed native loss/noise panel; no claim that videos were generated."""
    modes = {module: module.training for module in model.modules()}
    model.eval(); rows = []
    try:
        for pos in range(rank, len(panel), world):
            data, meta = dataset.branch(panel[pos], (pos % 2, (pos // 2) % 2))
            data = to_device(data, device)
            base = int(hashlib.sha256(meta['family_id'].encode()).hexdigest()[:8], 16)
            values = []
            for seed in SAMPLE_SEEDS:
                loss = model.loss(data, torch.Generator(device=device).manual_seed(base + seed))
                value = float(loss)
                if not math.isfinite(value): raise FloatingPointError('Invalid native validation objective')
                values.append(value)
            rows.append(dict(family=meta['family_id'], value=sum(values)/len(values), objective_draws=values))
    finally:
        for module, state in modes.items(): module.training = state
    gathered = [None]*world
    if world > 1: torch.distributed.all_gather_object(gathered, rows)
    else: gathered[0] = rows
    rows = sorted((r for group in gathered for r in group), key=lambda r:r['family'])
    if not rows: raise ValueError('Empty external validation panel')
    return dict(step=step, mean=sum(r['value'] for r in rows)/len(rows), rows=rows,
        scope='native_noise_objective_for_model_selection', objective_draw_seeds=SAMPLE_SEEDS,
        generated_video_sample_count=0, formal_experiment_results=False,
        independent_benchmark_metrics=False)


def rng_state(generator, device):
    return dict(generator=generator.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device), numpy=np.random.get_state(), python=random.getstate())


def restore_rng(rng, generator, device):
    generator.set_state(rng['generator']); torch.set_rng_state(rng['cpu'])
    torch.cuda.set_rng_state(rng['cuda'], device)
    np.random.set_state(rng['numpy']); random.setstate(rng['python'])


def save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream: np.savez_compressed(stream, **arrays)
    os.replace(temp, path)


class SamplingBoundaryStop(RuntimeError):
    pass


@contextmanager
def bounded_sampling(model, ledger, stop, stop_file=None):
    # Native denoising calls the DiT once per step; stop between calls without
    # exporting an incomplete video or modifying upstream sampler code.
    def boundary(_module, _args):
        if stop[0] or ledger.exhausted() or (stop_file and Path(stop_file).exists()):
            raise SamplingBoundaryStop('sampling_step_boundary')
    handle = model.dit.register_forward_pre_hook(boundary)
    try:
        yield
    finally:
        handle.remove()


def sample_corpus(a, cfg, model, dataset, checkpoint, root, device, ledger, stop):
    """Original formal-evaluation manifest/schema, without fictitious depth."""
    identity = checkpoint['identity']
    if not checkpoint.get('completed') or not checkpoint.get('validation_best'):
        raise ValueError('Formal exports require completed adaptation and held-out checkpoint selection')
    verify_sampling_implementation(identity, a.source, a.text, cfg['modality'])
    if identity['protocol_sha256'] != file_sha256(a.protocol) or identity['families_sha256'] != file_sha256(a.families):
        raise ValueError('Sampling data/protocol differs from adaptation')
    manifest = dict(schema='agcr-intervention-predictions-1', method=METHOD,
        sample_seeds=SAMPLE_SEEDS, training_seed=identity['seed'],
        checkpoint_sha256=file_sha256(a.checkpoint), index_sha256=file_sha256(a.families),
        split=a.split, development_only=False, conditions='history_only',
        sampling_steps=50, conditioning_modality=cfg['modality'], output_modality='rgb',
        native_upstream=LOCK[METHOD], adaptation_identity=identity, perturbation={'kind':'clean'})
    contract = root/'manifest.json'
    if contract.exists() and json.loads(contract.read_text()) != manifest:
        raise ValueError('Cannot mix native export provenance')
    atomic_json(contract, manifest)
    model.eval(); emitted = 0
    for index, row in enumerate(dataset.families):
        folder = root/row['family_id']; folder.mkdir(exist_ok=True)
        for draw, seed in enumerate(SAMPLE_SEEDS):
            marker = folder/('draw_%02d.json'%draw)
            if marker.exists():
                prior = json.loads(marker.read_text())
                if len(prior.get('files', {})) != 4 or any(file_sha256(folder/n)!=h for n,h in prior['files'].items()):
                    raise ValueError('Changed/incomplete native draw')
                emitted += 1; continue
            files = {}
            for w in range(2):
                for c in range(2):
                    if stop[0] or ledger.exhausted() or (a.stop_file and Path(a.stop_file).exists()):
                        atomic_json(root/'completion.json', dict(completed=False, completed_draws=emitted, reason='bounded_or_requested_stop'))
                        return
                    branch, _ = dataset.branch(index, (w,c))
                    inputs = to_device(branch['inputs'], device)
                    try:
                        with bounded_sampling(model, ledger, stop, a.stop_file), torch.no_grad():
                            rgb = model.sample(inputs, seed, 50)
                    except SamplingBoundaryStop:
                        atomic_json(root/'completion.json', dict(completed=False, completed_draws=emitted, reason='bounded_or_requested_stop'))
                        return
                    if tuple(rgb.shape) != (16,2,480,720,3): raise ValueError('Wrong native paired RGB shape')
                    name = 'world%d_camera%d_sample%02d.npz'%(w,c,draw)
                    save_npz(folder/name, rgb=rgb.cpu().numpy(), K=inputs['K'].cpu().numpy(),
                             camera_to_world=inputs['planned_camera_to_world'].cpu().numpy())
                    files[name] = file_sha256(folder/name)
                    ledger.commit('sampling', family=row['family_id'], draw=draw, world=w, camera=c)
            atomic_json(marker, dict(seed=seed, files=files)); emitted += 1
    completion = dict(completed=True, completed_draws=emitted, benchmark_results=False)
    atomic_json(root/'completion.json', completion)
    atomic_json(root/'completed.json', completion)


def benchmark(a, cfg, model, ordinary, device, root, ledger, stop):
    """One actual optimizer update and one 50-step sample; never save benchmark weights."""
    model.train(); model.vae.eval()
    branch = to_device(ordinary[0], device)
    generator = torch.Generator(device=device).manual_seed(a.seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    start = time.monotonic(); optimizer.zero_grad(set_to_none=True)
    loss = model.loss(branch, generator)
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite native benchmark loss')
    loss.backward(); nn.utils.clip_grad_norm_(parameters, cfg['clip_norm'], error_if_nonfinite=True)
    optimizer.step(); torch.cuda.synchronize(device)
    training_seconds = time.monotonic()-start
    report = dict(schema='agcr-base48-native-benchmark-1', method=METHOD, native_model=True,
        formal_experiment_results=False, adaptation_completed=False, ordinary_index=0,
        gradient_microbatches=1, train_microbatch_seconds=training_seconds,
        loss=float(loss.detach()), checkpoint_modified=False,
        training_peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30,
        generation_performed=False, sampling_steps=50)
    atomic_json(root/'benchmark.json', report)
    if stop[0] or ledger.exhausted() or (a.stop_file and Path(a.stop_file).exists()): return
    # The transient update above is only a throughput probe. The output below
    # is a timing artifact, not an exported adaptation or a quality sample.
    optimizer.zero_grad(set_to_none=True)
    del optimizer, loss
    model.eval(); torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    start = time.monotonic()
    try:
        with bounded_sampling(model, ledger, stop, a.stop_file), torch.no_grad():
            rgb = model.sample(branch['inputs'], SAMPLE_SEEDS[0], 50)
    except SamplingBoundaryStop:
        report.update(stopped_at_sampling_boundary=True)
        atomic_json(root/'benchmark.json', report)
        return
    torch.cuda.synchronize(device)
    if tuple(rgb.shape) != (16,2,480,720,3): raise ValueError('Wrong benchmark paired output dimensions')
    save_npz(root/'benchmark_pair_timing_only.npz', rgb=rgb.cpu().numpy())
    report.update(generation_performed=True, paired_generation_seconds=time.monotonic()-start,
        generation_peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30,
        output_shape=list(rgb.shape), output_modality='rgb')
    atomic_json(root/'benchmark.json', report)


def train(a, cfg, model, families, ordinary, heldout, panel, identity, previous,
          root, device, ledger, stop, world, rank, local):
    generator = torch.Generator(device=device).manual_seed(a.seed+100003*rank)
    fs, ds = FamilySampler(len(families),a.seed+107), FamilySampler(len(ordinary),a.seed+311)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    step = 0; prior_hours = 0.; best = {}; start = time.monotonic()
    if previous:
        if previous['identity'] != identity: raise ValueError('External exact resume identity changed')
        if previous.get('budget_ledger') != str(Path(a.budget_ledger).resolve()):
            raise ValueError('Resume must use its existing independent budget ledger')
        optimizer.load_state_dict(previous['optimizer']); step=previous['step']
        prior_hours=previous['cumulative_gpu_hours']; best=previous.get('validation_best',{})
        fs=FamilySampler(**previous['family_sampler']); ds=FamilySampler(**previous['ordinary_sampler'])
        restore_rng(previous['rng'][rank],generator,device)
    wrapped = Objective(model)
    if world>1: wrapped=DDP(wrapped,device_ids=[local],find_unused_parameters=False,broadcast_buffers=False)
    model.train(); model.vae.eval()
    def save(name='last.pt', complete=False):
        states=[None]*world
        if world>1: torch.distributed.all_gather_object(states,rng_state(generator,device))
        else: states[0]=rng_state(generator,device)
        if rank==0:
            atomic_checkpoint(root/name,dict(identity=identity,model=learned(model),optimizer=optimizer.state_dict(),
                step=step,rng=states,family_sampler=fs.state_dict(),ordinary_sampler=ds.state_dict(),
                validation_best=best,completed=complete,cumulative_gpu_hours=prior_hours+world*(time.monotonic()-start)/3600,
                budget_ledger=str(Path(a.budget_ledger).resolve())))
            ledger.commit('external_checkpoint',step=step,run=str(root))
        if world>1: torch.distributed.barrier()
    try:
        while step<cfg['steps']:
            requested=stop[0] or (rank==0 and ledger.exhausted()) or (a.stop_file and Path(a.stop_file).exists())
            flag=torch.tensor(int(bool(requested)),device=device)
            if world>1: torch.distributed.all_reduce(flag,op=torch.distributed.ReduceOp.MAX)
            if int(flag): break
            plan,nf,nd=draw_plan(step,cfg['batch'],True,fs,ds,world)
            optimizer.zero_grad(set_to_none=True); total=0.; begin=time.monotonic()
            for micro,item in enumerate(plan[rank::world]):
                kind,index,cell=item
                branch=families.branch(index,cell)[0] if kind=='family' else ordinary[index]
                branch=to_device(branch,device)
                sync=nullcontext() if world==1 or micro==cfg['batch']//world-1 else wrapped.no_sync()
                with sync:
                    loss=wrapped(branch,generator)/(cfg['batch']//world)
                    finite=torch.isfinite(loss).int()
                    if world>1: torch.distributed.all_reduce(finite,op=torch.distributed.ReduceOp.MIN)
                    if not int(finite): raise FloatingPointError('Nonfinite native objective before optimizer commit')
                    loss.backward()
                total+=float(loss.detach()); del loss,branch
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['clip_norm'],error_if_nonfinite=True)
            optimizer.step();step+=1;fs.advance(nf);ds.advance(nd)
            value=torch.tensor(total,device=device)
            if world>1: torch.distributed.all_reduce(value);value/=world
            if rank==0:
                status=dict(state='running',method=METHOD,step=step,target_steps=cfg['steps'],
                    update_seconds=time.monotonic()-begin,loss=float(value),
                    cumulative_gpu_hours=prior_hours+world*(time.monotonic()-start)/3600,
                    world_size=world,peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30,
                    formal_experiment_results=False,updated_unix=time.time())
                atomic_json(root/'status.json',status)
                with (root/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(status)+'\n')
                ledger.commit('external_training',step=step,run=str(root));print(json.dumps(status),flush=True)
            if step%a.save_every==0: save()
            if step==cfg['steps']:
                digests=[None]*world
                if world>1: torch.distributed.all_gather_object(digests,trainable_digest(model))
                else: digests[0]=trainable_digest(model)
                if len(set(digests))!=1: raise AssertionError('Native DDP parameter divergence before completion')
            if step%cfg['validate_every']==0 or step==cfg['steps']:
                report=validation(model,heldout,panel,device,step,rank,world)
                if rank==0:
                    atomic_json(root/('validation_%06d.json'%step),report)
                    atomic_json(root/'validation_latest.json',report)
                if report['mean']<best.get('value',float('inf')):
                    best=dict(value=report['mean'],step=step,scope=report['scope'])
                    save('best.pt',step==cfg['steps'])
                save(complete=step==cfg['steps'])
        completed=step==cfg['steps'];save(complete=completed)
        if rank==0:
            if completed and (root/'best.pt').exists():
                selected=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
                selected.update(completed=True,training_completed_step=step)
                atomic_checkpoint(root/'best.pt',selected)
            result=dict(completed=completed,step=step,target_steps=cfg['steps'],
                        reason='complete' if completed else 'bounded_or_requested_stop',
                        formal_experiment_results=False,validation_best=best)
            atomic_json(root/'completion.json',result)
            atomic_json(root/'status.json',dict(result,state='completed' if completed else 'paused',method=METHOD))
    except BaseException as error:
        if rank==0: atomic_json(root/'failure.json',dict(type=type(error).__name__,message=str(error),last_committed_step=step,completed=False))
        raise


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','benchmark','sample'])
    for key in ('protocol','source','backbone','text','families','ordinary','output','budget-ledger'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--gpu-hours',required=True,type=float)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--checkpoint');p.add_argument('--resume')
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--stop-file');p.add_argument('--threads',type=int,default=8)
    return p.parse_args(argv)


def main(argv=None):
    a=arguments(argv);cfg=settings(json.loads(Path(a.protocol).read_text()), benchmark_only=a.action=='benchmark')
    if a.gpu_hours<=0 or a.save_every<1: raise ValueError('Positive budget/save interval required')
    if a.resume and a.checkpoint: raise ValueError('Use resume or checkpoint, not both')
    if a.resume and (a.action!='train' or not Path(a.budget_ledger).exists()):
        raise ValueError('Resume requires its persisted training budget ledger')
    if a.action=='train' and a.checkpoint:
        raise ValueError('External adaptation starts from official weights or an exact Base48 resume')
    if a.action=='sample' and not a.checkpoint: raise ValueError('Formal sampling needs a completed checkpoint')
    check_destination(a.output,a.budget_ledger,a.source,a.backbone)
    world,rank,local=(int(os.environ.get(k,d)) for k,d in [('WORLD_SIZE',1),('RANK',0),('LOCAL_RANK',0)])
    if a.action=='train' and (cfg['batch']%world or cfg['batch']<world): raise ValueError('Global batch must divide by world size')
    if a.action!='train' and world!=1: raise ValueError('Native benchmark/sample runs use one explicit GPU')
    root=Path(a.output).resolve()
    if a.action in ('train','benchmark') and root.exists() and any(root.iterdir()) and not a.resume:
        raise FileExistsError('Fresh train/benchmark needs an empty independent output directory')
    root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    split=a.split if a.action=='sample' else 'train'
    families=FamilyDataset(a.families,split,verify_files=rank==0)
    ordinary=OrdinaryDataset(a.ordinary)
    if families.index.get('development_only') or ordinary.index.get('development_only'):
        raise ValueError('Native Base48 comparison requires audited formal data')
    heldout=FamilyDataset(a.families,'val',verify_files=rank==0) if a.action=='train' else None
    panel=panel_indices(heldout,cfg['panel']) if heldout else []
    if heldout and len(panel)!=cfg['panel']: raise ValueError('Insufficient registered validation panel')
    identity=identity_for(a,cfg,[heldout.families[i]['family_id'] for i in panel],world)
    previous=torch.load(a.resume or a.checkpoint,map_location='cpu',weights_only=False) if a.resume or a.checkpoint else None
    if a.resume and (previous['identity']!=identity or previous.get('budget_ledger')!=str(Path(a.budget_ledger).resolve())):
        raise ValueError('External exact resume source/schedule/budget identity changed')
    if a.action=='sample' and (not previous.get('completed') or not previous.get('validation_best')):
        raise ValueError('Sampling requires completed native adaptation and validated checkpoint selection')
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    ledger=None
    try:
        torch.cuda.set_device(local);device=torch.device('cuda',local)
        if world>1: torch.distributed.init_process_group('nccl',timeout=timedelta(minutes=60))
        if rank==0:
            ledger=BudgetLedger(a.budget_ledger,a.gpu_hours,world)
            if a.action=='train': atomic_json(root/'manifest.json',identity)
        model=create(METHOD,source=a.source,backbone=a.backbone,text_path=a.text,device=device,modality=cfg['modality'])
        if previous: load_learned(model,previous['model'])
        if a.action=='benchmark': benchmark(a,cfg,model,ordinary,device,root,ledger,stop)
        elif a.action=='sample': sample_corpus(a,cfg,model,families,previous,root,device,ledger,stop)
        else: train(a,cfg,model,families,ordinary,heldout,panel,identity,previous,root,device,ledger,stop,world,rank,local)
    finally:
        if ledger: ledger.close('stopped',action=a.action,output=str(root))
        if world>1 and torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__=='__main__': main()
