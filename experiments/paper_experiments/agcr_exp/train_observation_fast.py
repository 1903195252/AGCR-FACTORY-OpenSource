"""Validated batched continuation of observation/association pretraining."""
import argparse
import copy
import fcntl
import json
import math
import os
import random
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch

from .data import Dataset, atomic_json, read_json, read_observation_pair
from .observation import ObservationHead, observation_loss
from .observation_diagnostics import ObservationDiagnostics
from .observation_runtime import OrderedPrefetch, batched_forward, collate_pairs, pair_outputs, prepare_pair
from .protocol import configuration
from .selection import validation_mean, validation_panel
from .train_observation import frame_tensors, sha256


@torch.no_grad()
def validate(model, dino, dataset, rows, device, microbatch, rgb_only=False, reference=False):
    model.eval()
    values, diagnostics = [], ObservationDiagnostics()
    if reference:
        for row in rows:
            rgb, inputs, target = frame_tensors(read_observation_pair(dataset.root, row, 5), 0, device)
            out = model(dino.forward_features(rgb)['x_norm_patchtokens'], **inputs, rgb_only=rgb_only)
            values.append(float(observation_loss(model, out, target)[0]))
    else:
        for start in range(0, len(rows), microbatch):
            pairs = [prepare_pair((dataset.root, row, 5)) for row in rows[start:start+microbatch]]
            rgb, inputs, targets = collate_pairs(pairs, device)
            output = model(dino.forward_features(rgb)['x_norm_patchtokens'], **inputs, rgb_only=rgb_only)
            for i, target in enumerate(targets):
                out = pair_outputs(output, i)
                values.append(float(observation_loss(model, out, target)[0]))
                diagnostics.add(model, out, target)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError('Nonfinite validation loss')
    mean, by_layout = validation_mean(rows, values)
    model.train()
    return dict(validation_loss=mean, validation_by_layout=by_layout, window_losses=values,
                diagnostics=None if reference else diagnostics.result())


def main():
    parser = argparse.ArgumentParser()
    for key in ('index', 'output', 'dino-code', 'dino-weights'):
        parser.add_argument('--'+key, required=True)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument('--initialize-from')
    origin.add_argument('--resume')
    parser.add_argument('--migration-report')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--steps', type=int, default=40000)
    parser.add_argument('--variant', choices=['full'], default='full')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--microbatch', type=int, choices=[1,2,4,8,16], default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--max-updates', type=int)
    a = parser.parse_args()
    if a.max_updates is not None and (not a.development or a.max_updates < 1):
        raise ValueError('--max-updates requires --development and a positive limit')
    if a.steps != 40000 and not a.development:
        raise ValueError('Changed training budget requires --development')
    index, config = read_json(a.index), configuration(a.variant, a.seed)
    if index['development_only'] and not a.development:
        raise ValueError('Partial dataset cannot initialize a formal run')
    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.set_num_threads(4)
    dataset = Dataset(index, role='train', primary=True, load_surfaces=True)
    val = Dataset(index, role='val', primary=True, load_surfaces=True)
    rows, panel = validation_panel(val.rows, config['validation_panel_minimum'], config['validation_panel_seed'])
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    saved_path = Path(a.resume or a.initialize_from)
    saved = torch.load(saved_path, map_location=device, weights_only=False)
    source = {path.name: sha256(path) for path in Path(__file__).parent.glob('*.py')}
    execution = dict(microbatch_pairs=a.microbatch, workers=a.workers, prefetch=a.prefetch)
    manifest = dict(stage='observation_head_pretraining', development_only=a.development,
                    index_sha256=sha256(a.index), split_sha256=index['split_sha256'],
                    protocol_sha256=config['protocol_sha256'], dino_sha256=sha256(a.dino_weights),
                    steps=a.steps, effective_batch_pairs=16, variant=a.variant, source_sha256=source,
                    limitation='Observation-head/paired-association pretraining; temporal tracker not integrated.',
                    validation_panel=panel, config=config, precision='float32', execution=execution,
                    train_windows=len(dataset), validation_windows=len(rows),
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else str(device))
    parent = saved['manifest']
    if a.initialize_from:
        if (output/'manifest.json').exists():
            raise ValueError('Existing run: use --resume or a new output directory')
        if not a.migration_report:
            raise ValueError('A successful parity/throughput report is required')
        report = read_json(a.migration_report)
        if not report.get('accepted') or report['checkpoint_sha256'] != sha256(saved_path):
            raise ValueError('Benchmark did not accept this checkpoint')
        for key in ('index_sha256','dino_sha256'):
            if report[key] != manifest[key]:
                raise ValueError('Benchmark changed '+key)
        for key, value in execution.items():
            if report['selected'][key] != value:
                raise ValueError('Execution differs from benchmark selection: '+key)
        critical = ('train_observation.py','observation.py','modules.py','geometry.py','data.py','selection.py','protocol.py')
        for name in critical:
            if source[name] != parent['source_sha256'][name] or source[name] != report['source_sha256'][name]:
                raise ValueError('Changed training objective or inputs: '+name)
        if source['observation_runtime.py'] != report['source_sha256']['observation_runtime.py']:
            raise ValueError('Runtime changed after benchmarking')
        for key in ('index_sha256','split_sha256','dino_sha256','protocol_sha256','config','steps',
                    'effective_batch_pairs','variant','precision','validation_panel','train_windows','validation_windows'):
            if parent[key] != manifest[key]:
                raise ValueError('Migration changed data, weights or protocol: '+key)
        if parent['development_only'] and not a.development:
            raise ValueError('Development weights cannot initialize a formal continuation')
        manifest['lineage'] = dict(parent_checkpoint=str(saved_path), parent_checkpoint_sha256=sha256(saved_path),
                                   parent_step=saved['step'], benchmark_sha256=sha256(a.migration_report),
                                   benchmark_speedup=report['speedup'])
        shutil.copy2(a.migration_report, output/'throughput-benchmark.json')
        for name in ('training.jsonl','initial-validation.json','best.pt'):
            if (saved_path.parent/name).exists():
                shutil.copy2(saved_path.parent/name, output/name)
    else:
        manifest['lineage'] = parent['lineage']
        if manifest != parent:
            raise ValueError('Resume changed data, source, execution or protocol')
    step0 = saved['step']
    if step0 >= a.steps:
        raise ValueError('Checkpoint has already reached target updates')
    log = output/'training.jsonl'
    if log.exists():
        retained = [line for line in log.read_text().splitlines() if json.loads(line)['step'] <= step0]
        temp = log.with_suffix('.jsonl.tmp')
        temp.write_text('\n'.join(retained)+'\n')
        temp.replace(log)
    model = ObservationHead().to(device)
    model.load_state_dict(saved['model'])
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9,.999), weight_decay=.01)
    optim.load_state_dict(saved['optimizer'])
    dino = torch.hub.load(a.dino_code, 'dinov2_vitb14', source='local', pretrained=False).to(device)
    dino.load_state_dict(torch.load(a.dino_weights, map_location='cpu', weights_only=True))
    dino.eval().requires_grad_(False)
    torch.set_rng_state(saved['torch_rng'].cpu())
    if saved['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    consumed_state = copy.deepcopy(saved['numpy_rng'])
    best_validation, best_step = saved.get('best_validation_loss'), saved.get('best_step')
    prior_elapsed = saved.get('elapsed_training_seconds', 0.)
    del saved
    started = time.monotonic()
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    atomic_json(output/'manifest.json', manifest)
    atomic_json(output/'validation-panel.json', panel)
    def status(step, state, record=None):
        elapsed, count = time.monotonic()-started, step-step0
        atomic_json(output/'status.json', dict(status=state, pid=os.getpid(), completed_steps=step,
                    target_steps=a.steps, complete=step==a.steps, elapsed_seconds=prior_elapsed+elapsed,
                    session_elapsed_seconds=elapsed, remaining_seconds=(a.steps-step)*elapsed/count if count else None,
                    updated_unix=time.time(), best_validation_loss=best_validation, best_step=best_step,
                    development_only=a.development, last=record,
                    cuda_peak_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0.))
    def checkpoint(step, name='last.pt'):
        temp = output/(name+'.tmp')
        torch.save(dict(step=step, manifest=manifest, model=model.state_dict(), optimizer=optim.state_dict(),
                        numpy_rng=consumed_state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                        best_validation_loss=best_validation, best_step=best_step,
                        elapsed_training_seconds=prior_elapsed+time.monotonic()-started), temp)
        temp.replace(output/name)
    if a.initialize_from:
        status(step0, 'validating')
        reference = validate(model,dino,val,rows,device,a.microbatch,reference=True)
        migrated = validate(model,dino,val,rows,device,a.microbatch)
        old, new = np.array(reference['window_losses']), np.array(migrated['window_losses'])
        difference = np.abs(new-old)
        equivalent = bool(np.all(difference <= 2e-4*np.maximum(1,np.abs(old))))
        atomic_json(output/'migration-validation.json', dict(step=step0, reference=reference, batched=migrated,
                    max_window_loss_difference=float(difference.max()), equivalent=equivalent,
                    validation_panel_sha256=panel['panel_sha256']))
        if not equivalent:
            raise RuntimeError('Migration failed the fixed validation-panel equivalence check')
        checkpoint(step0)
        print(json.dumps(dict(step=step0,migration_validation_loss=migrated['validation_loss'],equivalent=True)),flush=True)
    status(step0,'running')
    prefetch = OrderedPrefetch(dataset.root,dataset.rows,consumed_state,a.workers,a.prefetch)
    try:
        for step in range(step0,a.steps):
            tick = time.monotonic()
            pairs, next_state, indices = prefetch.next()
            data_wait = time.monotonic()-tick
            factor=min((step+1)/1000,1.)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-1000)/max(1,a.steps-1000))))
            for group in optim.param_groups:
                group['lr']=1e-4*factor
            optim.zero_grad(set_to_none=True)
            total, components=batched_forward(model,dino,pairs,device,a.microbatch,backward=True)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optim.step()
            consumed_state=next_state
            record=dict(step=step+1,loss=total,gradient_norm=float(norm),lr=optim.param_groups[0]['lr'],
                        elapsed_seconds=time.monotonic()-started,loss_components=components,
                        update_seconds=time.monotonic()-tick,data_wait_seconds=data_wait,
                        sample_indices=indices)
            if (step+1)%2000==0:
                status(step+1,'validating',record)
                result=validate(model,dino,val,rows,device,a.microbatch)
                record.update(validation_loss=result['validation_loss'],validation_by_layout=result['validation_by_layout'],
                              validation_diagnostics=result['diagnostics'],validation_windows=len(rows),
                              validation_panel_sha256=panel['panel_sha256'])
                if best_validation is None or record['validation_loss']<best_validation:
                    best_validation,best_step=record['validation_loss'],step+1
                    checkpoint(step+1,'best.pt')
            record['elapsed_training_seconds']=prior_elapsed+time.monotonic()-started
            with log.open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            limited=a.max_updates is not None and step+1-step0>=a.max_updates
            if (step+1)%100==0 or step+1==a.steps or limited or stop_requested:
                checkpoint(step+1)
            if (step+1)%10==0 or step==step0:
                status(step+1,'running',record)
                print(json.dumps(record),flush=True)
            if limited or stop_requested:
                break
        status(step+1,'complete' if step+1==a.steps else 'checkpointed',record)
    finally:
        prefetch.close()


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        import sys
        if '--output' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--output')+1])/'status.json'
            if path.exists():
                state=read_json(path)
                if state.get('pid')==os.getpid():
                    state.update(status='failed',error=str(exc),updated_unix=time.time())
                    atomic_json(path,state)
        raise
