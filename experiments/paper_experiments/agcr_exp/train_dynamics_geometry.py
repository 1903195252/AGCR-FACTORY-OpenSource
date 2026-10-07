"""Stage-two training from accepted observation weights on the frozen split."""
import argparse
import copy
import fcntl
import json
import math
import multiprocessing
import os
import random
import shutil
import signal
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .data import Dataset, atomic_json, read_json
from .protocol import configuration
from .selection import validation_mean, validation_panel
from .stage2_data import load_window
from .stage2_system import FrozenImageEncoders, StageTwo, tensors
from .stage2_migration import inherit_records, migration_lineage
from .train_observation import sha256


class WindowPrefetch:
    """Bounded prefetch; checkpoints record consumed, not enqueued, RNG state."""
    def __init__(self, dataset, rng_state, workers=4, ahead=4):
        self.dataset = dataset
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = copy.deepcopy(rng_state)
        self.pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'))
        self.pending = deque()
        for _ in range(ahead):
            self.enqueue()

    def enqueue(self):
        index = int(self.rng.integers(len(self.dataset)))
        auxiliary = int(self.rng.integers(16))
        future = self.pool.submit(load_window, (self.dataset.root, self.dataset.rows[index], auxiliary))
        self.pending.append((future, copy.deepcopy(self.rng.bit_generator.state), index, auxiliary))

    def next(self):
        future, state, index, auxiliary = self.pending.popleft()
        sample = future.result()
        self.enqueue()
        return sample, state, index, auxiliary

    def close(self):
        for future, *_ in self.pending:
            future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)


def build_models(args, statistics, device):
    parent = torch.load(args.observation_checkpoint, map_location='cpu', weights_only=False)
    model = StageTwo(statistics, parent['model'], getattr(args, 'execution_policy', 'standard')).to(device)
    encoders = FrozenImageEncoders(args.dino_code, args.dino_weights, args.vae, device)
    return model, encoders, parent


def sample_loss(model, encoders, sample, device, geometry_chunk):
    batch = tensors(sample, device)
    encoded = encoders.history(batch['inputs']['rgb'])
    auxiliary = encoders.features(batch['auxiliary']['rgb'])
    return model.objective(batch, encoded, auxiliary, geometry_chunk)


@torch.no_grad()
def validate(model, encoders, dataset, rows, device, geometry_chunk, progress=None):
    model.eval()
    values, diagnostics, components = [], [], []
    try:
        for n, row in enumerate(rows):
            sample = load_window((dataset.root, row, 5))
            loss, parts, metrics = sample_loss(model, encoders, sample, device, geometry_chunk)
            value = float(loss)
            if not math.isfinite(value):
                raise RuntimeError('Nonfinite held-out validation loss for '+row['uid'])
            values.append(value)
            components.append({name:float(part) for name, part in parts.items()})
            diagnostics.append(dict(uid=row['uid'], **metrics))
            if progress is not None:
                progress(n+1, len(rows))
    finally:
        model.train()
    mean, by_layout = validation_mean(rows, values)
    return dict(validation_loss=mean, validation_by_layout=by_layout, window_losses=values,
                components=components, diagnostics=diagnostics)


def main():
    parser = argparse.ArgumentParser()
    for name in ('index', 'output', 'observation-checkpoint'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--statistics', default='artifacts/state_statistics.json')
    parser.add_argument('--dino-code', default='third_party/dinov2')
    parser.add_argument('--dino-weights', default='models/dinov2/dinov2_vitb14_pretrain.pth')
    parser.add_argument('--vae', default='models/Wan2.2-TI2V-5B/vae')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--steps', type=int, default=40000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prefetch', type=int, default=8)
    parser.add_argument('--geometry-chunk', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume')
    parser.add_argument('--execution-policy', choices=['standard', 'optimized_v1'], default='standard')
    parser.add_argument('--migrate-from')
    parser.add_argument('--migration-report')
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--development-batch', type=int, default=16)
    parser.add_argument('--validation-minimum', type=int, default=64)
    parser.add_argument('--validation-every', type=int, default=2000)
    parser.add_argument('--max-seconds', type=float)
    args = parser.parse_args()
    if bool(args.migrate_from) != bool(args.migration_report):
        raise ValueError('Migration requires both a checkpoint and its verification report')
    if not args.development and (args.steps != 40000 or args.development_batch != 16 or args.validation_minimum != 64 or args.validation_every != 2000 or args.max_updates is not None):
        raise ValueError('Changed protocol requires --development')
    if min(args.steps, args.workers, args.prefetch, args.geometry_chunk, args.development_batch, args.validation_every) < 1:
        raise ValueError('Nonpositive execution settings')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    index, statistics = read_json(args.index), read_json(args.statistics)
    config = configuration('full', args.seed)
    if index['development_only'] and not args.development:
        raise ValueError('Formal training requires fully accepted data')
    if statistics.get('role') != 'train' or statistics.get('index_sha256') != sha256(args.index) or statistics.get('split_sha256') != index['split_sha256']:
        raise ValueError('Statistics were not fitted on this frozen training split')
    train, val = Dataset(index, 'train'), Dataset(index, 'val')
    rows, panel = validation_panel(val.rows, args.validation_minimum, config['validation_panel_seed'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output/'manifest.json').exists() and not args.resume:
        raise ValueError('Existing run: use --resume or another output directory')
    model, encoders, parent = build_models(args, statistics, device)
    if parent['manifest']['index_sha256'] != sha256(args.index) or parent['manifest']['protocol_sha256'] != config['protocol_sha256']:
        raise ValueError('Observation checkpoint changed split, variant or seed')
    if parent['manifest']['development_only'] and not args.development:
        raise ValueError('Pilot observation weights cannot initialize formal training')
    if parent['manifest']['dino_sha256'] != sha256(args.dino_weights):
        raise ValueError('Observation/DINO checkpoint mismatch')
    execution = dict(workers=args.workers, prefetch=args.prefetch, geometry_chunk=args.geometry_chunk, clip_microbatch=1)
    execution.update(policy=args.execution_policy, voxel_chunk=65536 if args.execution_policy == 'optimized_v1' else 8192,
                     dynamics_checkpoint_above_tracks=model.dynamics.checkpoint_tracks_above)
    manifest = dict(stage='dynamics_geometry', development_only=args.development,
                    index_sha256=sha256(args.index), split_sha256=index['split_sha256'],
                    protocol_sha256=config['protocol_sha256'], config=config, steps=args.steps,
                    dependency_sha256={'observation':sha256(args.observation_checkpoint)},
                    observation_step=parent['step'], statistics_sha256=sha256(args.statistics),
                    dino_sha256=sha256(args.dino_weights),
                    vae_sha256={p.name:sha256(p) for p in sorted(Path(args.vae).glob('*')) if p.suffix in ('.json', '.safetensors')},
                    source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
                    effective_batch_pairs=args.development_batch, precision='float32', execution=execution,
                    train_windows=len(train), validation_windows=len(rows), validation_panel=panel,
                    validation_every=args.validation_every, torch_version=torch.__version__,
                    limitations=index.get('limitations', []),
                    schedule_note='Completes temporal frontend and appearance-bank training here; stage one supplied framewise observation/paired-association pretraining.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .999), weight_decay=.01)
    step0, best_loss, best_step, prior_elapsed = 0, None, None, 0.
    origin = None
    if args.migrate_from:
        origin = torch.load(args.migrate_from, map_location='cpu', weights_only=False)
        manifest['migration'] = migration_lineage(origin['manifest'], manifest, args.migrate_from, args.migration_report)
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved['manifest'] != manifest:
            raise ValueError('Resume changed source, inputs, protocol, encoders or execution settings')
    elif origin is not None:
        saved = origin
        if saved['step'] != manifest['migration']['origin_step']:
            raise ValueError('Migration checkpoint step does not match verification')
        inherit_records(args.migrate_from, output, manifest, saved)
    else:
        saved = None
    if saved is not None:
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        step0, best_loss, best_step = saved['step'], saved['best_validation_loss'], saved['best_step']
        prior_elapsed = saved['elapsed_training_seconds']
        del saved
    del origin
    if step0 >= args.steps:
        raise ValueError('Checkpoint already completed this stage')
    log = output/'training.jsonl'
    if log.exists():
        retained = [line for line in log.read_text().splitlines() if json.loads(line)['step'] <= step0]
        log.write_text('\n'.join(retained)+'\n')
    frozen = output/'source/agcr_exp'
    if not frozen.exists():
        shutil.copytree(Path(__file__).parent, frozen, ignore=shutil.ignore_patterns('__pycache__'))
    for name, expected in manifest['source_sha256'].items():
        if sha256(frozen/name) != expected:
            raise ValueError('Frozen source mismatch: '+name)
    atomic_json(output/'manifest.json', manifest)
    atomic_json(output/'validation-panel.json', panel)
    atomic_json(output/'state-normalization.json', statistics)
    del parent
    started, stop_requested = time.monotonic(), False
    consumed = copy.deepcopy(rng.bit_generator.state)
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    def status(step, state, record=None, validation_progress=None):
        elapsed = time.monotonic()-started
        atomic_json(output/'status.json', dict(status=state, pid=os.getpid(), complete=step == args.steps,
            completed_steps=step, target_steps=args.steps, best_step=best_step, best_validation_loss=best_loss,
            elapsed_seconds=prior_elapsed+elapsed, session_elapsed_seconds=elapsed, updated_unix=time.time(),
            remaining_seconds=(args.steps-step)*elapsed/(step-step0) if step > step0 else None,
            development_only=args.development, last=record, validation_progress=validation_progress,
            cuda_peak_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else 0.))
    def checkpoint_save(step, name='last.pt'):
        temporary = output/(name+'.tmp')
        torch.save(dict(step=step, manifest=manifest, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        numpy_rng=consumed, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                        best_validation_loss=best_loss, best_step=best_step,
                        elapsed_training_seconds=prior_elapsed+time.monotonic()-started), temporary)
        temporary.replace(output/name)
    model.train()
    checkpoint_save(step0)
    status(step0, 'running')
    prefetch = WindowPrefetch(train, consumed, args.workers, args.prefetch)
    record = None
    step = step0
    try:
        for step in range(step0+1, args.steps+1):
            tick, wait = time.monotonic(), 0.
            optimizer.zero_grad(set_to_none=True)
            factor = min(step/1000, 1.)*(.1+.9*.5*(1+math.cos(math.pi*max(0, step-1-1000)/max(1, args.steps-1000))))
            for group in optimizer.param_groups:
                group['lr'] = 1e-4*factor
            total, components, selected, diagnostics = 0., {}, [], []
            for pair in range(args.development_batch):
                load_started = time.monotonic()
                sample, next_state, index_number, auxiliary = prefetch.next()
                wait += time.monotonic()-load_started
                loss, parts, metrics = sample_loss(model, encoders, sample, device, args.geometry_chunk)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite joint loss: '+sample['uid'])
                (loss/args.development_batch).backward()
                total += float(loss.detach())/args.development_batch
                for key, value in parts.items():
                    components[key] = components.get(key, 0.)+float(value.detach())/args.development_batch
                selected.append([index_number, auxiliary]); diagnostics.append(metrics)
                # A within-update heartbeat is not a completed optimizer update.
                if time.monotonic()-load_started > 15:
                    status(step-1, 'running', dict(updating_step=step, accumulated_pairs=pair+1))
                del loss, parts
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            consumed = next_state
            record = dict(step=step, loss=total, gradient_norm=float(gradient), lr=optimizer.param_groups[0]['lr'],
                          elapsed_seconds=time.monotonic()-started, elapsed_training_seconds=prior_elapsed+time.monotonic()-started,
                          update_seconds=time.monotonic()-tick, data_wait_seconds=wait, loss_components=components,
                          sample_indices=selected, training_diagnostics=diagnostics)
            if step % args.validation_every == 0 or step == args.steps:
                result = validate(model, encoders, val, rows, device, args.geometry_chunk,
                                  lambda done, count: status(step, 'validating', record, dict(done=done, total=count)))
                record.update(validation_loss=result['validation_loss'], validation_by_layout=result['validation_by_layout'],
                              validation_windows=len(rows), validation_panel_sha256=panel['panel_sha256'])
                atomic_json(output/('validation-%06d.json' % step), result)
                if best_loss is None or result['validation_loss'] < best_loss:
                    best_loss, best_step = result['validation_loss'], step
                    checkpoint_save(step, 'best.pt')
            with log.open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
            limited = args.max_updates is not None and step-step0 >= args.max_updates
            timed_out = args.max_seconds is not None and time.monotonic()-started >= args.max_seconds
            if step % 10 == 0 or limited or timed_out or stop_requested or step == args.steps:
                checkpoint_save(step)
            status(step, 'running', record)
            print(json.dumps(record), flush=True)
            if limited or timed_out or stop_requested:
                break
        checkpoint_save(step)
        status(step, 'complete' if step == args.steps else 'checkpointed', record)
    finally:
        prefetch.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import sys
        if '--output' in sys.argv:
            path = Path(sys.argv[sys.argv.index('--output')+1])/'status.json'
            if path.exists():
                state = read_json(path)
                if state.get('pid') == os.getpid():
                    state.update(status='failed', error=str(exc), updated_unix=time.time())
                    atomic_json(path, state)
        raise
