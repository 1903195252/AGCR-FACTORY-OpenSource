"""Matched TRAIN-only four-arm historical identity and measurement adaptation.

The validated immutable cache from train_history_surface is consumed read-only.
No future labels, future cameras, renderer or evaluation reader are trained.
"""
import argparse
import fcntl
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import torch

from .history_identity import (HistoryIdentityRefiner, historical_parameters,
    identity_objective, install_identity_candidate, overlay_tensors)
from .region_probe import atomic_json, sha256
from .state_repair import load_parent, train_records
from .train_candidate import atomic_checkpoint, branch_index, supervise
from .train_history_surface import validate_cache_row

SCHEMA = 'agcr-history-identity-1'
SOURCES = ('experiments/research_extensions/history_identity.py',
    'experiments/research_extensions/train_history_identity.py',
    'experiments/research_extensions/history_surface_refiner.py',
    'experiments/research_extensions/train_history_surface.py',
    'experiments/paper_experiments/agcr_exp/observation.py',
    'experiments/paper_experiments/agcr_exp/temporal_state.py',
    'experiments/paper_experiments/agcr_exp/stage2_system.py',
    'experiments/paired_interventions/upstream_runtime.py')


@lru_cache(maxsize=1)
def expected_overlay_spec():
    """Exact fixed-width production tensor contract without loading big weights."""
    from experiments.paper_experiments.agcr_exp.temporal_state import TemporalState
    statistics = dict(state=dict(mean=[0.]*11, std=[1.]*11), acceleration=dict(mean=[0.]*4, std=[1.]*4),
        class_velocity={str(k): dict(mean=[0., 0., 0.], covariance=torch.eye(3).tolist()) for k in range(1, 5)})
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Module()
        model.observation = torch.nn.Module()
        model.observation.association = torch.nn.Sequential(torch.nn.Linear(515, 256), torch.nn.GELU(), torch.nn.Linear(256, 1))
        model.tracker = TemporalState(statistics, model.observation.association)
    return {key: (tuple(value.shape), value.dtype) for key, value in overlay_tensors(model).items()}


def read_cache(cache, parent_sha, panel_sha, index_sha, records):
    cache = Path(cache)
    manifest = json.loads((cache/'manifest.json').read_text())
    complete = json.loads((cache/'complete.json').read_text())
    data = manifest.get('data', {})
    if (manifest.get('schema') != 'agcr-history-surface-refiner-1-cache'
            or manifest.get('parent', {}).get('sha256') != parent_sha
            or data.get('panel_sha256') != panel_sha or data.get('index_sha256') != index_sha
            or data.get('split') != 'train' or data.get('test_used') is not False
            or data.get('selection_uses_validation_scores') is not False or data.get('branches') != records
            or complete != dict(completed=True, rows=len(records), manifest_sha256=sha256(cache/'manifest.json'))):
        raise ValueError('Immutable historical cache lineage/completion mismatch')
    # Reuse only features produced by the exact same frozen inference sources.
    root = Path(__file__).resolve().parents[2]
    for relative, digest in manifest.get('observation_sources', {}).items():
        if sha256(root/relative) != digest:
            raise ValueError('Frozen cache source changed: '+relative)
    rows = []
    for i, record in enumerate(records):
        path = cache/('%04d.pt' % i)
        stamp = dict(metadata=record, sha256=sha256(path), cache_manifest_sha256=sha256(cache/'manifest.json'))
        if json.loads(path.with_suffix('.json').read_text()) != stamp:
            raise ValueError('Historical cache row changed: '+str(path))
        rows.append(path)
    return manifest, rows


def validate_checkpoint(payload, parent_path, index_sha=None, require_complete=False):
    identity = payload.get('identity', {})
    config, data = identity.get('configuration', {}), identity.get('data', {})
    if payload.get('schema') != SCHEMA or payload.get('parent', {}).get('sha256') != sha256(parent_path):
        raise ValueError('Historical identity schema/parent mismatch')
    if (config.get('support_mode') not in ('coarse', 'native') or config.get('objective') not in ('posterior', 'full')
            or config.get('future_supervision') is not False or config.get('input_scope') != 'six_historical_RGB_D_frames_only'
            or data.get('split') != 'train' or data.get('test_used') is not False
            or data.get('selection_uses_validation_scores') is not False or not data.get('branches')
            or any(r.get('split') != 'train' or r.get('camera') != 0 or r.get('world') not in (0, 1) for r in data['branches'])):
        raise ValueError('Historical identity training scope invalid')
    for name in ('panel_sha256', 'index_sha256'):
        value = data.get(name, '')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid recorded '+name)
    if index_sha is not None and data['index_sha256'] != index_sha:
        raise ValueError('Historical identity native index mismatch')
    step, target = payload.get('step'), payload.get('target_steps')
    if (type(step) is not int or type(target) is not int or not 0 <= step <= target or target < 1
            or config.get('steps') != target or payload.get('completed') is not (step == target)):
        raise ValueError('Historical identity update contract invalid')
    if require_complete and not payload['completed']:
        raise ValueError('Incomplete identity candidate may not condition evaluation')
    with torch.random.fork_rng(devices=[]):
        model = HistoryIdentityRefiner(**payload['refiner_config'])
    expected, actual = model.state_dict(), payload.get('refiner_state_dict', {})
    if set(expected) != set(actual) or config['support_mode'] != payload['refiner_config']['support_mode']:
        raise ValueError('Historical identity refiner contract invalid')
    for key, value in actual.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                or value.dtype != expected[key].dtype or not torch.isfinite(value).all()):
            raise ValueError('Invalid identity refiner tensor '+key)
    overlay = payload.get('historical_overlay', {})
    allowed = ('tracker.association.', 'tracker.reliability.', 'tracker.memory.', 'tracker.token.',
               'tracker.presence.', 'tracker.readout.', 'tracker.appearance.', 'observation.association.')
    specs = expected_overlay_spec()
    if (set(overlay) != set(specs) or
            any((key != 'tracker.log_process' and not key.startswith(allowed))
                or not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                or tuple(value.shape) != specs[key][0] or value.dtype != specs[key][1]
                for key, value in overlay.items())):
        raise ValueError('Identity overlay includes forbidden/non-finite tensors')
    return dict(schema=SCHEMA, arm=config['support_mode']+'_'+config['objective'],
        support_mode=config['support_mode'], objective=config['objective'], step=step, target_steps=target,
        completed=payload['completed'], identity=identity, parent=payload['parent'],
        refiner_config=payload['refiner_config'], diagnostic_only=True)


def load_identity_checkpoint(path, parent_path, index_sha, upstream=None, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    metadata = validate_checkpoint(payload, parent_path, index_sha, require_complete=True)
    root = Path(__file__).resolve().parents[2]
    for relative in SOURCES:
        if payload['identity'].get('sources', {}).get(relative) != sha256(root/relative):
            raise ValueError('Historical identity source changed: '+relative)
    refiner = HistoryIdentityRefiner(**payload['refiner_config']).to(device)
    refiner.load_state_dict(payload['refiner_state_dict'], strict=True)
    refiner.eval().requires_grad_(False)
    overlay = payload['historical_overlay']
    if upstream is not None:
        install_identity_candidate(upstream, refiner, overlay)
    metadata['checkpoint_sha256'] = sha256(path)
    metadata['inference_inputs'] = 'historical_RGB_D_K_T_and_predicted_queries_only'
    return refiner, overlay, metadata


def run(args):
    from experiments.paired_interventions.data import to_device
    from experiments.paired_interventions.world import load_upstream
    paths = json.loads(Path(args.paths).read_text())
    out = Path(args.output)
    status = dict(schema=SCHEMA+'-report', state='initializing', completed=False, step=0,
                  target_steps=args.steps, arm=args.support_mode+'_'+args.objective)
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    def stopped():
        return stop[0] or bool(args.watchdog_stop and Path(args.watchdog_stop).exists()) or bool(args.stop_file and Path(args.stop_file).exists())
    def mark(stage, **values):
        status.update(stage=stage, updated_unix=time.time(), **values)
        atomic_json(out/'status.json', status)
        if args.worker_report:
            atomic_json(args.worker_report, status)
    try:
        torch.set_num_threads(args.threads)
        device = torch.device(args.device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        parent = load_parent(args.parent)
        records = train_records(args.panel, paths['families'])
        parent_info = dict(path=str(Path(args.parent).resolve()), sha256=sha256(args.parent), schema=parent.get('schema'))
        data = dict(panel_path=str(Path(args.panel).resolve()), panel_sha256=sha256(args.panel), index_sha256=sha256(paths['families']),
            branches=records, split='train', test_used=False, selection_uses_validation_scores=False)
        manifest, row_paths = read_cache(args.cache, parent_info['sha256'], data['panel_sha256'], data['index_sha256'], records)
        config = dict(support_mode=args.support_mode, objective=args.objective, steps=args.steps,
            warmup_steps=args.warmup_steps, lr=args.lr, seed=args.seed, optimizer='AdamW', weight_decay=.01,
            gradient_clip=1., batch_worlds=1, future_supervision=False, input_scope='six_historical_RGB_D_frames_only',
            loss_weights='unit weight for measurement terms and selected historical terms',
            trainable_scope='same refiner, identity residual, existing historical tracker and birth association in all arms',
            matched_exposures_not_equal_compute=True)
        workspace = Path(__file__).resolve().parents[2]
        identity = dict(configuration=config, data=data, sources={name: sha256(workspace/name) for name in SOURCES},
            statistics=dict(path=paths['statistics'], sha256=sha256(paths['statistics'])),
            frozen_cache_manifest_sha256=sha256(Path(args.cache)/'manifest.json'))
        system, _ = load_upstream(paths['upstream'], paths['statistics'], device)
        system.load_state_dict({k[len('upstream.'):]: v for k, v in parent['model'].items() if k.startswith('upstream.')}, strict=True)
        del parent
        torch.manual_seed(args.seed)
        refiner = HistoryIdentityRefiner(channels=args.channels, query_chunk=args.query_chunk,
                                         support_mode=args.support_mode).to(device).train()
        parameters = historical_parameters(system, refiner)
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
        step = 0
        if args.resume:
            payload = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
            validate_checkpoint(payload, args.parent, data['index_sha256'])
            if payload['identity'] != identity or payload['refiner_config'] != refiner.config:
                raise ValueError('Identity experiment resume changed configuration/data/source')
            refiner.load_state_dict(payload['refiner_state_dict'], strict=True)
            current = system.state_dict()
            if set(payload['historical_overlay']) != set(overlay_tensors(system)):
                raise ValueError('Resume historical tensor keys changed')
            with torch.no_grad():
                for key, value in payload['historical_overlay'].items():
                    current[key].copy_(value.to(current[key]))
            optimizer.load_state_dict(payload['optimizer'])
            torch.set_rng_state(payload['rng']['cpu'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state(payload['rng']['cuda'], device)
            step = payload['step']
        def save():
            payload = dict(schema=SCHEMA, parent=parent_info, identity=identity, refiner_config=refiner.config,
                refiner_state_dict={k: v.detach().cpu() for k, v in refiner.state_dict().items()},
                historical_overlay=overlay_tensors(system), step=step, target_steps=args.steps, completed=step == args.steps,
                optimizer=optimizer.state_dict(), rng=dict(cpu=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state(device).cpu() if device.type == 'cuda' else None))
            validate_checkpoint(payload, args.parent, data['index_sha256'])
            atomic_checkpoint(out/'last.pt', payload)
            mark('checkpoint_saved', checkpoint_step=step, checkpoint_sha256=sha256(out/'last.pt'))
        if not args.resume:
            save()
        mark('fit', state='running', configuration=config, cache_completed=len(records), cache_total=len(records),
             trainable_parameters=sum(p.numel() for p in parameters), refiner_parameters=sum(p.numel() for p in refiner.parameters()))
        while step < args.steps and not stopped():
            started = time.monotonic()
            i = branch_index(step, len(records), args.seed)
            row = torch.load(row_paths[i], map_location='cpu', weights_only=False)
            validate_cache_row(row, records[i])
            row = to_device(row, device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms, diagnostics = identity_objective(system, refiner, row, args.objective, step >= args.warmup_steps)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite historical identity objective')
            loss.backward()
            active = [p for p in parameters if p.grad is not None]
            norm = torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            metrics = dict({key: float(value.detach()) for key, value in terms.items()}, **diagnostics,
                loss=float(loss.detach()), step=step, step_seconds=time.monotonic()-started, gradient_norm=float(norm),
                phase=args.objective if step > args.warmup_steps else 'measurement',
                gradient_parameters=sum(p.numel() for p in active), family_id=row['metadata']['family_id'], world=row['metadata']['world'])
            with (out/'updates.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics, allow_nan=False)+'\n')
            mark('fit', step=step, last_update=metrics)
            del row, loss, terms
            if step % args.save_every == 0 or step == args.steps or stopped():
                save()
        if status.get('checkpoint_step') != step:
            save()
        mark('completed' if step == args.steps else 'paused', state='completed' if step == args.steps else 'paused',
             completed=step == args.steps, step=step)
        return 0
    except Exception as error:
        mark('failed', state='failed', completed=False, error=str(error), traceback=traceback.format_exc())
        return 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('paths', 'parent', 'panel', 'cache', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--support-mode', choices=('coarse', 'native'), required=True)
    p.add_argument('--objective', choices=('posterior', 'full'), required=True)
    p.add_argument('--steps', type=int, default=2048)
    p.add_argument('--warmup-steps', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=17029)
    p.add_argument('--channels', type=int, default=16)
    p.add_argument('--query-chunk', type=int, default=8)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--save-every', type=int, default=32)
    p.add_argument('--max-seconds', type=float, default=7200.)
    p.add_argument('--stop-file')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--worker-report', help=argparse.SUPPRESS)
    p.add_argument('--watchdog-stop', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not (1 <= args.steps <= 4096 and 0 <= args.warmup_steps <= args.steps and 0 < args.lr <= .001
            and math.isfinite(args.max_seconds) and 0 < args.max_seconds <= 14400 and 1 <= args.threads <= 16
            and args.save_every > 0 and 4 <= args.channels <= 32 and 1 <= args.query_chunk <= 16):
        p.error('Finite positive bounded experiment settings required')
    if args.worker_report:
        return run(args)
    output = Path(args.output).resolve()
    protected = [Path(args.parent).resolve().parent, Path(args.cache).resolve(),
                 Path(json.loads(Path(args.panel).read_text())['source_root']).resolve()]
    if any(output == value or value in output.parents or output in value.parents for value in protected):
        raise ValueError('Output overlaps immutable data/cache/parent')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            if not (output/'last.pt').exists():
                raise FileNotFoundError('Resume checkpoint missing')
        elif any(path.name != '.lock' for path in output.iterdir()):
            raise FileExistsError('Use an independent experiment output')
        attempt = output/'attempts'/('%d_%d' % (time.time_ns(), os.getpid()))
        attempt.mkdir(parents=True)
        command = [sys.executable, '-m', 'experiments.research_extensions.train_history_identity',
            *(sys.argv[1:] if argv is None else argv), '--worker-report', str(attempt/'report.json'),
            '--watchdog-stop', str(attempt/'STOP')]
        return supervise(command, output, attempt, args.max_seconds, 90., args.stop_file)


if __name__ == '__main__':
    raise SystemExit(main())
