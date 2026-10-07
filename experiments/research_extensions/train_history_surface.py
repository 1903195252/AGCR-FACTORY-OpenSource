"""TRAIN-history-only RGB-D measurement refinement and filter calibration.

Frozen encoder/query caches contain six observed frames only. Native instance
masks and world states are separate historical supervision, never model inputs.
The original model is immutable; the export is an explicitly typed overlay.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from .region_probe import atomic_json, sha256
from .state_repair import load_parent, train_records
from .train_candidate import atomic_checkpoint, branch_index, supervise

SCHEMA = 'agcr-history-surface-refiner-1'
INPUT_KEYS = {'rgb', 'depth_mm', 'depth_valid', 'K', 'camera_to_world', 'time_s'}
LABEL_KEYS = {'history_state', 'class_id', 'instance_id', 'history_instance', 'history_detectable'}


def load_history_branch(path):
    """Read only historical slices; future arrays are never returned/cached.

    NPZ compression may decode a full member before slicing, but no future
    pixels, camera plans, state, masks, or future time enter the returned object.
    """
    from experiments.paired_interventions.data import CLASS_MAP
    def tensor(value):
        array = np.ascontiguousarray(value)
        out = torch.from_numpy(array)
        return out.float() if out.dtype == torch.float64 else out
    with np.load(path, allow_pickle=False) as archive:
        rgb = archive['rgb'][:6].copy()
        depth = archive['depth_mm'][:6].copy()
        valid = archive['valid'][:6].astype(bool)
        K = archive['K'][:6].copy()
        cameras = archive['camera_to_world'][:6].copy()
        times = archive['time_s'][:6].copy()
        state = archive['world_state'][:6].copy()
        instance = archive['instance'][:6].copy()
        classes = archive['class_id'].copy()
        ids = archive['instance_id'].copy()
        detectable = ((archive['visible_fraction'][:6] >= .1) &
                      (archive['projection_area'][:6] >= 144))
    if rgb.shape[:2] != (6, 2) or rgb.shape[2:] != (480, 720, 3):
        raise ValueError('Require six native 480x720 paired historical images')
    if depth.shape != rgb.shape[:-1] or valid.shape != depth.shape or instance.shape != depth.shape:
        raise ValueError('Historical RGB-D/instance shape mismatch')
    if K.shape != (6, 2, 3, 3) or not np.array_equal(K, np.broadcast_to(K[0, 0], K.shape)):
        raise ValueError('Frozen observation head requires fixed shared intrinsics')
    if cameras.shape != (6, 2, 4, 4) or times.shape != (6,) or (times > 0).any() or not (np.diff(times) > 0).all():
        raise ValueError('Invalid historical cameras/time boundary')
    if len(set(map(int, ids))) != len(ids) or state.shape != (6, len(ids), 11) or detectable.shape != (6, 2, len(ids)):
        raise ValueError('Invalid historical target labels')
    if any(int(k) not in CLASS_MAP for k in classes):
        raise ValueError('Unknown native target class')
    inputs = dict(rgb=tensor(rgb), depth_mm=tensor(depth), depth_valid=tensor(valid), K=tensor(K[0, 0]),
                  camera_to_world=tensor(cameras), time_s=tensor(times))
    labels = dict(history_state=tensor(state), class_id=tensor(np.array([CLASS_MAP[int(k)] for k in classes], np.int64)),
                  instance_id=tensor(ids.astype(np.int64)), history_instance=tensor(instance.astype(np.int32)),
                  history_detectable=tensor(detectable))
    return dict(inputs=inputs, labels=labels)


@torch.no_grad()
def freeze_history_row(system, encoders, record, device):
    from experiments.paired_interventions.data import to_device
    value = load_history_branch(record['path'])
    x = to_device(value['inputs'], device)
    dino, vae = encoders.history(x['rgb'])
    pair = system.observation(dino.flatten(0, 1), x['depth_mm'].flatten(0, 1),
        x['depth_valid'].flatten(0, 1), x['K'][None].expand(12, -1, -1),
        x['camera_to_world'].flatten(0, 1), x['time_s'].repeat_interleave(2),
        torch.arange(2, device=device).repeat(6))
    return dict(metadata=record, inputs=value['inputs'], labels=value['labels'],
                encoded=dict(dino=dino.cpu(), vae=vae.cpu()), pair={k: v.cpu() for k, v in pair.items()})


def observations(pair):
    return [{key: value[2*t:2*t+2] for key, value in pair.items()} for t in range(6)]


def validate_cache_row(row, record):
    if row.get('metadata') != record or set(row.get('inputs', {})) != INPUT_KEYS or set(row.get('labels', {})) != LABEL_KEYS:
        raise ValueError('Cache provenance or history-only field boundary changed')
    if set(row.get('encoded', {})) != {'dino', 'vae'}:
        raise ValueError('Invalid frozen historical encoders')
    if row['inputs']['rgb'].shape[:2] != (6, 2) or bool((row['inputs']['time_s'] > 0).any()):
        raise ValueError('Cached inputs exceed historical boundary')
    if row['labels']['history_state'].shape[0] != 6 or row['labels']['history_instance'].shape[:2] != (6, 2):
        raise ValueError('Cached labels exceed historical boundary')
    if row['encoded']['dino'].shape != (6, 2, 864, 768) or row['encoded']['vae'].shape != (6, 2, 48, 30, 46):
        raise ValueError('Invalid native frozen feature shapes')
    if row['pair']['center'].shape != (12, 64, 3) or row['pair']['tokens'].shape != (12, 64, 256):
        raise ValueError('Invalid frozen observation query shapes')


def measurement_objective(refiner, original, refined, auxiliary, labels, position_scale):
    """Labels enter matching and losses only, after native support inference."""
    from experiments.paper_experiments.agcr_exp.temporal_state import historical_assignment
    mapping = historical_assignment(observations(original), labels)
    zero = refined['center'].sum()*0
    mask_bce, mask_dice, center_nll, displacement, errors = [], [], [], [], []
    centers = {}
    for t in range(6):
        for view in range(2):
            frame = 2*t+view
            selected = torch.where(mapping[t][view] >= 0)[0]
            if not len(selected):
                continue
            target = mapping[t][view][selected]
            supports = refiner.support_for_queries(original, auxiliary, frame, selected)
            truth = (labels['history_instance'][t, view][None] == labels['instance_id'][target, None, None]).to(supports)
            p = supports.clamp(1e-6, 1-1e-6)
            mask_bce.append(F.binary_cross_entropy(p, truth))
            mask_dice.append((1-(2*(p*truth).sum((-2, -1))+1)/(p.sum((-2, -1))+truth.sum((-2, -1))+1)).mean())
            error = (refined['center'][frame, selected]-labels['history_state'][t, target, :3])/position_scale
            variance = refined['covariance'][frame, selected]/position_scale.square()
            center_nll.append(.5*(error.square()/variance+variance.log()).mean())
            errors.append((error*position_scale).norm(dim=-1))
            for q, k in zip(selected.tolist(), target.tolist()):
                centers[(t, view, k)] = refined['center'][frame, q]
                previous = centers.get((t-1, view, k))
                if previous is not None:
                    actual = labels['history_state'][t, k, :3]-labels['history_state'][t-1, k, :3]
                    displacement.append(F.l1_loss((refined['center'][frame, q]-previous)/position_scale, actual/position_scale))
    mean = lambda values: torch.stack(values).mean() if values else zero
    terms = dict(mask_bce=mean(mask_bce), mask_dice=mean(mask_dice), center_nll=mean(center_nll),
                 displacement_l1=mean(displacement))
    return sum(terms.values()), terms, dict(measurement_targets=len(centers), consecutive_visible_pairs=len(displacement),
        measurement_position_error_m=float(torch.cat(errors).mean().detach()) if errors else None)


def history_objective(system, refiner, row, fit_posterior=True):
    """Native observations -> refiner -> original six-frame tracker; no GT input."""
    from experiments.paper_experiments.agcr_exp.temporal_state import supervised_loss
    x, labels, pair = row['inputs'], row['labels'], row['pair']
    refined, aux = refiner(pair, x['rgb'].flatten(0, 1), x['depth_mm'].flatten(0, 1),
        x['depth_valid'].flatten(0, 1), x['K'], x['camera_to_world'].flatten(0, 1))
    loss, terms, diagnostic = measurement_objective(refiner, pair, refined, aux, labels, system.tracker.scale.std[:3])
    terms['posterior_nll'] = loss.detach()*0
    if fit_posterior:
        obs = observations(refined)
        result = system.tracker(obs, system.observation, row['encoded']['dino'], row['encoded']['vae'],
                                x['camera_to_world'], x['time_s'])
        _, tracking, ids = supervised_loss(system.tracker, result, obs, labels)
        terms['posterior_nll'] = tracking['posterior']
        loss = loss+tracking['posterior']
        known = ids >= 0
        eligible = labels['history_detectable'].any(0).any(0)
        diagnostic.update(predicted_slots=len(ids), eligible_reference_targets=int(eligible.sum()),
                          matched_unique_reference_targets=len(set(ids[known].tolist())))
        if known.any():
            delta = result['initial_state'][known, :6]-labels['history_state'][-1, ids[known], :6]
            diagnostic.update(posterior_position_error_m=float(delta[:, :3].norm(dim=-1).mean().detach()),
                              posterior_velocity_error_m_s=float(delta[:, 3:6].norm(dim=-1).mean().detach()))
    return loss, terms, diagnostic


def validate_checkpoint(payload, parent_path, index_sha=None, require_complete=False):
    from .history_surface_refiner import HistorySurfaceRefiner
    if payload.get('schema') != SCHEMA or payload.get('parent', {}).get('sha256') != sha256(parent_path):
        raise ValueError('History-surface parent/schema mismatch')
    data = payload.get('identity', {}).get('data', {})
    config = payload.get('identity', {}).get('configuration', {})
    branches = data.get('branches', [])
    if (data.get('split') != 'train' or data.get('test_used') is not False or data.get('selection_uses_validation_scores') is not False
            or not branches or any(r.get('split') != 'train' or r.get('camera') != 0 or r.get('world') not in (0, 1) for r in branches)):
        raise ValueError('Overlay must originate from recorded TRAIN histories only')
    if index_sha is not None and data.get('index_sha256') != index_sha:
        raise ValueError('Native index SHA mismatch')
    for name in ('index_sha256', 'panel_sha256'):
        digest = data.get(name, '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid '+name)
    if config.get('future_supervision') is not False or config.get('frozen_parent_except_log_process') is not True:
        raise ValueError('Invalid historical refinement scope')
    step, target = payload.get('step'), payload.get('target_steps')
    if (type(step) is not int or type(target) is not int or target < 1 or not 0 <= step <= target
            or payload.get('completed') is not (step == target) or config.get('steps') != target):
        raise ValueError('Invalid update/completion contract')
    if require_complete and not (payload['completed'] and step > 0):
        raise ValueError('Require a fully completed positive-step overlay')
    # Validation must not perturb an ongoing experiment's initialization/RNG.
    with torch.random.fork_rng(devices=[]):
        model = HistorySurfaceRefiner(**payload['refiner_config'])
    weights = payload.get('refiner_state_dict', {})
    expected = model.state_dict()
    if set(weights) != set(expected):
        raise ValueError('Unexpected refiner tensor keys')
    for key, value in weights.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape or value.dtype != expected[key].dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid refiner tensor: '+key)
    process = payload.get('tracker_log_process')
    if not isinstance(process, torch.Tensor) or process.shape != (6,) or process.dtype != torch.float32 or not torch.isfinite(process).all():
        raise ValueError('Invalid tracker process parameter')
    arm = config.get('arm')
    if arm not in ('coarse_control', 'refined') or payload['refiner_config'].get('support_mode') != ('native' if arm == 'refined' else 'coarse'):
        raise ValueError('Arm/support-mode mismatch')
    return dict(schema=SCHEMA, arm=arm, step=step, target_steps=target, completed=payload['completed'], identity=payload['identity'],
                refiner_config=payload['refiner_config'], parent=payload['parent'], parent_sha256=payload['parent']['sha256'],
                index_sha256=data['index_sha256'], diagnostic_only=True)


def load_surface_checkpoint(path, parent_path, index_sha, upstream=None, device='cpu'):
    from .history_surface_refiner import HistorySurfaceRefiner, install_history_surface
    payload = torch.load(path, map_location='cpu', weights_only=False)
    metadata = validate_checkpoint(payload, parent_path, index_sha, require_complete=True)
    root = Path(__file__).resolve().parents[2]
    for relative in ('experiments/research_extensions/history_surface_refiner.py',):
        if payload['identity'].get('sources', {}).get(relative) != sha256(root/relative):
            raise ValueError('Refiner inference implementation differs from trained source')
    refiner = HistorySurfaceRefiner(**payload['refiner_config']).to(device)
    refiner.load_state_dict(payload['refiner_state_dict'], strict=True)
    refiner.eval().requires_grad_(False)
    process = payload['tracker_log_process'].to(device)
    if upstream is not None:
        with torch.no_grad():
            upstream.tracker.log_process.copy_(process)
        install_history_surface(upstream, refiner)
    metadata['checkpoint_sha256'] = sha256(path)
    return refiner, process, metadata


def load_checkpoint(path, parent_path, expected_index_sha256, device='cpu'):
    return load_surface_checkpoint(path, parent_path, expected_index_sha256, device=device)


def run(args):
    from experiments.paired_interventions.data import to_device
    from experiments.paired_interventions.world import load_upstream
    from experiments.paper_experiments.agcr_exp.stage2_system import FrozenImageEncoders
    from .history_surface_refiner import HistorySurfaceRefiner
    paths = json.loads(Path(args.paths).read_text())
    out, cache = Path(args.output), Path(args.cache)
    status = dict(schema=SCHEMA+'-report', state='initializing', completed=False, step=0, target_steps=args.steps, arm=args.arm)
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
        workspace = Path(__file__).resolve().parents[2]
        relative_sources = ['experiments/research_extensions/train_history_surface.py', 'experiments/research_extensions/history_surface_refiner.py',
            'experiments/paper_experiments/agcr_exp/observation.py', 'experiments/paper_experiments/agcr_exp/stage2_system.py',
            'experiments/paper_experiments/agcr_exp/temporal_state.py', 'experiments/paired_interventions/video_modules.py',
            'experiments/paired_interventions/world.py', 'experiments/paired_interventions/upstream_runtime.py']
        sources = {p: sha256(workspace/p) for p in relative_sources}
        data = dict(panel_path=str(Path(args.panel).resolve()), panel_sha256=sha256(args.panel), index_sha256=sha256(paths['families']),
                    branches=records, split='train', test_used=False, selection_uses_validation_scores=False)
        config = dict(arm=args.arm, steps=args.steps, warmup_steps=args.warmup_steps, lr=args.lr, seed=args.seed,
                      optimizer='AdamW', weight_decay=.01, gradient_clip=1., batch_worlds=1,
                      future_supervision=False, frozen_parent_except_log_process=True, native_history_frames=6,
                      matched_exposures_not_equal_compute=True, mask_bce_weight=1., mask_dice_weight=1.,
                      center_nll_weight=1., displacement_weight=1., posterior_nll_weight=1.)
        identity = dict(configuration=config, data=data, sources=sources,
                        statistics=dict(path=paths['statistics'], sha256=sha256(paths['statistics'])))
        vae_assets = sorted(p for p in (Path(paths['wan'])/'vae').rglob('*') if p.is_file() and p.suffix in ('.json', '.bin', '.safetensors'))
        if not vae_assets:
            raise ValueError('Frozen historical VAE assets missing')
        # Arm-independent cache contract. Training settings and refiner weights
        # do not enter frozen observations, so both arms consume identical rows.
        cache_contract = dict(schema=SCHEMA+'-cache', parent=parent_info, data=data, statistics=identity['statistics'],
            observation_sources={p: digest for p, digest in sources.items() if not p.endswith('history_surface_refiner.py')},
            dino=dict(path=paths['dino_weights'], sha256=sha256(paths['dino_weights'])),
            vae_assets={str(p): sha256(p) for p in vae_assets})
        cache.mkdir(parents=True, exist_ok=True)
        manifest = cache/'manifest.json'
        system, encoders, row_paths = None, None, []
        with (cache/'.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if manifest.exists() and json.loads(manifest.read_text()) != cache_contract:
                raise ValueError('Frozen history cache identity changed')
            if not manifest.exists():
                atomic_json(manifest, cache_contract)
            for i, record in enumerate(records):
                if stopped():
                    mark('cache_paused', state='paused', cache_completed=i, cache_total=len(records)); return 0
                destination = cache/('%04d.pt' % i)
                stamp = destination.with_suffix('.json')
                if destination.exists() and stamp.exists():
                    expected = dict(metadata=record, sha256=sha256(destination), cache_manifest_sha256=sha256(manifest))
                    if json.loads(stamp.read_text()) != expected:
                        raise ValueError('Frozen history cache row SHA changed')
                else:
                    mark('cache_history', cache_completed=i, cache_total=len(records))
                    if system is None:
                        system, _ = load_upstream(paths['upstream'], paths['statistics'], device)
                        system.load_state_dict({k[len('upstream.'):]: v for k, v in parent['model'].items() if k.startswith('upstream.')}, strict=True)
                        system.eval().requires_grad_(False)
                        encoders = FrozenImageEncoders(paths['dino_code'], paths['dino_weights'], paths['wan']+'/vae', device)
                    value = freeze_history_row(system, encoders, record, device)
                    validate_cache_row(value, record)
                    atomic_checkpoint(destination, value)
                    atomic_json(stamp, dict(metadata=record, sha256=sha256(destination), cache_manifest_sha256=sha256(manifest)))
                    del value
                row_paths.append(destination)
            atomic_json(cache/'complete.json', dict(completed=True, rows=len(records), manifest_sha256=sha256(manifest)))
        del encoders
        if args.cache_only:
            mark('cache_completed', state='completed', completed=True, cache_only=True, cache_completed=len(records), cache_total=len(records)); return 0
        if system is None:
            system, _ = load_upstream(paths['upstream'], paths['statistics'], device)
            system.load_state_dict({k[len('upstream.'):]: v for k, v in parent['model'].items() if k.startswith('upstream.')}, strict=True)
        system.eval().requires_grad_(False)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        torch.manual_seed(args.seed)  # Both arms start with identical parameter draws.
        refiner = HistorySurfaceRefiner(channels=args.channels, query_chunk=args.query_chunk,
            support_mode='native' if args.arm == 'refined' else 'coarse').to(device).train()
        parameters = list(refiner.parameters())+[system.tracker.log_process]
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
        step = 0
        if args.resume:
            payload = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
            validate_checkpoint(payload, args.parent, data['index_sha256'])
            if payload['identity'] != identity or payload['refiner_config'] != refiner.config:
                raise ValueError('Resume experiment identity changed')
            refiner.load_state_dict(payload['refiner_state_dict'], strict=True)
            with torch.no_grad():
                system.tracker.log_process.copy_(payload['tracker_log_process'].to(device))
            optimizer.load_state_dict(payload['optimizer'])
            torch.set_rng_state(payload['rng']['cpu'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state(payload['rng']['cuda'], device)
            step = payload['step']
        def save():
            payload = dict(schema=SCHEMA, parent=parent_info, identity=identity, refiner_config=refiner.config,
                refiner_state_dict={k: v.detach().cpu() for k, v in refiner.state_dict().items()},
                tracker_log_process=system.tracker.log_process.detach().cpu(), step=step, target_steps=args.steps,
                completed=step == args.steps, optimizer=optimizer.state_dict(), cache_manifest_sha256=sha256(manifest),
                rng=dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device).cpu() if device.type == 'cuda' else None))
            validate_checkpoint(payload, args.parent, data['index_sha256'])
            atomic_checkpoint(out/'last.pt', payload)
            mark('checkpoint_saved', checkpoint_step=step, checkpoint_sha256=sha256(out/'last.pt'))
        if not args.resume:
            save()
        mark('fit', state='running', configuration=config, cache_completed=len(records), cache_total=len(records),
             refiner_parameters=sum(p.numel() for p in refiner.parameters()), process_parameters=6)
        while step < args.steps and not stopped():
            started = time.monotonic()
            index = branch_index(step, len(records), args.seed)
            row = torch.load(row_paths[index], map_location='cpu', weights_only=False)
            validate_cache_row(row, records[index])
            row = to_device(row, device)
            fit_posterior = step >= args.warmup_steps
            system.tracker.log_process.requires_grad_(fit_posterior)
            optimizer.zero_grad(set_to_none=True)
            loss, terms, diagnostic = history_objective(system, refiner, row, fit_posterior=fit_posterior)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite history-surface objective')
            loss.backward()
            active = [p for p in parameters if p.grad is not None]
            norm = torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            metrics = dict({k: float(v.detach()) for k, v in terms.items()}, **diagnostic,
                loss=float(loss.detach()), step=step, step_seconds=time.monotonic()-started, gradient_norm=float(norm),
                phase='posterior' if fit_posterior else 'measurement', family_id=row['metadata']['family_id'], world=row['metadata']['world'],
                gradient_parameters=sum(p.numel() for p in active), arm=args.arm)
            metrics.update(mask_loss=metrics['mask_bce']+metrics['mask_dice'],
                           displacement_loss=metrics['displacement_l1'], posterior_loss=metrics['posterior_nll'])
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
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('paths', 'parent', 'panel', 'cache', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--arm', choices=('coarse_control', 'refined'), default='refined')
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--warmup-steps', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=17029)
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--query-chunk', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--save-every', type=int, default=16)
    parser.add_argument('--max-seconds', type=float, default=3600.)
    parser.add_argument('--stop-file')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cache-only', action='store_true')
    parser.add_argument('--worker-report', help=argparse.SUPPRESS)
    parser.add_argument('--watchdog-stop', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not (1 <= args.steps <= 4096 and 0 <= args.warmup_steps <= args.steps and 0 < args.lr <= .001
            and 0 < args.max_seconds <= 14400 and 1 <= args.threads <= 16 and args.save_every > 0
            and 4 <= args.channels <= 32 and 1 <= args.query_chunk <= 16):
        parser.error('Require finite bounded updates, learning rate, warmup, channels, chunk and wall cap')
    if args.worker_report:
        return run(args)
    output = Path(args.output).resolve()
    protected = [Path(args.parent).resolve().parent, Path(args.cache).resolve(),
                 Path(json.loads(Path(args.panel).read_text())['source_root']).resolve()]
    if any(output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError('Output overlaps immutable parent/data or frozen cache directory')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            if not (output/'last.pt').exists():
                raise FileNotFoundError('Resume checkpoint missing')
        elif any(p.name != '.lock' for p in output.iterdir()):
            raise FileExistsError('Use an independent output directory')
        attempt = output/'attempts'/('%d_%d' % (time.time_ns(), os.getpid()))
        attempt.mkdir(parents=True)
        command = [sys.executable, '-m', 'experiments.research_extensions.train_history_surface',
                   *(sys.argv[1:] if argv is None else argv), '--worker-report', str(attempt/'report.json'),
                   '--watchdog-stop', str(attempt/'STOP')]
        return supervise(command, output, attempt, args.max_seconds, 90., args.stop_file)


if __name__ == '__main__':
    raise SystemExit(main())
