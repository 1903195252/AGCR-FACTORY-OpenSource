"""Bounded, independent TRAIN-panel adaptation; never resumes a formal run.

Both arms use identical initialization, samples, seeds and optimizer settings.
Only existing video-stage parameters are updated. A parent watchdog requests a
step-boundary checkpoint before the hard wall deadline; the last atomic saved
checkpoint survives a stalled step or forced termination. Results are diagnostic.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from .region_probe import atomic_json, sha256, provenance, install_state_overlay, validate_renderer_weights

SCHEMA = 'agcr-peerdream-direction-candidate-1'
IDENTITY_SCHEMA = 'agcr-peerdream-direction-training-1'


def locked_train_branches(panel_path, index_path, verify_files=True):
    panel = json.loads(Path(panel_path).read_text())
    index = json.loads(Path(index_path).read_text())
    if panel.get('schema') != 'peerdream-direction-panel-1' or panel.get('test_used') is not False:
        raise ValueError('A locked, metadata-selected direction panel without TEST usage is required')
    if index.get('schema') != 'agcr-intervention-index-1' or not index.get('audit_passed'):
        raise ValueError('An audited native source index is required')
    index_sha = sha256(index_path)
    if panel.get('source_index_sha256') != index_sha:
        raise ValueError('Direction panel source index SHA changed')
    root = Path(panel['source_root']).resolve()
    if root != Path(index['root']).resolve():
        raise ValueError('Panel must preserve the source data root')
    by_id = {row['family_id']: row for row in index['families']}
    if len(by_id) != len(index['families']):
        raise ValueError('Duplicate source family IDs')
    selected = panel.get('families', [])
    if len({r['family_id'] for r in selected}) != len(selected):
        raise ValueError('Duplicate direction-panel families')
    records = []
    for family in selected:
        if family.get('split') not in ('train', 'val') or family != by_id.get(family['family_id']):
            raise ValueError('Panel family changed, is unregistered, or includes TEST')
        if family['split'] != 'train':
            continue  # VAL metadata is permitted in the panel; VAL files never enter training.
        rows = family['records']
        if len(rows) != 4 or {(r['world'], r['camera']) for r in rows} != {(w, c) for w in (0, 1) for c in (0, 1)}:
            raise ValueError('Every training family must retain its four complete cells')
        for row in sorted(rows, key=lambda r: (r['world'], r['camera'])):
            path = (root/row['path']).resolve()
            if root not in path.parents:
                raise ValueError('Branch path escapes the audited root')
            if row.get('split') not in (None, 'train'):
                raise ValueError('Non-TRAIN branch record')
            if verify_files and sha256(path) != row['sha256']:
                raise ValueError('TRAIN branch differs from audited SHA: ' + str(path))
            records.append(dict(family_id=family['family_id'], split='train', cohort=family['cohort'],
                                receiver=family['receiver'], world=row['world'], camera=row['camera'],
                                path=str(path), sha256=row['sha256']))
    if not records:
        raise ValueError('Direction panel has no training branches')
    return dict(panel_path=str(Path(panel_path).resolve()), panel_sha256=sha256(panel_path),
                index_sha256=index_sha, branches=records, selection_uses_test=False,
                selection_uses_validation_scores=False)


def branch_index(cursor, size, seed):
    if cursor < 0 or size < 1:
        raise ValueError('Nonnegative cursor and nonempty branch stream required')
    epoch, offset = divmod(cursor, size)
    return int(np.random.default_rng(np.random.SeedSequence([seed, 311, epoch])).permutation(size)[offset])


def configuration(args):
    return dict(arm=args.arm, steps=args.steps, seed=args.seed, batch=1,
                reappearance_weight=0. if args.arm == 'control' else args.reappearance_weight,
                lr_new=1e-4, lr_lora=1e-5, weight_decay=.01, betas=[.9, .999], clip_norm=1.,
                schedule='constant_original_video_group_lrs', branch_schedule='seeded_epoch_permutation_v1',
                device=args.device, threads=args.threads, memory_fraction=args.memory_fraction,
                save_every=args.save_every, upstream_frozen=True, history_condition_unchanged=True,
                execution=args.execution,
                depth_loss_unchanged=True, backbone_frozen=True,
                precision='original_frozen_Wan_bf16_and_trainable_fp32_depth_autocast_bf16')


def training_groups(model):
    """Fail before stepping if frozen state/reader/backbone enters an optimizer."""
    groups = model.trainable_groups()
    names = {id(p): name for name, p in model.named_parameters()}
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = []
    for group in groups:
        if group.get('name') not in ('new', 'lora'):
            raise ValueError('Only original video new/LoRA parameter groups may train')
        required_lr = 1e-4 if group['name'] == 'new' else 1e-5
        if group['lr'] != required_lr:
            raise ValueError('Original video parameter-group learning rate changed')
        for parameter in group['params']:
            name = names[id(parameter)]
            allowed = name.startswith(('condition.', 'depth.', 'wan.adapters.', 'wan.gate.', 'wan.encoder.')) or (
                name.startswith('wan.backbone.') and 'lora_' in name)
            if not parameter.requires_grad or not allowed:
                raise ValueError('Frozen or unregistered parameter entered optimizer: ' + name)
            actual.append(id(parameter))
    if not actual or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Optimizer parameter set is empty, duplicated, or incomplete')
    for module in (model.upstream, model.reader, model.encoders.dino, model.encoders.vae):
        if any(p.requires_grad for p in module.parameters()):
            raise ValueError('Historical frontend, reader and frozen image encoders must remain frozen')
    return groups


def capture_rng(generator, device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state(device).cpu() if torch.device(device).type == 'cuda' else None,
                generator=generator.get_state().cpu())


def restore_rng(state, generator, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if torch.device(device).type == 'cuda':
        if state['torch_cuda'] is None:
            raise ValueError('CUDA RNG state missing from candidate checkpoint')
        torch.cuda.set_rng_state(state['torch_cuda'], device)
    generator.set_state(state['generator'])


def validate_resume(payload, identity):
    if payload.get('schema') != SCHEMA or payload.get('identity') != identity:
        raise ValueError('Resume identity differs: configuration, source, data or parent changed')
    step = payload.get('step')
    target = identity['configuration']['steps']
    if (type(step) is not int or not 0 <= step <= target or payload.get('cursor') != step
            or payload.get('target_steps') != target or payload.get('completed') is not (step == target)):
        raise ValueError('Invalid candidate step/cursor/completion contract')
    if not {'model', 'optimizer', 'rng'} <= set(payload):
        raise ValueError('Candidate model/optimizer/RNG missing')


def atomic_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def scalar_metrics(parts):
    result = {}
    for name, value in parts.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = float(value.detach())
        if value is None or isinstance(value, (bool, int)):
            result[name] = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise FloatingPointError('Non-finite training metric: ' + name)
            result[name] = value
    return result


def gradient_metrics(groups):
    summary = {}
    all_parameters = []
    for group in groups:
        gradients = [p.grad for p in group['params'] if p.grad is not None]
        if any(not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('Non-finite gradient in ' + group['name'])
        squared = sum(float(g.detach().float().square().sum()) for g in gradients)
        summary[group['name']] = dict(l2=math.sqrt(squared), tensors_with_grad=len(gradients),
                                     trainable_tensors=len(group['params']))
        all_parameters.extend(group['params'])
    if not any(row['l2'] > 0 for row in summary.values()):
        raise FloatingPointError('All trainable gradients are absent or zero')
    norm = torch.nn.utils.clip_grad_norm_(all_parameters, 1., error_if_nonfinite=True)
    return dict(groups=summary, unclipped_total_l2=float(norm), clip_norm=1.)


def backward_pass(model, branch, generator, step, execution, stage_callback):
    """One complete objective's gradients; caller owns zero/clip/optimizer step."""
    if execution == 'original':
        loss, parts = model(branch, generator, update=step)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError('Invalid scalar candidate loss')
        stage_callback('backward')
        loss.backward()
        return float(loss.detach()), parts, dict(model.profile_times)
    if execution != 'depth_then_flow_backbone_cpu':
        raise ValueError('Unknown candidate execution policy')
    from .execution import staged_video_backward
    result = staged_video_backward(model, branch, generator, stage_callback=stage_callback)
    loss = float(result['loss'])
    if not math.isfinite(loss):
        raise FloatingPointError('Invalid staged candidate loss')
    parts = {key: result[key] for key in ('flow', 'depth', 'predicted_tracks')}
    return loss, parts, result['timings']


def parent_read_only_guard(output, paths, checkpoint, state_checkpoint, panel):
    """Independent candidate output must not be inside any parent run/assets."""
    output = Path(output).resolve()
    if 'agcr-intervention-runs' in output.parts or 'agcr-peerdream-recovery-runs' in output.parts:
        raise ValueError('Preserved formal run directories cannot hold candidate outputs')
    protected = [Path(checkpoint).resolve().parent, Path(state_checkpoint).resolve().parent,
                 Path(paths['upstream']).resolve().parent, Path(paths['wan']).resolve(),
                 Path(paths['dino_code']).resolve(), Path(json.loads(Path(panel).read_text())['source_root']).resolve()]
    if any(output == path or path in output.parents for path in protected):
        raise ValueError('Candidate output overlaps a read-only parent or asset directory')


def worker(args):
    from .region_objective import ReappearanceSystem
    from experiments.paired_interventions.data import load_branch, to_device
    from experiments.paired_interventions.train import import_weights, export_weights
    from experiments.base48.presence_refit import tensor_digest

    report = dict(schema=SCHEMA+'-report', state='initializing', completed=False, diagnostic_only=True,
                  formal_experiment_result=False, step=0, checkpoint_step=None, target_steps=args.steps,
                  started_unix=time.time(), arm=args.arm, writes_original_weights=False)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    out = Path(args.output).resolve()
    last = out/'last.pt'
    step = cursor = 0
    model = optimizer = identity = generator = None

    def stopped():
        return stop[0] or Path(args.watchdog_stop).exists() or (args.stop_file and Path(args.stop_file).exists())

    def status(stage, **extra):
        report.update(stage=stage, updated_unix=time.time(), **extra)
        atomic_json(args.worker_report, report)
        atomic_json(out/'status.json', report)

    def save(reason):
        frozen_current = {name: p._version for name, p in model.upstream.named_parameters()}
        if frozen_current != frozen_versions:
            raise RuntimeError('Frozen upstream changed during candidate training')
        payload = dict(schema=SCHEMA, identity=identity, model=export_weights(model),
                       optimizer=optimizer.state_dict(), rng=capture_rng(generator, device),
                       step=step, target_steps=args.steps, cursor=cursor, completed=step == args.steps,
                       last_metrics=report.get('metrics'), save_reason=reason, saved_unix=time.time(),
                       diagnostic_only=True, formal_experiment_result=False)
        atomic_checkpoint(last, payload)
        report['checkpoint_step'] = step
        report['checkpoint'] = str(last)

    try:
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        status('checking_sources_and_train_panel')
        paths = json.loads(Path(args.paths).read_text())
        parent_read_only_guard(out, paths, args.checkpoint, args.state_checkpoint, args.panel)
        data = locked_train_branches(args.panel, paths['families'])
        paths['_paths_file'] = args.paths
        sources = provenance(paths, args.checkpoint, paths['families'], data['branches'][0]['path'])
        for file in (Path(__file__), Path(__file__).with_name('region_objective.py'),
                     Path(__file__).with_name('execution.py')):
            sources['sources'][str(file.resolve())] = sha256(file)
        if stopped():
            status('stopped_before_model_allocation', state='paused', reason='stop_requested')
            return 0
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
        status('loading_models')
        renderer = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if renderer.get('identity', {}).get('phase') != 'video' or not renderer.get('completed'):
            raise ValueError('Completed original video renderer checkpoint required')
        cfg = configuration(args)
        model = ReappearanceSystem(paths, device, 'video', renderer['identity'].get('variant_options'),
                                   reappearance_weight=cfg['reappearance_weight'])
        validate_renderer_weights(model, renderer)
        import_weights(model, renderer['model'])
        del renderer
        state_conditioning = install_state_overlay(model, args.state_checkpoint,
                                                   expected_index_sha256=data['index_sha256'])
        if not state_conditioning['overlay_applied']:
            raise ValueError('A verified native or derived state overlay is required')
        if state_conditioning.get('candidate_identity') is not None:
            cfg['history_condition_unchanged'] = False
            cfg['history_condition_source'] = 'verified_development_state_repair'
            cfg['history_condition_implementation'] = 'unchanged_native_history_only_inference'
        # install_state_overlay only replaces upstream; do not call the formal
        # sampling installer, which freezes/releases video training modules.
        model.set_phase('video')
        model.train()
        groups = training_groups(model)
        frozen_digest = tensor_digest(model.upstream.state_dict())
        identity = dict(schema=IDENTITY_SCHEMA, configuration=cfg, provenance=sources, data=data,
                        state_conditioning=state_conditioning, frozen_upstream_sha256=frozen_digest,
                        runtime=dict(torch=torch.__version__, torch_cuda=torch.version.cuda,
                                     python=sys.version, gpu_name=torch.cuda.get_device_name(device)),
                        diagnostic_only=True, formal_experiment_result=False)
        if state_conditioning.get('candidate_identity') is not None:
            identity['state_candidate_identity'] = state_conditioning['candidate_identity']
            identity['adaptation_scope'] = 'renderer_only_with_frozen_repaired_historical_state'
        manifest = out/'identity.json'
        optimizer = torch.optim.AdamW(groups, betas=(.9, .999), weight_decay=.01)
        generator = torch.Generator(device=device).manual_seed(args.seed + 100003)
        if args.resume:
            previous = torch.load(last, map_location='cpu', weights_only=False)
            validate_resume(previous, identity)
            if not manifest.exists() or json.loads(manifest.read_text()) != identity:
                raise ValueError('Saved candidate identity manifest changed')
            validate_renderer_weights(model, previous)
            import_weights(model, previous['model'])
            if tensor_digest(model.upstream.state_dict()) != frozen_digest:
                raise ValueError('Candidate checkpoint changed frozen upstream state')
            optimizer.load_state_dict(previous['optimizer'])
            step = previous['step']; cursor = previous['cursor']
            restore_rng(previous['rng'], generator, device)
            report['checkpoint_step'] = step
            del previous
        else:
            if last.exists() or manifest.exists():
                raise FileExistsError('Candidate output already has weights or identity; use exact --resume')
            atomic_json(manifest, identity)
        frozen_versions = {name: p._version for name, p in model.upstream.named_parameters()}
        original_timed = model.timed
        model.profile = True
        @contextmanager
        def traced_stage(name):
            status('forward_backward', failure_stage=name)
            with original_timed(name):
                yield
        model.timed = traced_stage
        report.update(step=step, state='running', identity_path=str(manifest),
                      trainable_parameters=sum(p.numel() for g in groups for p in g['params']),
                      state_conditioning_mode=state_conditioning['mode'])
        # Step zero is resumable if the first actual backward or memory check fails.
        if not args.resume:
            status('saving_initial_candidate')
            save('initialized_before_first_update')
        metrics_path = out/'updates.jsonl'
        while step < args.steps and not stopped():
            selected = data['branches'][branch_index(cursor, len(data['branches']), args.seed)]
            status('loading_train_branch', step=step, cursor=cursor, branch=selected)
            # Recheck selected bytes so a later mutation cannot enter training.
            if sha256(selected['path']) != selected['sha256']:
                raise ValueError('TRAIN branch changed after panel validation')
            branch = load_branch(selected['path'])
            if (branch['inputs']['rgb'].shape != (6, 2, 480, 720, 3)
                    or branch['supervision']['rgb'].shape != (16, 2, 480, 720, 3)):
                raise ValueError('Native paired 6+16 frame 720x480 training required')
            branch['cohort'] = selected['cohort']
            branch = to_device(branch, device)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            began = time.monotonic()
            status('forward_backward', step=step)
            loss, parts, timings = backward_pass(model, branch, generator, step, args.execution,
                                                 lambda name: status('forward_backward', failure_stage=name))
            gradients = gradient_metrics(groups)
            status('optimizer_step', failure_stage='optimizer_step')
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = time.monotonic()-began
            step += 1; cursor += 1
            metrics = dict(step=step, family_id=selected['family_id'], world=selected['world'],
                           camera=selected['camera'], split='train', loss=loss,
                           losses=scalar_metrics(parts), region=scalar_metrics(getattr(model, 'last_region_terms', {})),
                           gradient=gradients, step_seconds=elapsed, paired_clips_per_second=1/max(elapsed, 1e-9),
                           components_seconds=timings, execution=args.execution,
                           attempt=str(Path(args.worker_report).parent),
                           peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                           updated_unix=time.time())
            report.update(step=step, cursor=cursor, metrics=metrics)
            with metrics_path.open('a') as stream:
                stream.write(json.dumps(metrics, allow_nan=False)+'\n')
                stream.flush()
            del loss, parts, branch
            if step % args.save_every == 0 or step == args.steps or stopped():
                status('saving_checkpoint')
                save('complete' if step == args.steps else 'periodic_or_requested_stop')
            status('step_complete')
        if report.get('checkpoint_step') != step:
            status('saving_stop_checkpoint')
            save('step_boundary_stop')
        status('complete' if step == args.steps else 'paused', completed=step == args.steps,
               state='completed' if step == args.steps else 'paused', step=step,
               reason='target_updates_completed' if step == args.steps else 'stop_requested',
               finished_unix=time.time())
        return 0
    except Exception as error:
        # Never commit a half-update after OOM or a failed backward/optimizer.
        # The previous last.pt remains the only admitted resume state.
        oom = isinstance(error, torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower()
        status(report.get('stage', 'failed'), state='out_of_memory' if oom else 'failed', completed=False,
               error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
               finished_unix=time.time())
        return 1


def supervise(command, output, attempt, seconds, grace_seconds, stop_file=None):
    """Soft stop is inside the wall cap; hard kill never waits for another step."""
    output, attempt = Path(output), Path(attempt)
    report_path, internal_stop, log_path = (attempt/name for name in ('report.json', 'STOP', 'worker.log'))
    began = time.monotonic()
    hard_at = began + seconds
    soft_at = hard_at - min(grace_seconds, seconds*.25)
    requested = [False]
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in old_handlers:
        signal.signal(sig, lambda *_: requested.__setitem__(0, True))
    timed_out = False
    child = None
    try:
        with log_path.open('xb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                now = time.monotonic()
                external_stop = stop_file and Path(stop_file).exists()
                if requested[0] or external_stop or now >= soft_at:
                    if not internal_stop.exists():
                        atomic_json(internal_stop, dict(reason='user_stop' if requested[0] or external_stop else 'bounded_wall_soft_stop',
                                                        requested_unix=time.time()))
                    if requested[0] or external_stop:
                        hard_at = min(hard_at, now + grace_seconds)
                        # Avoid continually moving the earlier deadline forward.
                        requested[0] = False
                if now >= hard_at:
                    timed_out = True
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    break
                time.sleep(min(.25, max(.001, hard_at-now)))
            code = child.wait(timeout=3)
    except BaseException:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=3)
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    report = json.loads(report_path.read_text()) if report_path.exists() else dict(completed=False)
    if timed_out:
        report.update(state='hard_timeout', completed=False,
                      error='Hard wall deadline reached; last atomic checkpoint retained')
    elif code != 0 and report.get('state') not in ('out_of_memory', 'failed'):
        report.update(state='worker_failed', completed=False, error='Worker exited without a usable result')
    elif report.get('state') not in ('completed', 'paused', 'out_of_memory', 'failed'):
        report.update(state='worker_failed', completed=False, error='Missing terminal worker report')
    report.update(worker_exit_code=code, wall_seconds=time.monotonic()-began, finished_unix=time.time(),
                  max_seconds=seconds, allocated_gpus=1, allocated_gpu_hours=(time.monotonic()-began)/3600.,
                  logfile=str(log_path), attempt=str(attempt))
    atomic_json(report_path, report)
    atomic_json(output/'status.json', report)
    prior = [json.loads(p.read_text()) for p in (output/'attempts').glob('*/report.json')]
    atomic_json(output/'usage.json', dict(allocated_gpu_hours=sum(p.get('allocated_gpu_hours', 0.) for p in prior),
                                         attempts=len(prior), accounting='single_allocated_GPU_wall_time_per_attempt_including_load_save'))
    return 0 if report.get('state') in ('completed', 'paused') else 1


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('paths', 'checkpoint', 'state-checkpoint', 'panel', 'output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--arm', choices=('control', 'reappearance'), required=True)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--seed', type=int, default=17029)
    p.add_argument('--reappearance-weight', type=float, default=.05)
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--stop-file')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--memory-fraction', type=float, default=.90)
    p.add_argument('--execution', choices=('original', 'depth_then_flow_backbone_cpu'),
                   default='depth_then_flow_backbone_cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--save-every', type=int, default=4)
    p.add_argument('--checkpoint-grace-seconds', type=float, default=120.)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--worker-report', help=argparse.SUPPRESS)
    p.add_argument('--watchdog-stop', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not 1 <= args.steps <= 4096 or not 0 <= args.seed < 2**31 or not 1 <= args.save_every <= args.steps:
        # --steps 1 is the explicit native backward/memory preflight.
        if args.steps == 1 and args.save_every == 4 and 0 <= args.seed < 2**31:
            args.save_every = 1
        else:
            p.error('Require 1..4096 steps, a valid seed and save-every within the update budget')
    if (not 0 < args.max_seconds <= 14400 or not 0 < args.checkpoint_grace_seconds <= 300
            or not 0 < args.memory_fraction <= .95 or not 1 <= args.threads <= 32
            or not math.isfinite(args.reappearance_weight) or not 0 < args.reappearance_weight <= 1):
        p.error('Finite positive wall cap <=14400s, grace<=300s, VRAM fraction<=.95 and event weight<=1 required')
    if not args.device.startswith('cuda:') or not args.device[5:].isdigit():
        p.error('Use one explicit cuda:N device')
    if bool(args.worker_report) != bool(args.watchdog_stop):
        p.error('Worker report/stop options must be passed together by the watchdog')
    return args


def main(argv=None):
    args = arguments(argv)
    if args.worker_report:
        return worker(args)
    output = Path(args.output).resolve()
    paths = json.loads(Path(args.paths).read_text())
    parent_read_only_guard(output, paths, args.checkpoint, args.state_checkpoint, args.panel)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError('Another candidate process owns this output')
        if args.resume:
            if not (output/'last.pt').is_file():
                raise FileNotFoundError('Resume requires this output directory\'s last.pt')
        elif any(p.name != '.lock' for p in output.iterdir()):
            raise FileExistsError('A new candidate must use a new empty output directory')
        attempt = output/'attempts'/('%d_%d' % (time.time_ns(), os.getpid()))
        attempt.mkdir(parents=True)
        command = [sys.executable, '-m', 'experiments.research_extensions.train_candidate',
                   *(sys.argv[1:] if argv is None else argv), '--worker-report', str(attempt/'report.json'),
                   '--watchdog-stop', str(attempt/'STOP')]
        return supervise(command, output, attempt, args.max_seconds, args.checkpoint_grace_seconds, args.stop_file)


if __name__ == '__main__':
    raise SystemExit(main())
