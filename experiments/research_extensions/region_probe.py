"""One read-only native flow forward with reference-defined region accounting.

No optimizer, video decode, Euler sampling or model-file write is performed.
This is an isolated diagnostic, not a generation-quality or convergence result.
The parent process enforces the wall limit even if a CUDA call stops responding.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import torch
from torch.nn import functional as F

SCHEMA = 'agcr-peerdream-region-probe-1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def reappearance_frames(visible, projection_area, history_visible, history_area, present):
    """First three detectable frames after a projected but hidden predecessor.

    Future frame zero uses the final historical observation as predecessor.
    A hidden predecessor needs projected area >=144, so entry from outside the
    image is not counted. Continuing invisibility inside a 3-frame window is
    excluded. This does not require an earlier visible frame in this view;
    first receiver-view appearance after partner-only history is included.
    """
    if visible.ndim != 3 or visible.shape[1] != 2 or projection_area.shape != visible.shape:
        raise ValueError('Visibility/area must have shape [T,2,K]')
    if history_visible.shape != visible.shape[1:] or history_area.shape != visible.shape[1:]:
        raise ValueError('Historical predecessor must have shape [2,K]')
    if present.shape != (visible.shape[0], visible.shape[2]):
        raise ValueError('Presence must have shape [T,K]')
    previous_visible = torch.cat([history_visible[None], visible[:-1]], 0)
    previous_area = torch.cat([history_area[None], projection_area[:-1]], 0)
    # Historical native targets exist; future existence is explicit supervision.
    previous_present = torch.cat([torch.ones_like(present[:1]), present[:-1]], 0)
    starts = visible & ~previous_visible & (previous_area >= 144) & previous_present[:, None] & present[:, None]
    window = starts.clone()
    for offset in (1, 2):
        window[offset:] |= starts[:-offset]
    return window & visible & present[:, None], starts


def reference_regions(supervision):
    """Iterator of CPU masks [T,2,H,W]; no mask is passed to the predictor."""
    y = supervision
    instance = y['future_instance']
    support = y['isolated_support'].bool()
    ids = y['instance_id']
    if instance.ndim != 4 or instance.shape[1] != 2:
        raise ValueError('future_instance must be [T,2,H,W]')
    if support.shape != (*instance.shape[:2], len(ids), *instance.shape[-2:]):
        raise ValueError('isolated_support must be [T,2,K,H,W]')
    if ids.ndim != 1 or len(torch.unique(ids)) != len(ids) or (ids <= 0).any():
        raise ValueError('Positive unique reference instance IDs required')
    visible = (y['visible_fraction'] >= .1) & (y['projection_area'] >= 144)
    history_visible = (y['history_visible_fraction'][-1] >= .1) & (y['history_projection_area'][-1] >= 144)
    window, starts = reappearance_frames(visible, y['projection_area'], history_visible,
                                        y['history_projection_area'][-1], y['present'].bool())
    visible_union = torch.zeros_like(instance, dtype=torch.bool)
    reappear_union = torch.zeros_like(visible_union)
    for k, target_id in enumerate(ids.tolist()):
        mask = instance == target_id
        visible_union |= mask
        reappear_union |= mask & window[:, :, k, None, None]
    full_union = support.any(2)
    # Disjoint global partition. A vehicle covering another belongs to visible,
    # while individual-target occluded masks can overlap other visible targets.
    yield 'all', torch.ones_like(visible_union), {'partition': False}
    yield 'visible', visible_union, {'partition': True}
    yield 'occluded_support', full_union & ~visible_union, {'partition': True}
    yield 'background', ~(full_union | visible_union), {'partition': True}
    yield 'reappearance_first3', reappear_union, {'partition': False}
    for k, target_id in enumerate(ids.tolist()):
        mask = instance == target_id
        metadata = {'target_index': k, 'instance_id': target_id,
                    'transition_frames_by_view': [torch.where(starts[:, v, k])[0].tolist() for v in range(2)],
                    'overlaps_other_target_regions': True, 'partition': False}
        yield 'target_%d/visible' % k, mask, metadata
        yield 'target_%d/occluded_support' % k, support[:, :, k] & ~mask, metadata
        yield 'target_%d/reappearance_first3' % k, mask & window[:, :, k, None, None], metadata


def pooled_region(mask, latent_size):
    if mask.ndim != 4 or mask.shape[1] != 2:
        raise ValueError('Region must be [T,2,H,W]')
    return F.adaptive_avg_pool3d(F.pad(mask.float().permute(1, 0, 2, 3)[:, None],
                                     (8, 8, 0, 0, 0, 0)), latent_size)


def region_statistics(error_squared, weight, domain, mask):
    """Exact partition of the existing latent-weighted scalar objective.

    region/domain is fractional latent-cell attribution, not decoded RGB error.
    Region-normalized MSE uses pooled coverage; original-objective contribution
    uses (weight * region/domain). Disjoint native masks therefore sum back to
    the unchanged weighted loss, including the partial padding boundary cells.
    """
    if error_squared.ndim != 5 or error_squared.shape[0] != 2:
        raise ValueError('Squared flow error must be [2,C,T,H,W]')
    expected = (2, 1, *error_squared.shape[2:])
    if weight.shape != expected or domain.shape != expected:
        raise ValueError('Flow weight/domain shape mismatch')
    for value in (error_squared, weight, domain):
        if not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError('Non-finite or negative error/weight/domain')
    region = pooled_region(mask, error_squared.shape[2:]).to(error_squared.device)
    fraction = torch.where(domain > 0, region / domain.clamp_min(1e-12), torch.zeros_like(region))
    if (fraction > 1.00001).any():
        raise ValueError('Region extends beyond flow domain')
    channels = error_squared.shape[1]
    weighted_error = error_squared * weight
    full_den = float(domain.sum()) * channels
    full_num = float(weighted_error.sum())

    def one(views):
        e, r, d, w, f = (x[views] for x in (error_squared, region, domain, weight, fraction))
        area, total_area = float(r.sum()), float(d.sum())
        numerator = float((e * w * f).sum())
        unweighted = float((e * r).sum())
        view_den = total_area * channels
        native = mask[:, views]
        return dict(native_area_fraction=float(native.float().mean()),
                    latent_area_fraction=area / total_area if total_area else None,
                    latent_coverage_sum=area, empty=area == 0,
                    region_normalized_mse=unweighted / (area * channels) if area else None,
                    original_weight_mean=float((w * f).sum()) / area if area else None,
                    original_loss_contribution_global=numerator / full_den if full_den else None,
                    original_loss_contribution_view=numerator / view_den if view_den else None,
                    original_weighted_error_share=numerator / full_num if full_num else None)
    return dict(combined=one(slice(None)), aerial=one(slice(0, 1)), ground=one(slice(1, 2)))


@torch.no_grad()
def flow_forward(model, inputs, future_rgb, seed, flow_time, device):
    """Strict boundary: model condition only receives inputs, never supervision."""
    history_rng = torch.Generator(device=device).manual_seed(seed)
    _, history, _, condition = model.infer_condition(inputs, history_rng)
    latent = model.vae.encode(torch.cat([inputs['rgb'][-1:], future_rgb], 0))
    if latent.shape != (2, 48, 5, 30, 46):
        raise ValueError('Expected unchanged native Wan latent [2,48,5,30,46]')
    # A separate recorded RNG makes the flow noise invariant to the number of
    # history tracks or random numbers consumed during world prediction.
    noise_rng = torch.Generator(device=device).manual_seed(seed + 1000003)
    noise = torch.randn(latent[:, :, 1:].shape, device=device, dtype=latent.dtype, generator=noise_rng)
    timestep = latent.new_tensor(flow_time)
    corrupted = torch.cat([latent[:, :, :1], (1-timestep)*latent[:, :, 1:] + timestep*noise], 2)
    velocity = model.wan(corrupted, timestep, condition)
    error = (velocity[:, :, 1:] - (noise-latent[:, :, 1:])).square()
    return error.detach().cpu(), len(history['result']['tracks'])


def provenance(paths, checkpoint, index_path, branch_path):
    files = {'paths': paths['_paths_file'], 'checkpoint': checkpoint, 'index': index_path,
             'branch': branch_path}
    for key in ('upstream', 'statistics', 'dino_weights', 'empty_text'):
        files[key] = paths[key]
    root = Path(__file__).resolve().parents[2]
    source = [Path(__file__)]
    # A diagnostic state derivative has its own admission/parameter rules.
    # Pin that implementation as well as the unchanged scientific runtime.
    state_repair = Path(__file__).with_name('state_repair.py')
    if state_repair.exists():
        source.append(state_repair)
    for directory in ('base48', 'paired_interventions', 'paper_experiments/agcr_exp', 'performance'):
        source.extend(p for p in (root / 'experiments' / directory).glob('*.py')
                      if not p.name.startswith('test_'))
    assets = []
    for subfolder in ('vae', 'transformer'):
        directory = Path(paths['wan']) / subfolder
        selected = [p for p in directory.rglob('*') if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin')]
        if not selected:
            raise ValueError('Missing native Wan assets: ' + str(directory))
        assets.extend(selected)
    source.extend(Path(paths['dino_code']).rglob('*.py'))
    return dict(files={key: dict(path=str(value), sha256=sha256(value)) for key, value in files.items()},
                sources={str(p): sha256(p) for p in sorted(set(source))},
                backbone_assets={str(p): sha256(p) for p in sorted(assets)})


def validate_upstream_derivation(parent, candidate, declared_keys):
    """A typed state repair can never modify renderer or undeclared tensors."""
    old, new = parent.get('model', {}), candidate.get('model', {})
    if not old or set(new) != set(old):
        raise ValueError('Derived state tensor set differs from its verified parent')
    if (not isinstance(declared_keys, (list, tuple)) or not declared_keys
            or len(set(declared_keys)) != len(declared_keys)
            or any(not isinstance(key, str) or not key.startswith('upstream.') or key not in old
                   for key in declared_keys)):
        raise ValueError('Derived state must declare unique existing upstream-only trainable tensors')
    allowed = set(declared_keys)
    changed = []
    for key in old:
        before, after = old[key], new[key]
        if (not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor)
                or before.shape != after.shape or before.dtype != after.dtype
                or not torch.isfinite(after).all()):
            raise ValueError('Invalid derived state tensor shape/dtype/value: ' + key)
        if not torch.equal(before, after):
            if key not in allowed:
                raise ValueError('Derived state changed an undeclared or nonupstream tensor: ' + key)
            changed.append(key)
    return sorted(changed)


def load_state_overlay(state_path, ancestors=()):
    """CPU lineage admission; no live model is changed until every check passes."""
    path = Path(state_path).resolve()
    if path in ancestors or len(ancestors) >= 8:
        raise ValueError('Cyclic or excessive state checkpoint lineage')
    source_sha = sha256(path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    lineage = None
    candidate_identity = None
    if state.get('schema') == 'agcr-peerdream-direction-state-repair-1':
        from .state_repair import validate_checkpoint
        parent_record = state.get('parent', {})
        if not isinstance(parent_record.get('path'), str) or not parent_record['path']:
            raise ValueError('State repair requires an explicit parent checkpoint path')
        parent_path = Path(parent_record['path'])
        if not parent_path.is_absolute():
            raise ValueError('State repair parent path must be absolute')
        parent, parent_metadata = load_state_overlay(parent_path, ancestors+(path,))
        parent_sha = parent_metadata['sha256']
        if (parent_record.get('sha256') != parent_sha
                or parent_record.get('schema') != parent.get('schema')):
            raise ValueError('State repair parent SHA/schema changed')
        validate_checkpoint(state, parent, parent_sha)
        # The typed repair declares dynamics-only optimization. This second
        # boundary also prevents accidental reader/renderer tensor replacement.
        allowed = state.get('trainable_keys', [])
        if any(not isinstance(key, str) or not key.startswith('upstream.dynamics.') for key in allowed):
            raise ValueError('State repair scope must be dynamics-only')
        changed = validate_upstream_derivation(parent, state, allowed)
        if (type(state.get('step')) is not int or state['step'] < 1
                or type(state.get('target_steps')) is not int
                or not state['step'] <= state['target_steps']
                or state.get('completed') is not True or state['step'] != state['target_steps']):
            raise ValueError('Only a completed positive-step state repair may condition renderer experiments')
        identity = parent_metadata['identity']
        completed = True
        mode = 'overlay_direction_state_repair'
        lineage = dict(path=str(parent_path.resolve()), sha256=parent_sha,
                       schema=parent.get('schema'), mode=parent_metadata['mode'],
                       ancestor=parent_metadata.get('parent'),
                       parent_signature_and_unchanged_nondynamics_tensors_verified=True)
        candidate_identity = dict(schema=state['schema'], identity=state['identity'],
                                  step=state['step'], target_steps=state['target_steps'], completed=True,
                                  declared_trainable_prefixes=['upstream.dynamics.'],
                                  declared_trainable_keys=sorted(allowed),
                                  changed_keys=changed, diagnostic_only=True,
                                  formal_experiment_result=False)
    elif state.get('schema') == 'agcr-base-presence-refit-1-checkpoint':
        from experiments.base48.presence_refit import validate_derived
        parent_path = Path(state['refit']['parent_checkpoint_path']).resolve()
        parent_sha = sha256(parent_path)
        parent = torch.load(parent_path, map_location='cpu', weights_only=False)
        validate_derived(state, parent, parent_sha)
        identity = parent.get('identity', {})
        completed = parent.get('completed')
        lineage = dict(path=str(parent_path), sha256=parent_sha,
                       parent_signature_and_unchanged_nonpresence_tensors_verified=True,
                       additional_presence_updates=state['refit']['updates_completed'])
        mode = 'overlay_derived_presence_state'
    else:
        identity = state.get('identity', {})
        completed = state.get('completed')
        mode = 'overlay_base_state'
    if identity.get('schema') != 'agcr-base48-training-1' or identity.get('phase') != 'state' or not completed:
        raise ValueError('Overlay requires completed Base state weights or their verified presence derivative')
    metadata = dict(mode=mode, overlay_applied=True, path=str(path), sha256=source_sha,
                    parent=lineage, identity=identity, formal_sampler_admission_checked=False,
                    note='Diagnostic upstream overlay only; original source/protocol quality-gate admission is not claimed')
    if candidate_identity is not None:
        metadata['candidate_identity'] = candidate_identity
    return state, metadata


def install_state_overlay(model, state_path=None, expected_index_sha256=None):
    """Install only verified upstream weights, preserving renderer and reader.

    The original Base/presence admission remains unchanged. A completed typed
    development repair preserves its own identity and verified parent chain;
    it does not inherit the parent's formal experiment completion claim.
    """
    if state_path is None:
        return dict(mode='renderer_embedded_state_diagnostic', overlay_applied=False,
                    formal_sampler_admission_checked=False,
                    note='Embedded renderer state only; not a replay of the separately installed formal S1/S2 state')
    state, metadata = load_state_overlay(state_path)
    if metadata.get('candidate_identity') is not None and expected_index_sha256 is not None:
        candidate_data = metadata['candidate_identity']['identity'].get('data', {})
        if candidate_data.get('index_sha256') != expected_index_sha256:
            raise ValueError('State repair native training index differs from the requested dataset')
    incoming = {key[len('upstream.'):]: value for key, value in state['model'].items() if key.startswith('upstream.')}
    expected = model.upstream.state_dict()
    if set(incoming) != set(expected):
        raise ValueError('Incomplete or extra upstream overlay keys')
    for key, value in incoming.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                or value.dtype != expected[key].dtype or not torch.isfinite(value).all()):
            raise ValueError('Invalid upstream overlay shape/dtype/value: ' + key)
    model.upstream.load_state_dict(incoming, strict=True)
    return dict(metadata, upstream_tensor_count=len(incoming))


def validate_renderer_weights(model, renderer):
    """Match sample.install_checkpoints: all learned renderer keys must exist."""
    required = {key for key in model.state_dict()
                if not key.startswith('wan.backbone.') or 'lora_' in key}
    missing = required - set(renderer['model'])
    if missing:
        raise ValueError('Incomplete renderer weights: ' + str(sorted(missing)[:8]))


def run_worker(args, report):
    from experiments.base48.model import Base48System
    from experiments.paired_interventions.data import load_branch, to_device
    from experiments.paired_interventions.train import import_weights
    from experiments.paired_interventions.video_losses import flow_weights

    def stage(name):
        report.update(stage=name, updated_unix=time.time())
        atomic_json(args.worker_report, report)

    torch.set_num_threads(args.threads)
    stage('loading_index')
    paths = json.loads(Path(args.paths).read_text())
    index = json.loads(Path(paths['families']).read_text())
    if index.get('schema') != 'agcr-intervention-index-1' or not index.get('audit_passed'):
        raise ValueError('An audited native family index is required')
    matches = [row for row in index['families'] if row['family_id'] == args.family and row['split'] == args.split]
    if len(matches) != 1:
        raise ValueError('Exactly one selected TRAIN/VAL family must exist')
    row = matches[0]
    records = [r for r in row['records'] if (r['world'], r['camera']) == (args.world, args.camera)]
    if len(records) != 1:
        raise ValueError('Selected family cell missing or duplicated')
    path = Path(index['root']) / records[0]['path']
    stage('hashing_read_only_inputs')
    paths['_paths_file'] = args.paths
    report['provenance'] = provenance(paths, args.checkpoint, paths['families'], path)
    if report['provenance']['files']['branch']['sha256'] != records[0]['sha256']:
        raise ValueError('Selected branch differs from audited SHA')
    report['family_metadata'] = {k: v for k, v in row.items() if k != 'records'}
    stage('loading_branch')
    branch = load_branch(path)
    if branch['inputs']['rgb'].shape != (6, 2, 480, 720, 3) or branch['supervision']['rgb'].shape != (16, 2, 480, 720, 3):
        raise ValueError('Native 6+16 frame paired 720x480 branch required')
    weight, domain = flow_weights(branch)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.manual_seed(args.seed)
    stage('loading_read_only_models')
    model = Base48System(paths, device, 'video')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if payload.get('identity', {}).get('phase') != 'video' or not payload.get('completed'):
        raise ValueError('Completed learned video renderer checkpoint required')
    validate_renderer_weights(model, payload)
    import_weights(model, payload['model'])
    report['checkpoint_identity'] = payload.get('identity')
    report['checkpoint_step'] = payload.get('step')
    del payload
    stage('installing_optional_read_only_state_overlay')
    report['state_conditioning'] = install_state_overlay(model, args.state_checkpoint,
                                                       expected_index_sha256=sha256(paths['families']))
    model.eval().requires_grad_(False)
    versions = {name: p._version for name, p in model.named_parameters()}
    stage('single_history_condition_and_flow_forward')
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    began = time.monotonic()
    error, tracks = flow_forward(model, to_device(branch['inputs'], device),
                                branch['supervision']['rgb'].to(device), args.seed, args.flow_time, device)
    torch.cuda.synchronize(device)
    if versions != {name: p._version for name, p in model.named_parameters()}:
        raise RuntimeError('Read-only parameter-version check failed')
    report.update(forward_seconds=time.monotonic()-began, inferred_tracks=tracks,
                  peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                  gpu_name=torch.cuda.get_device_name(device),
                  total_vram_gib=torch.cuda.get_device_properties(device).total_memory/2**30,
                  parameters_unchanged=True, inference_calls=1, optimizer_updates=0,
                  original_flow_loss=float((error*weight).sum()/(domain.sum()*48).clamp_min(1)))
    stage('cpu_reference_region_accounting')
    report['regions'] = {name: dict(metadata=metadata, **region_statistics(error, weight, domain, mask))
                         for name, mask, metadata in reference_regions(branch['supervision'])}
    additive = sum(report['regions'][name]['combined']['original_loss_contribution_global']
                   for name in ('visible', 'occluded_support', 'background'))
    if not math.isclose(additive, report['original_flow_loss'], rel_tol=2e-6, abs_tol=1e-7):
        raise AssertionError('Region partition no longer reproduces original flow loss')
    report.update(state='completed', stage='complete', completed=True,
                  partition_sum=additive, finished_unix=time.time())


def initial_report(args):
    return dict(schema=SCHEMA, state='initializing', completed=False, diagnostic_only=True,
                formal_experiment_result=False, writes_weights=False, started_unix=time.time(),
                family=args.family, split=args.split, world=args.world, camera=args.camera,
                requested_state_checkpoint=args.state_checkpoint,
                history_seed=args.seed, flow_noise_seed=args.seed + 1000003, flow_time=args.flow_time,
                configuration={key: getattr(args, key) for key in ('device', 'max_seconds', 'memory_fraction', 'threads')},
                definitions=dict(
                    supervision_boundary='Future RGB only forms the flow target; future masks only partition loss statistics. Conditions receive historical inputs and planned poses only.',
                    region_attribution='Fractional adaptive pooling into existing flow-loss bins; not decoded RGB error or generated-video quality. VAE temporal receptive fields cross these bins.',
                    reappearance='First three frames following hidden-to-detectable transition with previous projected area>=144; detectable means fraction>=0.1 and area>=144; only currently detectable mask pixels count.',
                    partition='visible + occluded_support excluding visible + background is disjoint; target-specific regions and reappearance overlap and must not be summed.',
                    timestep='Fixed post-shift flow time; noise uses an independent recorded seed. Not an exact replay of a training RNG stream.',
                    interpretation='Single-forward supervised flow diagnostic; regional dilution and downstream gains remain hypotheses, not established by one example.'))


def worker_main(args):
    report = initial_report(args)
    try:
        run_worker(args, report)
    except Exception as error:
        oom = isinstance(error, torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower()
        report.update(state='out_of_memory' if oom else 'failed', completed=False,
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                      finished_unix=time.time())
    atomic_json(args.worker_report, report)
    return 0 if report['completed'] else 1


def supervise(command, report_path, worker_path, log_path, seconds):
    """CPU watchdog; child has its own process group for bounded cleanup."""
    started = time.monotonic()
    timed_out = False
    with open(log_path, 'xb') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=3)
        except BaseException:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=3)
            raise
    report = json.loads(Path(worker_path).read_text()) if Path(worker_path).exists() else json.loads(Path(report_path).read_text())
    if timed_out:
        report.update(state='timeout', completed=False, error='Worker exceeded the wall limit; process group terminated')
    elif (code != 0 or not report.get('completed')) and report.get('state') not in ('out_of_memory', 'failed'):
        report.update(state='worker_failed', completed=False, error='Worker exited without a complete result')
    report.update(worker_exit_code=code, wall_seconds=time.monotonic()-started,
                  finished_unix=time.time(), worker_log=str(log_path),
                  max_seconds_includes='Child startup, checksum validation, asset load, forward, and statistics; termination grace <=6 seconds')
    atomic_json(report_path, report)
    return 0 if report.get('completed') else 1


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('paths', 'checkpoint', 'family', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--state-checkpoint', help='Optional completed Base state, verified presence derivative or completed typed development repair; upstream only')
    p.add_argument('--split', choices=('train', 'val'), required=True)
    p.add_argument('--world', type=int, choices=(0, 1), default=0)
    p.add_argument('--camera', type=int, choices=(0, 1), default=0)
    p.add_argument('--seed', type=int, default=17029)
    p.add_argument('--flow-time', type=float, default=.5)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-seconds', type=float, default=600)
    p.add_argument('--memory-fraction', type=float, default=.90)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--worker-report', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not 0 < args.max_seconds <= 600 or not 0 < args.memory_fraction <= .95:
        p.error('Require 0 < max-seconds <= 600 and 0 < memory-fraction <= .95')
    if not 0 < args.flow_time < 1 or not 0 <= args.seed < 2**31 or not 1 <= args.threads <= 32:
        p.error('Require finite flow time in (0,1), seed in [0,2^31), and 1..32 CPU threads')
    if not args.device.startswith('cuda:') or not args.device[5:].isdigit():
        p.error('The native model probe requires an explicit cuda:N device')
    return args


def main(argv=None):
    args = arguments(argv)
    if args.worker_report:
        return worker_main(args)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a new diagnostic report path')
    output.parent.mkdir(parents=True, exist_ok=True)
    worker = output.with_name(output.name + '.worker.json')
    log = output.with_name(output.name + '.worker.log')
    if worker.exists() or log.exists():
        raise FileExistsError('Diagnostic worker artifacts already exist')
    atomic_json(output, initial_report(args))
    cli = list(sys.argv[1:] if argv is None else argv)
    command = [sys.executable, '-m', 'experiments.research_extensions.region_probe', *cli,
               '--worker-report', str(worker)]
    return supervise(command, output, worker, log, args.max_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
