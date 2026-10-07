"""Bounded native VAL sampling for the isolated PeerDream direction study.

This exporter uses the original 50-step pure-noise solver and stream seeds.
Selected cells are a development panel, never a completed formal corpus.
Future references stay on CPU and do not construct generation conditions.
"""
import argparse
import copy
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from .region_probe import atomic_json, sha256, provenance, install_state_overlay, validate_renderer_weights

SCHEMA = 'agcr-peerdream-direction-samples-1'
SAMPLE_SEEDS = (101, 211, 307, 401, 503, 601, 701, 809)
HISTORY_FIELDS = ('history_state', 'class_id', 'history_visible_fraction', 'history_projection_area')


class BoundedStop(RuntimeError):
    pass


def select_panel(index, panel, family_ids, worlds, cameras, draws):
    if index.get('schema') != 'agcr-intervention-index-1' or not index.get('audit_passed'):
        raise ValueError('An audited native family index is required')
    if not family_ids or len(set(family_ids)) != len(family_ids):
        raise ValueError('An explicit nonempty unique family list is required')
    for name, values, allowed in (('world', worlds, (0, 1)), ('camera', cameras, (0, 1)),
                                  ('draw', draws, tuple(range(8)))):
        if not values or len(set(values)) != len(values) or any(v not in allowed for v in values):
            raise ValueError('Invalid explicit ' + name + ' selection')
    # Unchanged development readout enumerates seeds to filename indices.
    if list(draws) != list(range(len(draws))):
        raise ValueError('Select a seed prefix (draws 0; or 0 1; ... 0..7) for native reader compatibility')
    indexed = {r['family_id']: r for r in index['families']}
    selected = {r['family_id']: r for r in panel['families']}
    if len(indexed) != len(index['families']) or len(selected) != len(panel['families']):
        raise ValueError('Duplicate family identity')
    rows = []
    for name in family_ids:
        if name not in indexed or name not in selected:
            raise ValueError('Family is not in both native index and preregistered development panel: ' + name)
        row = indexed[name]
        if row['split'] != 'val' or selected[name]['split'] != 'val':
            raise ValueError('Sampling is restricted to explicitly selected VAL families')
        records = {(r['world'], r['camera']): r for r in row['records']}
        panel_records = {(r['world'], r['camera']): r for r in selected[name]['records']}
        if len(records) != len(row['records']) or set(records) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise ValueError('Native family must contain exactly four cells')
        for cell in records:
            if cell not in panel_records or any(records[cell][k] != panel_records[cell][k] for k in ('path', 'sha256')):
                raise ValueError('Panel branch differs from audited native branch')
        rows.append(copy.deepcopy(row))
    return rows


def evaluation_index(index, rows, worlds, cameras):
    result = copy.deepcopy(index)
    result.update(development_only=True, diagnostic_only=True, formal_population=False,
                  selection='explicit_VAL_direction_panel', families=[])
    for row in rows:
        row = copy.deepcopy(row)
        row['records'] = [r for r in row['records'] if r['world'] in worlds and r['camera'] in cameras]
        result['families'].append(row)
    return result


def require_immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Cannot reuse output with changed provenance: ' + str(path))
    else:
        atomic_json(path, value)
    return sha256(path)


def validate_candidate(candidate, renderer_sha, state_sha, index_sha, allow_partial=False):
    if candidate.get('schema') != 'agcr-peerdream-direction-candidate-1':
        raise ValueError('Unknown isolated candidate checkpoint schema')
    identity = candidate.get('identity', {})
    if identity.get('schema') != 'agcr-peerdream-direction-training-1':
        raise ValueError('Missing isolated candidate training identity')
    files = identity.get('provenance', {}).get('files', {})
    if files.get('checkpoint', {}).get('sha256') != renderer_sha:
        raise ValueError('Candidate parent renderer differs')
    if identity.get('state_conditioning', {}).get('sha256') != state_sha:
        raise ValueError('Candidate parent state checkpoint differs')
    if identity.get('data', {}).get('index_sha256') != index_sha:
        raise ValueError('Candidate native training index differs')
    branches = identity.get('data', {}).get('branches', [])
    if not branches or any(r.get('split') != 'train' for r in branches):
        raise ValueError('Candidate must have TRAIN-only native branch provenance')
    if candidate.get('step', 0) < 1 or candidate.get('step', 0) > candidate.get('target_steps', 0):
        raise ValueError('Candidate update count invalid')
    if bool(candidate.get('completed')) != (candidate['step'] == candidate['target_steps']):
        raise ValueError('Candidate completion disagrees with actual update count')
    if not candidate['completed'] and not allow_partial:
        raise ValueError('Incomplete candidate requires explicit --allow-partial-candidate')
    return identity


def method_identity(state_conditioning, renderer_candidate=None, history_surface=None):
    """A repaired-state-only sample is never labelled as the untouched parent."""
    repaired = state_conditioning.get('candidate_identity') is not None
    suffix = renderer_candidate['configuration']['arm'] if renderer_candidate else 'parent'
    if repaired:
        suffix = 'state_repair' + ('_' + suffix if renderer_candidate else '')
    if history_surface is not None:
        if history_surface.get('arm') not in ('coarse_control', 'refined'):
            raise ValueError('Unknown historical surface refinement arm')
        suffix = 'history_surface_' + history_surface['arm'] + ('_' + suffix if renderer_candidate else '')
    return 'peerdream_direction_' + suffix


def install_surface_candidate(model, path, parent_path, index_sha, device):
    """Explicit inference extension; the parent S2 schema is never rewritten."""
    if path is None:
        return None
    from .train_history_surface import load_checkpoint
    from .history_surface_refiner import install_history_surface
    refiner, process, metadata = load_checkpoint(path, parent_path, index_sha, device)
    if metadata.get('completed') is not True or metadata.get('step', 0) != metadata.get('target_steps'):
        raise ValueError('Only a completed historical surface candidate can condition video generation')
    if (not isinstance(process, torch.Tensor) or process.shape != model.upstream.tracker.log_process.shape
            or not torch.isfinite(process).all()):
        raise ValueError('Invalid learned historical process covariance')
    install_history_surface(model.upstream, refiner)
    with torch.no_grad():
        model.upstream.tracker.log_process.copy_(process.to(model.upstream.tracker.log_process))
    return dict(metadata, checkpoint_sha256=sha256(path), inference_inputs='historical_RGB_D_K_T_and_predicted_queries_only')


def validate_candidate_tensors(model, candidate):
    validate_renderer_weights(model, candidate)
    current = model.state_dict()
    expected = {key for key in current if not key.startswith('wan.backbone.') or 'lora_' in key}
    if set(candidate['model']) != expected:
        raise ValueError('Candidate tensor set must exactly match native learned renderer export')
    for key, value in candidate['model'].items():
        if (not isinstance(value, torch.Tensor) or value.shape != current[key].shape
                or value.dtype != current[key].dtype or not torch.isfinite(value).all()):
            raise ValueError('Candidate tensor shape/dtype/value invalid: ' + key)
        if key.startswith('upstream.') and not torch.equal(value.cpu(), current[key].cpu()):
            raise ValueError('Renderer candidate changed the frozen S2/derived upstream state: ' + key)


def load_history_branch(path):
    from experiments.paired_interventions.data import load_branch
    branch = load_branch(path)
    if (tuple(branch['inputs']['rgb'].shape) != (6, 2, 480, 720, 3)
            or tuple(branch['supervision']['rgb'].shape) != (16, 2, 480, 720, 3)):
        raise ValueError('Unchanged six-history/sixteen-future native 720x480 data required')
    # CPU-side references are discarded before anything is moved to the GPU.
    return dict(inputs=branch['inputs'], supervision={k: branch['supervision'][k] for k in HISTORY_FIELDS})


def validate_cell_marker(root, family, world, camera, draw, manifest_sha):
    from experiments.base48.sample import file_name, validate_arrays
    file = Path(root) / family / file_name(world, camera, draw)
    marker = file.with_suffix('.complete.json')
    if not marker.exists():
        return None
    row = json.loads(marker.read_text())
    if (row.get('manifest_sha256') != manifest_sha or row.get('seed') != SAMPLE_SEEDS[draw]
            or row.get('world') != world or row.get('camera') != camera or row.get('draw') != draw
            or row.get('family_id') != family or not row.get('completed') or not file.exists()
            or row.get('sha256') != sha256(file)):
        raise ValueError('Committed generated clip no longer matches sampling identity: ' + str(file))
    with np.load(file, allow_pickle=False) as arrays:
        validate_arrays(arrays)
    return row


def commit_cell(root, family, world, camera, draw, arrays, manifest_sha, elapsed, gpu, metadata, diagnostics):
    from experiments.base48.sample import publish_cell_file
    folder = Path(root) / family
    name, digest = publish_cell_file(folder, draw, world, camera, arrays)
    row = dict(completed=True, diagnostic_only=True, manifest_sha256=manifest_sha,
               family_id=family, world=world, camera=camera, draw=draw, seed=SAMPLE_SEEDS[draw],
               sha256=digest, seconds=elapsed,
               timing_scope='Condition construction, 50 Euler steps, native decode and CPU output transfer; shared historical state setup excluded',
               **gpu)
    atomic_json((folder / name).with_suffix('.complete.json'), row)
    draw_path = folder / ('draw_%02d.json' % draw)
    packed = json.loads(draw_path.read_text()) if draw_path.exists() else dict(
        seed=SAMPLE_SEEDS[draw], files={}, metadata=metadata, history_assignment={},
        development_only=True, formal_experiment_result=False, whole_family_draw_complete=False)
    if packed['seed'] != SAMPLE_SEEDS[draw]:
        raise ValueError('Existing draw seed changed')
    packed['files'][name] = digest
    packed['history_assignment'][str(world)] = diagnostics
    packed['whole_family_draw_complete'] = len(packed['files']) == 4
    atomic_json(draw_path, packed)
    return row


@torch.no_grad()
def generate_selected_world(model, base, branches, family_id, world, draw_index, training_seed,
                            device, boundary, publish, camera_ids, before_cell=lambda: None,
                            decode_backbone_offload=True):
    """Same state/noise streams, solver, export fields as Base48.generate_family."""
    from experiments.paired_interventions.sample import stream_seed
    from experiments.paired_interventions.sampling import pure_noise_euler
    from experiments.paired_interventions.data import historical_labels
    from experiments.paired_interventions.world import predict_world, world_conditions, query_labels
    from experiments.formal_evaluation.state_likelihood import unique_history_assignment
    from experiments.base48.sample import checked_decode
    from experiments.base48.inference_cache import inference_condition_cache
    seed = SAMPLE_SEEDS[draw_index]
    generator = lambda tag: torch.Generator(device=device).manual_seed(stream_seed(training_seed, family_id, seed, tag))
    boundary()
    encoded = model.encoders.history(base['inputs']['rgb'])
    history, draw = predict_world(model.upstream, base['inputs'], encoded, generator('state'))
    labels = historical_labels(base)
    raw_ids = query_labels(history, labels)
    eligible = labels['history_detectable'].any(0).any(0)
    assignment = unique_history_assignment(history, raw_ids, eligible)
    diagnostics = dict(assignment['stats'],
        duplicate_present_slot_steps=int(draw['present'][0, :, assignment['duplicate_mask']].sum()),
        unassociated_present_slot_steps=int(draw['present'][0, :, assignment['unassociated_mask']].sum()))
    anchor = encoded[1][-1, :, None]
    if tuple(anchor.shape) != (2, 1, 48, 30, 46):
        raise ValueError('Native anchor latent changed')
    noises = torch.stack([torch.randn(4, 48, 30, 46, device=device, generator=generator(tag)) for tag in ('air', 'ground')])
    for camera in camera_ids:
        before_cell()
        boundary()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        began = time.monotonic()
        current = branches[camera]
        for key in ('rgb', 'depth_mm', 'depth_valid', 'K', 'camera_to_world', 'time_s', 'future_time_s'):
            if not torch.equal(base['inputs'][key], current['inputs'][key]):
                raise ValueError('Passive camera branch changed historical information: ' + key)
        condition = world_conditions(model.upstream, history, draw, current['inputs'])
        def velocity(z, s, conditions):
            boundary()
            return model.wan(z.permute(0, 2, 1, 3, 4), s, conditions).permute(0, 2, 1, 3, 4)
        with inference_condition_cache(model.wan):
            latent = pure_noise_euler(velocity, noises, anchor, condition, 50, 2).permute(0, 2, 1, 3, 4)
        boundary()
        from .execution import offload_backbone
        # Wan is not called during native RGB/depth decoding. Relocate only
        # its backbone, preserving the exact solver, dtypes and decode tensors.
        relocation = offload_backbone(model.wan.backbone, device) if decode_backbone_offload else nullcontext()
        with relocation:
            decoded = checked_decode(model, latent, condition)
        tracks = history['result']['tracks']
        result = dict(
            rgb=((decoded['rgb'].permute(0, 1, 3, 4, 2).clamp(-1, 1) + 1) * 127.5).round().byte(),
            depth_mm=decoded['depth_mm'], validity_probability=decoded['validity_logits'].sigmoid(),
            internal_state=draw['state'][0], state_present=draw['present'][0],
            historical_reference_ids=assignment['ids'], historical_reference_ids_raw=raw_ids,
            historical_selected_slot_mask=assignment['selected_mask'], historical_duplicate_slot_mask=assignment['duplicate_mask'],
            historical_unassociated_slot_mask=assignment['unassociated_mask'], historical_unmatched_slot_mask=assignment['unmatched_mask'],
            history_eligible_reference=eligible, predicted_visible_layers=condition['M'], predicted_full_layers=condition['P'],
            predicted_surface_depth_mm=condition['Z'], initial_state=history['result']['initial_state'],
            class_id=torch.stack([t['class_prob'][1:].argmax() + 1 for t in tracks]) if tracks else raw_ids.new_empty(0),
            time_s=current['inputs']['future_time_s'], K=current['inputs']['K'], camera_to_world=current['inputs']['planned_camera_to_world'])
        arrays = {key: value.cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value) for key, value in result.items()}
        torch.cuda.synchronize(device)
        gpu = dict(peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
        publish(camera, arrays, time.monotonic() - began, gpu, diagnostics)
        del arrays, result, decoded, latent, condition


def run_worker(a):
    from experiments.base48.model import Base48System
    from experiments.base48.sample import _ReleasedTrainingReader
    from experiments.paired_interventions.train import import_weights
    from experiments.paired_interventions.data import to_device
    root = Path(a.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = json.loads(Path(a.paths).read_text())
    index = json.loads(Path(paths['families']).read_text())
    panel = json.loads(Path(a.panel).read_text())
    rows = select_panel(index, panel, a.families, a.worlds, a.cameras, a.draws)
    native_files = {}
    for row in rows:
        for r in row['records']:
            if r['world'] not in a.worlds or r['camera'] not in set(a.cameras + [0]):
                continue
            path = (Path(index['root']) / r['path']).resolve()
            if sha256(path) != r['sha256']:
                raise ValueError('Selected native branch differs from audited SHA: ' + str(path))
            native_files[str(path)] = r['sha256']
    paths['_paths_file'] = a.paths
    witness = provenance(paths, a.renderer_checkpoint, paths['families'], next(iter(native_files)))
    witness['sources'][str(Path(__file__).resolve())] = sha256(__file__)
    execution = Path(__file__).with_name('execution.py')
    witness['sources'][str(execution.resolve())] = sha256(execution)
    for key in ('state_checkpoint', 'candidate_checkpoint', 'history_surface_checkpoint', 'panel'):
        value = getattr(a, key, None)
        if value:
            witness['files'][key] = dict(path=str(Path(value).resolve()), sha256=sha256(value))
    witness['selected_native_branches'] = native_files
    eval_index = evaluation_index(index, rows, a.worlds, a.cameras)
    eval_sha = require_immutable_json(root / 'evaluation_index.json', eval_index)
    renderer = torch.load(a.renderer_checkpoint, map_location='cpu', weights_only=False)
    ri = renderer.get('identity', {})
    if (ri.get('schema') != 'agcr-base48-training-1' or ri.get('phase') != 'video'
            or not renderer.get('completed') or not renderer.get('validation_best')):
        raise ValueError('Completed validated native video renderer required')
    candidate = torch.load(a.candidate_checkpoint, map_location='cpu', weights_only=False) if a.candidate_checkpoint else None
    ci = validate_candidate(candidate, sha256(a.renderer_checkpoint), sha256(a.state_checkpoint),
                            sha256(paths['families']), a.allow_partial_candidate) if candidate else None
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction, device)
    model = Base48System(paths, device, 'video', ri.get('variant_options'))
    validate_renderer_weights(model, renderer)
    import_weights(model, renderer['model'])
    del renderer
    state = install_state_overlay(model, a.state_checkpoint,
                                  expected_index_sha256=sha256(paths['families']))
    if candidate:
        validate_candidate_tensors(model, candidate)
        import_weights(model, candidate['model'])
    surface = install_surface_candidate(model, getattr(a, 'history_surface_checkpoint', None),
        a.state_checkpoint, sha256(paths['families']), device)
    if surface is not None:
        for filename in ('history_surface_refiner.py', 'train_history_surface.py'):
            source = Path(__file__).with_name(filename)
            witness['sources'][str(source.resolve())] = sha256(source)
    state_seed = state['identity']['seed']
    manifest = dict(schema='agcr-intervention-predictions-1', diagnostic_schema=SCHEMA,
        method=method_identity(state, ci, surface), split='val',
        corpus_role='direction_development', development_only=True, diagnostic_only=True, formal_experiment_result=False,
        whole_corpus_completed=False, selected_family_ids=a.families, worlds=a.worlds, cameras=a.cameras, sample_indices=a.draws,
        sample_seeds=[SAMPLE_SEEDS[d] for d in a.draws], sampling_steps=50, conditions='history_inferred_state_only',
        future_reference_conditioning=False, downstream_augmentation=False, output_modality='rgbd', conditioning_modality='rgbd',
        depth_unit='mm_optical_axis_Z', primary_aggregation='development_panel_mean_only',
        same_world_state_draw_across_camera_paths=True, historical_labels_used_for_scoring_only=True,
        unmatched_slots_retained_in_generation=True, auxiliary_training_reader_used=False,
        checkpoint_sha256=sha256(a.candidate_checkpoint or a.renderer_checkpoint),
        renderer_checkpoint_sha256=sha256(a.renderer_checkpoint), state_checkpoint_sha256=sha256(a.state_checkpoint),
        candidate_identity=ci, candidate_completed=None if candidate is None else candidate['completed'],
        candidate_step=None if candidate is None else candidate['step'], renderer_identity=ri, state_conditioning=state,
        training_seed=state_seed, renderer_training_seed=ri['seed'], index_sha256=eval_sha,
        native_index_sha256=sha256(paths['families']), provenance=witness,
        rgb_shape=[16, 2, 480, 720, 3], gpu=torch.cuda.get_device_name(device))
    if state.get('candidate_identity') is not None:
        manifest['state_candidate_identity'] = state['candidate_identity']
        manifest['state_candidate_formal_admission'] = False
    if surface is not None:
        manifest['history_surface_conditioning'] = surface
        manifest['history_surface_formal_admission'] = False
    manifest['execution'] = dict(decode_backbone_offload=a.decode_backbone_offload,
        precision='Original model dtypes unchanged',
        operation='Only Wan backbone moves to CPU during native decode and returns before next cell')
    manifest_sha = require_immutable_json(root / 'manifest.json', manifest)
    del candidate
    model.reader = _ReleasedTrainingReader()
    model.eval().requires_grad_(False)
    began = time.monotonic()
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    def boundary():
        if stopped[0] or (a.stop_file and Path(a.stop_file).exists()) or time.monotonic() - began >= a.max_seconds:
            raise BoundedStop('Sampling stopped; committed clips can be resumed')
    total = len(rows) * len(a.worlds) * len(a.cameras) * len(a.draws)
    done = 0
    def status(state_name, **extra):
        atomic_json(root / 'status.json', dict(state=state_name, completed_clips=done, selected_clips=total,
            diagnostic_only=True, formal_experiment_result=False, whole_corpus_completed=False,
            manifest_sha256=manifest_sha, updated_unix=time.time(), **extra))
    try:
        for row in rows:
            fid = row['family_id']
            records = {(r['world'], r['camera']): r for r in row['records']}
            for world in a.worlds:
                needed = []
                for draw in a.draws:
                    pending = []
                    for camera in a.cameras:
                        marker = validate_cell_marker(root, fid, world, camera, draw, manifest_sha)
                        if marker:
                            done += 1
                        else:
                            pending.append(camera)
                    if pending:
                        needed.append((draw, pending))
                if not needed:
                    continue
                boundary()
                selected_cameras = sorted(set([0] + [c for _, cameras in needed for c in cameras]))
                branches = {c: to_device(load_history_branch(Path(index['root']) / records[world, c]['path']), device)
                            for c in selected_cameras}
                for draw, cameras in needed:
                    status('sampling', family_id=fid, world=world, draw=draw)
                    def publish(camera, arrays, seconds, gpu, diagnostics):
                        nonlocal done
                        clip = commit_cell(root, fid, world, camera, draw, arrays, manifest_sha, seconds, gpu,
                                           {k: v for k, v in row.items() if k != 'records'}, diagnostics)
                        done += 1
                        status('sampling', latest_clip=clip)
                        print(json.dumps(clip), flush=True)
                    generate_selected_world(model, branches[0], branches, fid, world, draw, state_seed,
                                            device, boundary, publish, cameras,
                                            decode_backbone_offload=a.decode_backbone_offload)
                del branches
        state_name = 'selected_panel_completed'
    except BoundedStop:
        state_name = 'bounded_partial'
    status(state_name)
    result = dict(completed=done == total, selected_panel_completed=done == total, whole_corpus_completed=False,
                  diagnostic_only=True, formal_experiment_result=False, generated_cells=done, selected_cells=total,
                  manifest_sha256=manifest_sha, reason=state_name, updated_unix=time.time())
    atomic_json(root / 'completion.json', result)
    return 0 if done == total else 2


def supervise(command, root, seconds, stop_file=None):
    """Hard wall cap includes model loads and hashing, not just denoising."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    attempt = '%d_%d' % (time.time_ns(), os.getpid())
    interrupted = None
    with (root / ('worker_' + attempt + '.log')).open('xb') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                if time.monotonic() - began >= seconds:
                    interrupted = 'wall_budget_exhausted'
                    break
                if stop_file and Path(stop_file).exists():
                    interrupted = 'stop_file'
                    break
                time.sleep(.25)
            if interrupted:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=3)
            code = child.returncode
        except BaseException:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=3)
            raise
    completion = root / 'completion.json'
    if interrupted or code != 0:
        old = json.loads(completion.read_text()) if completion.exists() else {}
        old.update(completed=False, selected_panel_completed=False, whole_corpus_completed=False,
                   diagnostic_only=True, formal_experiment_result=False,
                   reason=interrupted or 'worker_failed', worker_exit_code=code, updated_unix=time.time())
        atomic_json(completion, old)
    report = dict(worker_exit_code=code, stop_reason=interrupted, elapsed_seconds=time.monotonic() - began,
                  max_seconds=seconds, termination_grace_seconds=6, diagnostic_only=True,
                  completed=code == 0 and not interrupted, updated_unix=time.time())
    atomic_json(root / ('attempt_' + attempt + '.json'), report)
    return 0 if report['completed'] else 2


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('paths', 'panel', 'renderer-checkpoint', 'state-checkpoint', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--candidate-checkpoint')
    p.add_argument('--history-surface-checkpoint', help='Completed typed historical RGB-D refiner; original S2 remains the explicit parent')
    p.add_argument('--allow-partial-candidate', action='store_true')
    p.add_argument('--families', nargs='+', required=True)
    p.add_argument('--worlds', nargs='+', type=int, default=[0, 1])
    p.add_argument('--cameras', nargs='+', type=int, default=[0])
    p.add_argument('--draws', nargs='+', type=int, default=[0])
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--stop-file')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--memory-fraction', type=float, default=.92)
    p.add_argument('--decode-backbone-offload', dest='decode_backbone_offload', action='store_true', default=True)
    p.add_argument('--no-decode-backbone-offload', dest='decode_backbone_offload', action='store_false')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if not math.isfinite(a.max_seconds) or not 0 < a.max_seconds <= 4 * 3600:
        p.error('Require a finite positive per-process wall cap no greater than four hours')
    if not 0 < a.memory_fraction <= .95 or not 1 <= a.threads <= 32:
        p.error('Invalid memory fraction or CPU threads')
    if not a.device.startswith('cuda:') or not a.device[5:].isdigit():
        p.error('Explicit cuda:N device required')
    return a


def main(argv=None):
    args = arguments(argv)
    root = Path(args.output).resolve()
    for path in (args.paths, args.panel, args.renderer_checkpoint, args.state_checkpoint, args.candidate_checkpoint, args.history_surface_checkpoint):
        if path:
            source = Path(path).resolve()
            if root == source or root in source.parents:
                raise ValueError('Sampling output must not contain or replace immutable inputs')
    root.mkdir(parents=True, exist_ok=True)
    from experiments.base48.sample import exclusive
    if args.worker:
        try:
            return run_worker(args)
        except BaseException as error:
            atomic_json(root / 'failure.json', dict(type=type(error).__name__, message=str(error),
                traceback=traceback.format_exc(), completed=False, diagnostic_only=True, updated_unix=time.time()))
            raise
    cli = list(sys.argv[1:] if argv is None else argv)
    with exclusive(root / '.sampling.lock'):
        return supervise([sys.executable, '-m', 'experiments.research_extensions.sample_candidate', *cli, '--worker'],
                         root, args.max_seconds, args.stop_file)


if __name__ == '__main__':
    raise SystemExit(main())
