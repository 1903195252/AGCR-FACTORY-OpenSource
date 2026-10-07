"""Full-budget independent-seed replication of the isolated evidence loss.

Both coefficients (0 and 0.1) retain the original 64 paired TRAIN families,
2048 updates, frozen historical checkpoint, original initialization, and every
other loss. This loop is derived from train_future_no_evidence.py; its original
implementation and frozen result records are not modified. Random evidence
rollouts execute even when the coefficient is zero.
"""
import argparse
import fcntl
import json
import math
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
from .state_repair import load_parent, train_records, ordinary_objective
from .train_candidate import atomic_checkpoint, branch_index, supervise
from .train_history_identity import load_identity_checkpoint, read_cache
from .train_history_surface import observations, validate_cache_row

SCHEMA = 'agcr-isolated-evidence-seed-replication-1'
CACHE_SCHEMA = 'agcr-history-future-1-cache'
PREFIXES = ('dynamics.', 'geometry.')
SOURCES = ('experiments/research_extensions/train_history_future.py',
    'experiments/research_extensions/train_history_identity.py',
    'experiments/research_extensions/history_identity.py',
    'experiments/research_extensions/state_repair.py',
    'experiments/base48/model.py', 'experiments/paired_interventions/objectives.py',
    'experiments/paper_experiments/agcr_exp/future_prediction.py',
    'experiments/paper_experiments/agcr_exp/geometry.py',
    'experiments/research_extensions/train_evidence_repeats.py')


def future_tensors(system):
    return {key: value.detach().cpu().clone() for key, value in system.state_dict().items()
            if key.startswith(PREFIXES)}


def validate_checkpoint(payload, parent_path, index_sha=None, require_complete=False):
    config = payload.get('identity', {}).get('configuration', {})
    data = payload.get('identity', {}).get('data', {})
    if (payload.get('schema') != SCHEMA or payload.get('parent', {}).get('sha256') != sha256(parent_path)
            or config.get('arm') != 'paired_geometry'
            or config.get('component_ablation') != 'isolated_evidence_replication'
            or config.get('evidence_forward_retained') is not True
            or config.get('seed') not in (17017, 17029, 17043) or config.get('steps') != 2048
            or config.get('lr') != 1e-5
            or config.get('weights', {}).get('evidence') not in (0., .1)
            or config.get('weights') != dict(evidence=config.get('weights', {}).get('evidence'), motion_projection=.5, common_geometry=.1, reappearance_geometry=.1)
            or config.get('history_frozen') is not True or config.get('future_truth_conditions') is not False
            or config.get('deployment_sampler_changed') is not False
            or config.get('ambiguous_nll_and_presence_only') is not True
            or data.get('split') != 'train' or data.get('test_used') is not False
            or data.get('selection_uses_validation_scores') is not False or not data.get('branches')
            or any(r.get('split') != 'train' or r.get('camera') != 0 for r in data['branches'])):
        raise ValueError('Invalid future adaptation identity or information boundary')
    if index_sha is not None and data.get('index_sha256') != index_sha:
        raise ValueError('Future adaptation index changed')
    step, target = payload.get('step'), payload.get('target_steps')
    if (type(step) is not int or type(target) is not int or not 0 <= step <= target or target < 1
            or config.get('steps') != target or payload.get('completed') is not (step == target)
            or (require_complete and not payload['completed'])):
        raise ValueError('Invalid future adaptation update contract')
    hist = payload.get('history_checkpoint', {})
    if not hist.get('path') or hist.get('sha256') != sha256(hist['path']):
        raise ValueError('Historical checkpoint changed')
    tensors = payload.get('future_overlay', {})
    if not tensors or any(not k.startswith(PREFIXES) or not isinstance(v, torch.Tensor)
            or not torch.isfinite(v).all() for k, v in tensors.items()):
        raise ValueError('Invalid future overlay scope/tensors')
    return dict(schema=SCHEMA, arm=config['arm'], step=step, target_steps=target,
        completed=payload['completed'], history_checkpoint=hist, identity=payload['identity'],
        diagnostic_only=True, inference_inputs='historical_RGB_D_and_known_camera_plans_only')


def load_future_checkpoint(path, parent_path, index_sha, upstream=None, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    metadata = validate_checkpoint(payload, parent_path, index_sha, require_complete=True)
    root = Path(__file__).resolve().parents[2]
    for name in SOURCES:
        if payload['identity']['sources'].get(name) != sha256(root/name):
            raise ValueError('Future adaptation source changed: '+name)
    refiner, historical, history_meta = load_identity_checkpoint(
        payload['history_checkpoint']['path'], parent_path, index_sha, upstream=upstream, device=device)
    if upstream is not None:
        current = upstream.state_dict()
        expected = {key for key in current if key.startswith(PREFIXES)}
        if set(payload['future_overlay']) != expected:
            raise ValueError('Future overlay exact keys mismatch')
        with torch.no_grad():
            for key, value in payload['future_overlay'].items():
                if value.shape != current[key].shape or value.dtype != current[key].dtype:
                    raise ValueError('Future tensor shape/dtype mismatch: '+key)
                if '.scale.' in key and not torch.equal(value, current[key].cpu()):
                    raise ValueError('Frozen training statistics changed: '+key)
                current[key].copy_(value.to(current[key]))
    metadata.update(checkpoint_sha256=sha256(path), historical=history_meta)
    return metadata


def free_mode_means(model, history, horizon=16):
    """History-only conditional-alive mean paths; no reference inputs or masks."""
    if history['empty'] or not 1 <= horizon <= 16:
        raise ValueError('Nonempty history and 1..16 horizon required')
    initial = history['initial']
    logits = model(history['tokens'], history['shared'], initial[:, None])['mode_logits'][0]
    modes = logits.numel()
    expand = lambda x: x.expand(modes, *x.shape[1:])
    tokens, shared = expand(history['tokens']), expand(history['shared'])
    states = [expand(initial)]
    survival = expand(history['presence_logits']).sigmoid()
    alive = []
    for _ in range(horizon):
        out = model(tokens, shared, torch.stack(states, 1), torch.arange(modes, device=initial.device))
        acc = out['acc_mean'][:, 0, -1]*model.scale.acc_std+model.scale.acc_mean
        yaw = out['yaw_mean'][:, 0, -1]
        old = states[-1]
        states.append(torch.cat((old[..., :3]+.2*old[..., 3:6]+.02*acc,
            old[..., 3:6]+.2*acc, ((old[..., 6]+yaw+math.pi)%(2*math.pi)-math.pi)[..., None],
            (yaw/.2)[..., None], old[..., 8:]), -1))
        survival = survival*out['presence_logits'][:, 0, -1].sigmoid()
        alive.append(survival)
    return dict(state=torch.stack(states[1:], 1), survival=torch.stack(alive, 1), mode_logits=logits)


def project_centers(points, K, cameras):
    """Continuous calibrated projection; keeps gradients even outside hard boxes."""
    camera = torch.einsum('tvkj,tvjl->tvkl', points-cameras[..., None, :3, 3], cameras[..., :3, :3])
    pixel = torch.einsum('ij,tvkj->tvki', K, camera)
    return pixel[..., :2]/camera[..., 2:].clamp_min(.1), camera[..., 2]


def free_motion_projection_loss(free, row, scale):
    slots = row['selected_slots']
    if not slots:
        return free['state'].sum()*0, dict(projected_targets=0)
    refs = row['selected_references']
    predicted = free['state'][:, :, slots]
    reference = row['future_state'][:, refs].detach()
    present = row['future_present'][:, refs].detach().float()
    weight = free['mode_logits'].softmax(-1)
    error = F.smooth_l1_loss(predicted[..., :6]/scale.std[:6],
        reference[None, ..., :6].expand_as(predicted[..., :6])/scale.std[:6], reduction='none').mean(-1)
    motion = ((error*present[None]).sum((1, 2))/present.sum().clamp_min(1)*weight).sum()
    # Every mode's center is projected with known planned cameras. Reference
    # projection is a loss target only; target support never constructs a path.
    cameras, K = row['cameras'], row['K']
    target_uv, target_z = project_centers(reference[:, None, :, :3].expand(-1, 2, -1, -1), K, cameras)
    active = (target_z > .1)&present[:, None].bool()
    normalizer = target_uv.new_tensor([720., 480.])
    costs = []
    for state in predicted:
        uv, _ = project_centers(state[:, None, :, :3].expand(-1, 2, -1, -1), K, cameras)
        err = F.smooth_l1_loss(uv/normalizer, target_uv/normalizer, reduction='none').mean(-1)
        costs.append((err*active).sum()/active.sum().clamp_min(1))
    projection = (torch.stack(costs)*weight).sum()
    return motion+projection, dict(free_motion_huber=motion, projected_center_huber=projection,
                                   projected_targets=len(slots))


def evidence_pair_loss(model, rows, references, step, seed):
    from experiments.base48.model import persistent_target_rollouts, evidence_scores
    from experiments.paired_interventions.objectives import candidate_log_density, evidence_preference
    losses, covered = [], 0
    for world, row in enumerate(rows):
        hist, slot = row['history'], row['focal_slot']
        if hist['empty'] or slot < 0:
            losses.append(next(model.parameters()).sum()*0)
            continue
        rng = torch.Generator(device=hist['initial'].device).manual_seed(seed+500009+2*step+world)
        bank = persistent_target_rollouts(model, hist['tokens'], hist['shared'], hist['initial'],
            hist['presence_logits'], rng, slot, covariance=hist['covariance'], draws=2)
        scores = candidate_log_density(bank['state'][..., slot, :6], bank['mode_logits'],
            references[None], model.scale.std[:6], .2, bank['log_survival'])
        losses.append(evidence_preference(evidence_scores(scores, 16*6, 'mean'),
            torch.tensor([world], device=scores.device)))
        covered += 1
    return torch.stack(losses).sum()/2, dict(evidence_covered_worlds=covered, evidence_reference_worlds=2)


def geometry_surrogate(system, free, row, step, reappearance=True):
    from experiments.paper_experiments.agcr_exp.future_prediction import geometry_intrinsics, geometry_loss
    times = sorted({f for event in row['events'] for f in event['frames']}) if reappearance else []
    # Deterministic coverage of the complete scene when a TRAIN world has no
    # reappearance; otherwise cycle all event frames without dropping targets.
    frame = times[step % len(times)] if times else step % 16
    mode = (step//16) % free['state'].shape[0]
    state = free['state'][mode, frame]
    alive = free['survival'][mode, frame]
    history = row['history']; ids = row['reference_ids']; known = ids >= 0
    target = {key: row['geometry_targets'][key][frame, :, ids.clamp_min(0)]*known[None, :, None, None]
              for key in ('isolated_depth_mm', 'full_mask', 'visible_mask')}
    target.update({key: row['geometry_targets'][key][frame] for key in ('static_depth_mm', 'static_support')})
    prediction = system.geometry(history['shared'], history['separate'], history['tokens'][0],
        state[None].expand(2, -1, -1), alive[None].expand(2, -1), row['projection'][2*frame:2*frame+2],
        geometry_intrinsics(row['K']), row['cameras'][frame], row['times'][frame].expand(2),
        torch.arange(2, device=state.device))
    loss, terms = geometry_loss(prediction, target)
    # Uniform deterministic mode cycling, importance weighted to the predicted
    # scene mixture. Labelled as a surrogate, not an exact discrete-sample loss.
    weighted = loss*free['mode_logits'].softmax(-1)[mode]*free['state'].shape[0]
    return weighted, dict(geometry_loss=loss, geometry_frame=frame, geometry_mode=mode,
                         geometry_event_frames=len(times), **{'geometry_'+k: v for k, v in terms.items()})


@torch.no_grad()
def prepare_row(system, refiner, cached, record, focal, device):
    from experiments.paired_interventions.data import to_device
    from experiments.paired_interventions.world import query_labels
    from experiments.base48.model import focal_history_slot
    from experiments.paper_experiments.agcr_exp.stage2_system import matched_state_supervision
    from experiments.paper_experiments.agcr_exp.stage2_data import depth8, area8
    from experiments.paper_experiments.agcr_exp.future_prediction import geometry_intrinsics
    from .region_objective import reappearance_events
    validate_cache_row(cached, record)
    cached = to_device(cached, device); x, labels = cached['inputs'], cached['labels']
    pair, _ = refiner(cached['pair'], x['rgb'].flatten(0, 1), x['depth_mm'].flatten(0, 1),
        x['depth_valid'].flatten(0, 1), x['K'], x['camera_to_world'].flatten(0, 1))
    obs = observations(pair)
    result = system.tracker(obs, system.observation, cached['encoded']['dino'], cached['encoded']['vae'], x['camera_to_world'], x['time_s'])
    shared, separate = system.scene(obs)
    history = dict(empty=not result['tracks'], shared=shared, separate=separate, result=result, observations=obs)
    if history['empty']:
        return dict(metadata=record, empty=True, history=dict(empty=True), eligible_reference_count=int(labels['history_detectable'].any(0).any(0).sum()))
    history.update(tokens=result['tokens'][None], initial=result['initial_state'][None],
        covariance=torch.stack([t['covariance'] for t in result['tracks']])[None],
        presence_logits=torch.stack([t['presence_logit'] for t in result['tracks']])[None])
    ids = query_labels(history, labels)
    eligible = labels['history_detectable'].any(0).any(0)
    selected = [(focal_history_slot(history, ids, k)[0], k) for k in range(len(eligible)) if bool(eligible[k])]
    selected = [(s, k) for s, k in selected if s >= 0]
    # Native future supervision is read only after historical prediction exists.
    if sha256(record['path']) != record['sha256']:
        raise ValueError('Native TRAIN branch changed during cache preparation')
    with np.load(record['path'], allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in ('world_state', 'present', 'instance', 'instance_id', 'visible_fraction',
            'projection_area', 'isolated_depth_mm', 'static_depth_mm', 'camera_to_world', 'time_s')}
    tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    future_state, future_present = tensor(arrays['world_state'][6:]).float(), tensor(arrays['present'][6:]).bool()
    cameras, times = tensor(arrays['camera_to_world'][6:]).float(), tensor(arrays['time_s'][6:]).float()
    event_labels = dict(future_instance=tensor(arrays['instance'][6:]).long(), instance_id=tensor(arrays['instance_id']).long(),
        visible_fraction=tensor(arrays['visible_fraction'][6:]), projection_area=tensor(arrays['projection_area'][6:]),
        history_visible_fraction=tensor(arrays['visible_fraction'][:6]), history_projection_area=tensor(arrays['projection_area'][:6]), present=future_present)
    events = reappearance_events(event_labels)
    z, p = depth8(arrays['isolated_depth_mm'][6:]); sz, sp = depth8(arrays['static_depth_mm'][6:])
    visible = area8(arrays['instance'][6:, :, None] == arrays['instance_id'][None, None, :, None, None])
    geometry_targets = dict(isolated_depth_mm=tensor(z), full_mask=tensor(p), visible_mask=tensor(visible),
        static_depth_mm=tensor(sz), static_support=tensor(sp))
    full_labels = dict(labels, history_eligible=eligible, state=future_state.to(device), present=future_present.to(device))
    _, _, matched, previous, reference, presence = matched_state_supervision(history['initial'][0], ids, full_labels)
    _, projection = system.project_history(x, obs, geometry_intrinsics(x['K']), cameras.to(device).flatten(0, 1))
    slot, _ = focal_history_slot(history, ids, focal)
    # Omit RGB, depth, masks, and full observations from the immutable history
    # cache. Static projection and learned history were built without labels.
    history = dict({k: history[k] for k in ('tokens', 'shared', 'separate', 'initial', 'covariance', 'presence_logits', 'empty')},
        result=dict(tracks=[{k: t[k] for k in ('birth', 'score', 'presence_logit')} for t in result['tracks']]))
    return to_device(dict(metadata=record, empty=False, history=history, reference_ids=ids, history_eligible=eligible,
        selected_slots=[s for s, _ in selected], selected_references=[k for _, k in selected],
        previous_reference=previous, reference=reference, present=presence, eligible=matched,
        future_state=future_state, future_present=future_present, focal_reference=focal, focal_slot=slot,
        geometry_targets=geometry_targets, events=events, projection=projection,
        K=x['K'], cameras=cameras, times=times, eligible_reference_count=int(eligible.sum())), 'cpu')


def run(args):
    from experiments.paired_interventions.data import to_device
    from experiments.paired_interventions.world import load_upstream
    from experiments.base48.model import free_prefix_presence_objective
    paths = json.loads(Path(args.paths).read_text()); out, cache = Path(args.output), Path(args.cache)
    status = dict(schema=SCHEMA+'-report', state='initializing', completed=False, step=0, target_steps=args.steps, arm=args.arm)
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    def stopped(): return stop[0] or bool(args.watchdog_stop and Path(args.watchdog_stop).exists()) or bool(args.stop_file and Path(args.stop_file).exists())
    def mark(stage, **values):
        status.update(stage=stage, updated_unix=time.time(), **values); atomic_json(out/'status.json', status)
        if args.worker_report: atomic_json(args.worker_report, status)
    try:
        torch.set_num_threads(args.threads); device = torch.device(args.device)
        if device.type == 'cuda': torch.cuda.set_device(device)
        torch.manual_seed(args.seed)
        parent = load_parent(args.parent); parent_sha = sha256(args.parent)
        records = train_records(args.panel, paths['families'])
        data = dict(panel_sha256=sha256(args.panel), index_sha256=sha256(paths['families']), branches=records,
            split='train', test_used=False, selection_uses_validation_scores=False)
        _, history_paths = read_cache(args.history_cache, parent_sha, data['panel_sha256'], data['index_sha256'], records)
        system, _ = load_upstream(paths['upstream'], paths['statistics'], device)
        system.load_state_dict({k[len('upstream.'):]: v for k, v in parent['model'].items() if k.startswith('upstream.')}, strict=True)
        del parent
        system.eval().requires_grad_(False)
        refiner, _, history_meta = load_identity_checkpoint(args.history_overlay, args.parent, data['index_sha256'], upstream=system, device=device)
        if history_meta['arm'] not in ('coarse_full', 'native_full'):
            raise ValueError('Only predetermined completed coarse_full/native_full history arms are permitted')
        root = Path(__file__).resolve().parents[2]
        config = dict(arm=args.arm, steps=args.steps, seed=args.seed, lr=args.lr,
            history_frozen=True, future_truth_conditions=False, deployment_sampler_changed=False,
            component_ablation='isolated_evidence_replication', evidence_forward_retained=True, stochastic_order_unchanged=True,
            ambiguous_nll_and_presence_only=True, geometry_surrogate='conditional_alive_mode_mean_soft_survival',
            weights=dict(evidence=args.evidence_coefficient, motion_projection=.5, common_geometry=.1, reappearance_geometry=.1), batch_worlds=2,
            matched_updates_not_equal_compute=True)
        identity = dict(configuration=config, data=data, sources={p: sha256(root/p) for p in SOURCES},
            statistics_sha256=sha256(paths['statistics']))
        history_identity = dict(path=str(Path(args.history_overlay).resolve()), sha256=sha256(args.history_overlay))
        contract = dict(schema=CACHE_SCHEMA, parent_sha256=parent_sha, history_checkpoint=history_identity,
            data=data, sources={k:v for k,v in identity['sources'].items() if k != 'experiments/research_extensions/train_evidence_repeats.py'}, historical_cache_manifest_sha256=sha256(Path(args.history_cache)/'manifest.json'))
        panel = json.loads(Path(args.panel).read_text()); focal = {f['family_id']: f['target_index'] for f in panel['families']}
        cache.mkdir(parents=True, exist_ok=True)
        with (cache/'.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            manifest = cache/'manifest.json'
            if manifest.exists() and json.loads(manifest.read_text()) != contract: raise ValueError('Future cache identity changed')
            if not manifest.exists(): atomic_json(manifest, contract)
            row_paths = []
            for i, record in enumerate(records):
                if stopped(): mark('cache_paused', state='paused', cache_completed=i, cache_total=len(records)); return 0
                target = cache/('%04d.pt' % i); stamp = target.with_suffix('.json')
                if target.exists() and stamp.exists():
                    if json.loads(stamp.read_text()) != dict(metadata=record, sha256=sha256(target), manifest_sha256=sha256(manifest)):
                        raise ValueError('Future cache row changed')
                else:
                    mark('cache_history_future', cache_completed=i, cache_total=len(records))
                    original = torch.load(history_paths[i], map_location='cpu', weights_only=False)
                    row = prepare_row(system, refiner, original, record, focal[record['family_id']], device)
                    atomic_checkpoint(target, row); atomic_json(stamp, dict(metadata=record, sha256=sha256(target), manifest_sha256=sha256(manifest)))
                row_paths.append(target)
            atomic_json(cache/'complete.json', dict(completed=True, rows=len(records), manifest_sha256=sha256(manifest)))
        if args.cache_only: mark('cache_completed', state='completed', completed=True, cache_only=True); return 0
        groups = [[i for i, r in enumerate(records) if r['family_id'] == family] for family in sorted({r['family_id'] for r in records})]
        if any(len(g) != 2 or [records[i]['world'] for i in g] != [0, 1] for g in groups): raise ValueError('Unpaired TRAIN worlds')
        parameters = []
        for module in (system.dynamics, system.geometry):
            module.train().requires_grad_(True); parameters.extend(module.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
        step = 0
        if args.resume:
            payload = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
            validate_checkpoint(payload, args.parent, data['index_sha256'])
            if payload['identity'] != identity or payload['history_checkpoint'] != history_identity: raise ValueError('Future resume identity differs')
            current = system.state_dict()
            if set(payload['future_overlay']) != set(future_tensors(system)): raise ValueError('Future resume keys differ')
            with torch.no_grad():
                for k, v in payload['future_overlay'].items(): current[k].copy_(v.to(current[k]))
            optimizer.load_state_dict(payload['optimizer']); step = payload['step']; torch.set_rng_state(payload['rng']['cpu'])
            if device.type == 'cuda': torch.cuda.set_rng_state(payload['rng']['cuda'], device)
        def save():
            payload = dict(schema=SCHEMA, parent=dict(path=str(Path(args.parent).resolve()), sha256=parent_sha),
                history_checkpoint=history_identity, identity=identity, future_overlay=future_tensors(system),
                step=step, target_steps=args.steps, completed=step == args.steps, optimizer=optimizer.state_dict(),
                rng=dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device).cpu() if device.type == 'cuda' else None))
            validate_checkpoint(payload, args.parent, data['index_sha256'])
            if step==0:
                assert_parent_initialization(payload,args.parent,out/'initialization.json')
            atomic_checkpoint(out/'last.pt', payload)
            mark('checkpoint_saved', checkpoint_step=step, checkpoint_sha256=sha256(out/'last.pt'))
        if not args.resume: save()
        while step < args.steps and not stopped():
            start = time.monotonic(); indexes = groups[branch_index(step, len(groups), args.seed)]
            rows = [to_device(torch.load(row_paths[i], map_location='cpu', weights_only=False), device) for i in indexes]
            optimizer.zero_grad(set_to_none=True); metrics = {}; total = 0.
            for world, row in enumerate(rows):
                if row['empty']: metrics['empty_world_'+str(world)] = True; continue
                common, parts = ordinary_objective(system.dynamics, row)
                rng = torch.Generator(device=device).manual_seed(args.seed+200003+2*step+world)
                presence, diagnostic = free_prefix_presence_objective(system.dynamics, row['history'], row['reference_ids'],
                    row['future_present'], row['history_eligible'], rng)
                loss = (common+presence)/2
                metrics.update({'w%d_%s' % (world, k): float(v.detach()) for k, v in parts.items()})
                metrics['w%d_free_prefix_presence' % world] = float(presence.detach())
                if row['metadata']['cohort'] != 'ambiguous':
                    free = free_mode_means(system.dynamics, row['history'])
                    # Both arms update the same geometry module from predicted
                    # histories on the ordinary deterministic frame stream.
                    common_geometry, _ = geometry_surrogate(system, free, row, step, reappearance=False)
                    loss = loss+.1*common_geometry/2
                    metrics['w%d_common_geometry' % world] = float(common_geometry.detach())
                    if args.arm == 'paired_geometry':
                        motion, more = free_motion_projection_loss(free, row, system.dynamics.scale)
                        geometry, geom = geometry_surrogate(system, free, row, step)
                        loss = loss+(.5*motion+.1*geometry)/2
                        metrics.update({'w%d_%s' % (world, k): float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in {**more, **geom}.items()})
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite future loss')
                loss.backward(); total += float(loss.detach())
                del loss
            if args.arm == 'paired_geometry' and records[indexes[0]]['cohort'] == 'evidence':
                # Future trajectories are comparison candidates only, after
                # both historical posteriors and free paths have been built.
                if not any(row['empty'] for row in rows):
                    refs = torch.stack([r['future_state'][:, r['focal_reference'], :6] for r in rows])
                    evidence, more = evidence_pair_loss(system.dynamics, rows, refs, step, args.seed)
                    # Identical forward and RNG stream in both coefficient arms.
                    (args.evidence_coefficient*evidence).backward(); total += args.evidence_coefficient*float(evidence.detach())
                    metrics.update(more, evidence_loss=float(evidence.detach()), evidence_weight=args.evidence_coefficient)
            active = [p for p in parameters if p.grad is not None]
            if any(not torch.isfinite(p.grad).all() for p in active): raise FloatingPointError('Nonfinite future gradients')
            norm = torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True) if active else 0.
            if active: optimizer.step()
            step += 1
            if device.type == 'cuda': torch.cuda.synchronize(device)
            metrics.update(step=step, loss=total, gradient_norm=float(norm), step_seconds=time.monotonic()-start,
                family_id=records[indexes[0]]['family_id'], optimizer_update=bool(active), arm=args.arm)
            with (out/'updates.jsonl').open('a') as stream: stream.write(json.dumps(metrics, allow_nan=False)+'\n')
            mark('fit', state='running', step=step, last_update=metrics, configuration=config)
            if step % args.save_every == 0 or step == args.steps or stopped(): save()
        if status.get('checkpoint_step') != step: save()
        mark('completed' if step == args.steps else 'paused', state='completed' if step == args.steps else 'paused', completed=step == args.steps, step=step)
        return 0
    except Exception as exc:
        mark('failed', state='failed', error=str(exc), traceback=traceback.format_exc(), completed=False); return 1



def validate_fixed_inputs(args):
    from .input_lock import verify
    paths=json.loads(Path(args.paths).read_text())
    inputs={key:getattr(args,key) for key in ('parent','history_overlay','panel','paths')}
    inputs['families']=paths['families']
    verify(args.input_lock,inputs)
    records=train_records(args.panel,paths['families'])
    if len(records)!=128 or len({r['family_id']for r in records})!=64:
        raise ValueError('64 original TRAIN families, two worlds required')
    for family in {r['family_id']for r in records}:
        if {(r['world'],r['camera'],r['split'])for r in records if r['family_id']==family}!={(0,0,'train'),(1,0,'train')}:
            raise ValueError('Changed paired TRAIN exposure')


def assert_parent_initialization(payload,parent_path,receipt_path):
    import hashlib
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    expected={k[len('upstream.'):]:v for k,v in parent['model'].items()
              if k.startswith(('upstream.dynamics.','upstream.geometry.'))}
    actual=payload['future_overlay']
    if set(actual)!=set(expected)or any(v.dtype!=expected[k].dtype or v.shape!=expected[k].shape
                                      or not torch.equal(v,expected[k])for k,v in actual.items()):
        raise ValueError('Evidence replication must start from identical original S2 future tensors')
    digest=hashlib.sha256()
    for k,v in sorted(actual.items()):
        digest.update(k.encode());digest.update(v.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    atomic_json(receipt_path,dict(schema=SCHEMA+'-initialization',step=0,seed=payload['identity']['configuration']['seed'],
        same_original_parent=True,future_tensor_sha256=digest.hexdigest(),parent_sha256=sha256(parent_path),
        source_sha256=sha256(__file__),optimizer_inherited=False))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('paths', 'parent', 'history-overlay', 'history-cache', 'panel', 'cache', 'output', 'budget-ledger', 'job-id'): p.add_argument('--'+name, required=True)
    p.add_argument('--input-lock',required=True)
    p.add_argument('--arm', choices=('paired_geometry',), default='paired_geometry')
    p.add_argument('--steps', type=int, default=2048); p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--evidence-coefficient', type=float, choices=(0., .1), required=True)
    p.add_argument('--seed', type=int, choices=(17017,17029,17043), required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=4); p.add_argument('--save-every', type=int, default=16)
    p.add_argument('--max-seconds', type=float, default=3600.); p.add_argument('--stop-file')
    p.add_argument('--resume', action='store_true'); p.add_argument('--cache-only', action='store_true')
    p.add_argument('--worker-report', help=argparse.SUPPRESS); p.add_argument('--watchdog-stop', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not (not args.cache_only and args.steps == 2048 and args.seed in (17017,17029,17043) and args.lr == 1e-5 and math.isfinite(args.lr)
            and 0 < args.max_seconds <= 7200 and 1 <= args.threads <= 16 and args.save_every > 0):
        p.error('Require bounded finite steps/lr/wall/threads')
    validate_fixed_inputs(args)
    if args.worker_report: return run(args)
    from .typed_state_test import budget_guard
    ledger=json.loads(Path(args.budget_ledger).read_text())
    job=budget_guard(args.budget_ledger,args.job_id,dict(ledger_path=args.budget_ledger,gpu_hour_cap=ledger['gpu_hour_cap'],deadline_unix=ledger['deadline_unix']))
    if args.max_seconds>job['maximum_wall_seconds']-90:raise ValueError('Reserve90s outer cleanup')
    out = Path(args.output).resolve()
    protected = [Path(x).resolve() for x in (args.cache, args.history_cache, Path(args.parent).parent,
                                            Path(args.history_overlay).parent)]
    if any(out == x or x in out.parents or out in x.parents for x in protected): raise ValueError('Output overlaps preserved assets/cache')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            if not (out/'last.pt').exists(): raise FileNotFoundError('Resume checkpoint missing')
        elif any(x.name != '.lock' for x in out.iterdir()): raise FileExistsError('Fresh future output required')
        attempt = out/'attempts'/('%d_%d' % (time.time_ns(), os.getpid())); attempt.mkdir(parents=True)
        command = [sys.executable, '-m', 'experiments.research_extensions.train_evidence_repeats',
            *(sys.argv[1:] if argv is None else argv), '--worker-report', str(attempt/'report.json'), '--watchdog-stop', str(attempt/'STOP')]
        code=supervise(command, out, attempt, args.max_seconds, 60., args.stop_file)
        payload=torch.load(out/'last.pt',map_location='cpu',weights_only=False)if(out/'last.pt').exists()else None
        complete=bool(code==0 and payload and payload.get('completed')is True and payload.get('step')==2048)
        if payload:validate_checkpoint(payload,args.parent,require_complete=complete)
        atomic_json(out/'ablation_completion.json',dict(schema=SCHEMA+'-completion',completed=complete,
            component_ablation='isolated_evidence_replication',seed=args.seed,evidence_coefficient=args.evidence_coefficient,step=payload.get('step')if payload else None,
            checkpoint_sha256=sha256(out/'last.pt')if payload else None,
            initialization_sha256=sha256(out/'initialization.json')if(out/'initialization.json').exists()else None,
            native_sources_unchanged=True,source_sha256=sha256(__file__),returncode=code))
        return 0 if complete else 2


if __name__ == '__main__': raise SystemExit(main())
