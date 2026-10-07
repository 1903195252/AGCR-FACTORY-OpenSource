"""Independent TRAIN-only dynamics repair on the unchanged historical posterior.

No video transformer is instantiated. Cache preparation runs frozen DINO/VAE
and the actual historical observation/filter; fitting updates JointDynamics
only. Original likelihood and (evidence-cohort) mean-rollout losses remain.
The added energy score follows the posterior/noise/absorbing-presence sampler,
with analytic scene-mode weights and score-function gradients for Bernoulli
events. Labels enter costs only. This does not repair historical localization.
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

import torch
from torch.nn import functional as F

from .region_probe import atomic_json, sha256
from .train_candidate import atomic_checkpoint, branch_index, supervise

SCHEMA = 'agcr-peerdream-direction-state-repair-1'
PREFIX = 'upstream.dynamics.'


def parent_training(parent):
    if parent.get('schema') == 'agcr-base-presence-refit-1-checkpoint':
        return parent['parent_training']
    return {k: parent[k] for k in ('identity', 'step', 'completed', 'validation_best') if k in parent}


def validate_checkpoint(payload, parent_payload, parent_sha):
    """Typed overlay contract; renderer, frontend and statistics cannot change."""
    if payload.get('schema') != SCHEMA or payload.get('parent', {}).get('sha256') != parent_sha:
        raise ValueError('Wrong state-repair schema or parent SHA')
    if payload.get('parent_training') != parent_training(parent_payload):
        raise ValueError('State-repair Base ancestor identity changed')
    if payload['parent'].get('schema') != parent_payload.get('schema') or not payload['parent'].get('path'):
        raise ValueError('State-repair parent schema/path missing')
    data = payload.get('identity', {}).get('data', {})
    config = payload.get('identity', {}).get('configuration', {})
    if (data.get('split') != 'train' or data.get('test_used') is not False
            or data.get('selection_uses_validation_scores') is not False or not data.get('branches')
            or any(r.get('split') != 'train' or r.get('camera') != 0 or r.get('world') not in (0, 1) for r in data['branches'])
            or config.get('historical_frontend_frozen') is not True
            or config.get('original_dynamics_objective') is not True
            or config.get('ambiguous_nll_only') is not True
            or config.get('horizon') != 16 or config.get('modes') != 8):
        raise ValueError('Invalid TRAIN-only dynamics-repair identity')
    for name in ('index_sha256', 'panel_sha256'):
        digest = data.get(name, '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Missing or invalid '+name)
    old, new = parent_payload['model'], payload.get('model', {})
    if set(old) != set(new):
        raise ValueError('State-repair tensor keys changed')
    allowed = payload.get('trainable_keys', [])
    if not allowed or len(set(allowed)) != len(allowed) or any(not k.startswith(PREFIX) or k not in old for k in allowed):
        raise ValueError('Only existing dynamics parameters may be fitted')
    changed = []
    for key, value in new.items():
        if not isinstance(value, torch.Tensor) or value.shape != old[key].shape or value.dtype != old[key].dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid state-repair tensor: '+key)
        if not torch.equal(value, old[key]):
            if key not in allowed or key.startswith(PREFIX+'scale.'):
                raise ValueError('Frozen frontend/statistics/non-dynamics tensor changed: '+key)
            changed.append(key)
    step, target = payload.get('step'), payload.get('target_steps')
    if type(step) is not int or type(target) is not int or not 0 <= step <= target or payload.get('completed') is not (step == target):
        raise ValueError('Invalid state-repair update/completion contract')
    if config.get('steps') != target:
        raise ValueError('Target updates disagree with configuration')
    return dict(schema=SCHEMA, step=step, target_steps=target, completed=payload['completed'],
                changed_keys=sorted(changed), trainable_keys=allowed, identity=payload['identity'],
                historical_frontend_unchanged=True, diagnostic_only=True)


def train_records(panel_path, index_path, verify=True):
    panel, index = [json.loads(Path(p).read_text()) for p in (panel_path, index_path)]
    if panel.get('schema') != 'peerdream-direction-panel-1' or panel.get('test_used') is not False:
        raise ValueError('Require locked TRAIN/VAL panel without TEST')
    if index.get('schema') != 'agcr-intervention-index-1' or not index.get('audit_passed') or panel.get('source_index_sha256') != sha256(index_path):
        raise ValueError('Audited source index SHA differs')
    root = Path(index['root']).resolve()
    if root != Path(panel['source_root']).resolve():
        raise ValueError('Panel data root changed')
    byid = {r['family_id']: r for r in index['families']}
    selected = panel['families']
    if len(byid) != len(index['families']) or len({r['family_id'] for r in selected}) != len(selected):
        raise ValueError('Duplicate family')
    records = []
    for family in selected:
        if family != byid.get(family['family_id']) or family['split'] not in ('train', 'val'):
            raise ValueError('Unregistered panel family or TEST family')
        if family['split'] != 'train':
            continue  # Do not read VAL pixels/labels, even during cache preparation.
        cells = [r for r in family['records'] if r['camera'] == 0]
        if len(cells) != 2 or {r['world'] for r in cells} != {0, 1}:
            raise ValueError('Require both TRAIN worlds, camera0')
        for row in sorted(cells, key=lambda r: r['world']):
            path = (root/row['path']).resolve()
            if root not in path.parents or (verify and sha256(path) != row['sha256']):
                raise ValueError('Unverified or escaped native branch')
            records.append(dict(family_id=family['family_id'], split='train', cohort=family['cohort'],
                                world=row['world'], camera=0, path=str(path), sha256=row['sha256']))
    if not records:
        raise ValueError('Empty TRAIN panel')
    return records


def sample_bank(model, history, generator, horizon=16):
    """One independent draw per mode; exact absorbing-presence native law.

    Same initial posterior and step function as native sample_shared_future.
    Scene mode is analytically enumerated, not argmax selected. No supervision
    is accepted, including no reference presence mask controlling integration.
    """
    from experiments.paired_interventions.rollout import _initial, _step
    initial, tokens, scene = (history[k] for k in ('initial', 'tokens', 'shared'))
    logits = model(tokens, scene, initial[:, None])['mode_logits'][0]
    count = len(logits)
    expand = lambda x: x.expand(count, *x.shape[1:])
    modes = torch.arange(count, device=initial.device)
    current = _initial(expand(initial), expand(history['covariance']), generator)
    initial_logits = expand(history['presence_logits'])
    alive = torch.rand(initial_logits.shape, device=initial.device, generator=generator) < initial_logits.sigmoid()
    logp = -F.binary_cross_entropy_with_logits(initial_logits, alive.to(initial), reduction='none').sum(-1)
    states, present = [current], []
    for _ in range(horizon):
        prediction = model(expand(tokens), expand(scene), torch.stack(states, 1), modes)
        hazard = prediction['presence_logits'][:, 0, -1]
        event = torch.rand(hazard.shape, device=initial.device, generator=generator) < hazard.sigmoid()
        logp = logp-(F.binary_cross_entropy_with_logits(hazard, event.to(initial), reduction='none')*alive).sum(-1)
        alive = alive & event
        new = _step(model, states[-1], prediction, generator, .2)
        states.append(torch.where(alive[..., None], new, states[-1]))
        present.append(alive)
    return dict(state=torch.stack(states[1:], 1), present=torch.stack(present, 1),
                log_probability=logp, mode_weights=logits.softmax(-1))


def trajectory_vector(states, present, mean, std):
    """Injective representation of present p/u/heading plus explicit existence.

    Position/velocity use the unchanged TRAIN statistics. Absent motion is
    omitted while an existence coordinate distinguishes absence from the mean.
    The score is on these motion variables, not on unscored dimensions/IDs.
    """
    state = (states[..., :6]-mean[:6])/std[:6]
    angular = torch.stack((states[..., 6].sin(), states[..., 6].cos()), -1)
    a = present.to(states)[..., None]
    vector = torch.cat((a, a*state, a*angular), -1)
    return vector.flatten(-3)/math.sqrt(vector.shape[-3]*vector.shape[-2]*vector.shape[-1])


def energy_score(a, b, target, weights, logpa=None, logpb=None):
    """Unbiased two-bank stratified energy-score estimator and gradient.

    Independent A/B within each mode keep even diagonal pair distances valid.
    Bernoulli score terms have zero forward value; mixture and continuous
    gradients remain pathwise. No sample-dependent baseline biases REINFORCE.
    """
    if a.shape != b.shape or a.ndim != 2 or target.shape != a.shape[1:] or weights.shape != a.shape[:1]:
        raise ValueError('Energy score shape mismatch')
    da, db = (a-target).norm(dim=-1), (b-target).norm(dim=-1)
    cross = (a[:, None]-b[None]).norm(dim=-1)
    first = .5*(weights*(da+db)).sum()
    repulsion = .5*(cross*weights[:, None]*weights[None]).sum()
    value = first-repulsion
    if logpa is not None:
        if logpb is None or logpa.shape != weights.shape or logpb.shape != weights.shape:
            raise ValueError('Both Bernoulli log-probability banks are required')
        reward_a = .5*da-.5*(cross*weights[None]).sum(-1)
        reward_b = .5*db-.5*(cross*weights[:, None]).sum(0)
        value = value+(weights.detach()*reward_a.detach()*(logpa-logpa.detach())).sum()
        value = value+(weights.detach()*reward_b.detach()*(logpb-logpb.detach())).sum()
    return value, dict(energy_attraction=first.detach(), energy_repulsion=repulsion.detach())


def distribution_objective(model, row, generator):
    slots = row['selected_slots']
    if not len(slots):
        return sum(p.sum()*0 for p in model.parameters()), dict(energy_targets=0)
    a = sample_bank(model, row['history'], generator)
    b = sample_bank(model, row['history'], generator)
    targets = row['selected_references']
    vectors = [trajectory_vector(bank['state'][:, :, slots], bank['present'][:, :, slots],
                                 model.scale.mean, model.scale.std) for bank in (a, b)]
    target = trajectory_vector(row['future_state'][:, targets], row['future_present'][:, targets],
                               model.scale.mean, model.scale.std)
    loss, parts = energy_score(*vectors, target, a['mode_weights'], a['log_probability'], b['log_probability'])
    return loss, dict(parts, energy_targets=len(slots))


def ordinary_objective(model, row):
    loss, parts, _ = model.objective(row['history']['tokens'][0], row['history']['shared'],
        row['history']['initial'][0], row['previous_reference'], row['reference'], row['eligible'], row['present'])
    if row['metadata']['cohort'] == 'ambiguous':
        loss = loss-.1*parts['rollout_l1']
    return loss, parts


def load_parent(path):
    parent = torch.load(path, map_location='cpu', weights_only=False)
    if parent.get('schema') == 'agcr-base-presence-refit-1-checkpoint':
        from experiments.base48.presence_refit import validate_derived
        original_path = parent['refit']['parent_checkpoint_path']
        original = torch.load(original_path, map_location='cpu', weights_only=False)
        validate_derived(parent, original, sha256(original_path))
    elif parent.get('identity', {}).get('schema') != 'agcr-base48-training-1' or parent['identity'].get('phase') != 'state' or not parent.get('completed'):
        raise ValueError('Completed Base state parent or admitted presence derivative required')
    return parent


@torch.no_grad()
def cache_row(system, encoders, record, device):
    from experiments.paired_interventions.data import load_branch, to_device, historical_labels
    from experiments.paired_interventions.bridge import historical_posterior
    from experiments.paired_interventions.world import query_labels
    from experiments.base48.model import focal_history_slot
    from experiments.paper_experiments.agcr_exp.stage2_system import matched_state_supervision
    branch = load_branch(record['path'])
    x = to_device(branch['inputs'], device)
    y = to_device(branch['supervision'], device)
    history = historical_posterior(system, x, encoders.history(x['rgb']))
    labels = historical_labels(dict(supervision=y))
    if history['empty']:
        return dict(metadata=record, empty=True, eligible_reference_count=int(labels['history_detectable'].any(0).any(0).sum()))
    ids = query_labels(history, labels)  # Association is historical, after prediction.
    eligible = labels['history_detectable'].any(0).any(0)
    selected = [(focal_history_slot(history, ids, k)[0], k) for k in range(len(eligible)) if bool(eligible[k])]
    selected = [(slot, k) for slot, k in selected if slot >= 0]
    full_labels = dict(labels, history_eligible=eligible, state=y['state'], present=y['present'])
    _, _, matched, previous, reference, present = matched_state_supervision(history['initial'][0], ids, full_labels)
    return to_device(dict(metadata=record, empty=False,
        history=dict({k: history[k] for k in ('tokens', 'shared', 'initial', 'covariance', 'presence_logits')},
            empty=False, result=dict(tracks=[{k: t[k] for k in ('birth', 'score', 'presence_logit')}
                                            for t in history['result']['tracks']])),
        reference_ids=ids, history_eligible=eligible,
        selected_slots=[s for s, _ in selected], selected_references=[k for _, k in selected],
        previous_reference=previous, reference=reference, present=present, eligible=matched,
        future_state=y['state'], future_present=y['present'],
        eligible_reference_count=int(eligible.sum())), 'cpu')


def run(args):
    from experiments.paired_interventions.data import to_device
    from experiments.paired_interventions.world import load_upstream
    from experiments.paper_experiments.agcr_exp.stage2_system import FrozenImageEncoders
    from experiments.paper_experiments.agcr_exp.future_prediction import JointDynamics
    paths = json.loads(Path(args.paths).read_text())
    out, cache = Path(args.output), Path(args.cache)
    status = dict(schema=SCHEMA+'-report', state='initializing', completed=False, step=0,
                  target_steps=args.steps, diagnostic_only=True, historical_frontend_unchanged=True)
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    def stopped():
        return stop[0] or Path(args.watchdog_stop).exists() or bool(args.stop_file and Path(args.stop_file).exists())
    def mark(stage, **values):
        status.update(stage=stage, updated_unix=time.time(), **values)
        atomic_json(args.worker_report, status)
        atomic_json(out/'status.json', status)
    try:
        torch.set_num_threads(args.threads)
        device = torch.device(args.device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        torch.manual_seed(args.seed)
        parent = load_parent(args.parent)
        records = train_records(args.panel, paths['families'])
        workspace = Path(__file__).resolve().parents[2]
        sources = [Path(__file__)]
        for directory in ('base48', 'paired_interventions', 'paper_experiments/agcr_exp'):
            sources.extend(p for p in (workspace/'experiments'/directory).glob('*.py') if not p.name.startswith('test_'))
        sources.extend(Path(paths['dino_code']).rglob('*.py'))
        identity = dict(configuration=dict(steps=args.steps, seed=args.seed, lr=args.lr, energy_weight=args.energy_weight,
                            optimizer='AdamW', weight_decay=.01, gradient_clip=1., batch_worlds=1,
                            modes=8, independent_banks=2, horizon=16, ambiguous_nll_only=True,
                            energy_space='standardized_p_u_heading_plus_absorbing_presence',
                            historical_frontend_frozen=True, original_dynamics_objective=True,
                            free_prefix_presence_weight=1.,
                            matched_updates_not_equal_compute=True),
            data=dict(panel_path=str(Path(args.panel).resolve()), panel_sha256=sha256(args.panel),
                      index_sha256=sha256(paths['families']), branches=records, split='train',
                      selection_uses_validation_scores=False, test_used=False),
            sources={str(p.resolve()): sha256(p) for p in sources},
            statistics=dict(path=paths['statistics'], sha256=sha256(paths['statistics'])))
        parent_info = dict(path=str(Path(args.parent).resolve()), sha256=sha256(args.parent), schema=parent.get('schema'))
        vae_assets = sorted(p for p in (Path(paths['wan'])/'vae').rglob('*')
                            if p.is_file() and p.suffix in ('.json', '.bin', '.safetensors'))
        if not vae_assets:
            raise ValueError('Missing frozen historical VAE assets')
        cache_contract = dict(schema=SCHEMA+'-cache', parent=parent_info, data=identity['data'],
                              statistics=identity['statistics'], sources=identity['sources'],
                              encoders={k: dict(path=paths[k], sha256=sha256(paths[k])) for k in ('dino_weights',)},
                              vae_assets={str(p): sha256(p) for p in vae_assets})
        cache.mkdir(parents=True, exist_ok=True)
        manifest = cache/'manifest.json'
        # Lock shared cache creation: two arms may read only committed rows.
        with (cache/'.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if manifest.exists() and json.loads(manifest.read_text()) != cache_contract:
                raise ValueError('Cache identity changed')
            if not manifest.exists():
                atomic_json(manifest, cache_contract)
            rows, system, encoders = [], None, None
            for i, record in enumerate(records):
                if stopped():
                    mark('cache_paused', state='paused', cache_completed=len(rows), cache_total=len(records)); return 0
                path = cache/('%04d.pt' % i)
                stamp = path.with_suffix('.json')
                if path.exists() and stamp.exists():
                    marker = json.loads(stamp.read_text())
                    if marker != dict(metadata=record, sha256=sha256(path), cache_manifest_sha256=sha256(manifest)):
                        raise ValueError('Cache row SHA/metadata changed')
                else:
                    mark('cache_history', cache_completed=i, cache_total=len(records))
                    if system is None:
                        system, _ = load_upstream(paths['upstream'], paths['statistics'], device)
                        system.load_state_dict({k[len('upstream.'):]: v for k, v in parent['model'].items() if k.startswith('upstream.')}, strict=True)
                        system.eval().requires_grad_(False)
                        encoders = FrozenImageEncoders(paths['dino_code'], paths['dino_weights'], paths['wan']+'/vae', device)
                    value = cache_row(system, encoders, record, device)
                    atomic_checkpoint(path, value)
                    atomic_json(stamp, dict(metadata=record, sha256=sha256(path), cache_manifest_sha256=sha256(manifest)))
                value = torch.load(path, map_location='cpu', weights_only=False)
                if value['metadata'] != record:
                    raise ValueError('Cached row metadata mismatch')
                rows.append(value)
            del system, encoders
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        coverage = dict(worlds=len(rows), empty_histories=sum(r['empty'] for r in rows),
                        eligible_references=sum(r['eligible_reference_count'] for r in rows),
                        scored_unique_references=sum(len(r.get('selected_slots', [])) for r in rows),
                        targets_by_cohort={c: sum(len(r.get('selected_slots', [])) for r in rows if r['metadata']['cohort'] == c) for c in ('evidence', 'ambiguous')})
        if not coverage['scored_unique_references']:
            raise ValueError('No historical targets available; no GT tracks may be inserted')
        atomic_json(cache/'complete.json', dict(completed=True, rows=len(rows), coverage=coverage, manifest_sha256=sha256(manifest)))
        if args.cache_only:
            mark('cache_completed', state='completed', completed=True, cache_only=True, cache_coverage=coverage)
            return 0
        model = JointDynamics(json.loads(Path(paths['statistics']).read_text())).to(device)
        model.load_state_dict({k[len(PREFIX):]: v for k, v in parent['model'].items() if k.startswith(PREFIX)}, strict=True)
        model.checkpoint_tracks_above = 8
        model.train()
        # Cache creation/encoder construction must not change either arm's
        # training RNG. Current blocks have zero dropout; still pin this boundary.
        torch.manual_seed(args.seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        step = 0
        if args.resume:
            previous = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
            validate_checkpoint(previous, parent, parent_info['sha256'])
            if previous['identity'] != identity:
                raise ValueError('Resume identity changed')
            model.load_state_dict({k[len(PREFIX):]: v for k, v in previous['model'].items() if k.startswith(PREFIX)}, strict=True)
            optimizer.load_state_dict(previous['optimizer'])
            generator.set_state(previous['rng']['private'])
            torch.set_rng_state(previous['rng']['cpu'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state(previous['rng']['cuda'], device)
            step = previous['step']
        def save():
            weights = dict(parent['model'])
            weights.update({PREFIX+k: v.detach().cpu() for k, v in model.state_dict().items()})
            payload = dict(schema=SCHEMA, parent=parent_info, parent_training=parent_training(parent),
                identity=identity, trainable_keys=[PREFIX+k for k, _ in model.named_parameters()], model=weights,
                step=step, target_steps=args.steps, completed=step == args.steps, optimizer=optimizer.state_dict(),
                cache_coverage=coverage, rng=dict(private=generator.get_state().cpu(), cpu=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state(device).cpu() if device.type == 'cuda' else None))
            validate_checkpoint(payload, parent, parent_info['sha256'])
            atomic_checkpoint(out/'last.pt', payload)
            mark('checkpoint_saved', checkpoint_step=step, checkpoint_sha256=sha256(out/'last.pt'))
        if not args.resume:
            save()
        mark('fit', state='running', configuration=identity['configuration'], cache_coverage=coverage)
        while step < args.steps and not stopped():
            begin = time.monotonic()
            row = to_device(rows[branch_index(step, len(rows), args.seed)], device)
            optimizer.zero_grad(set_to_none=True)
            if row['empty']:
                # Preserve the exposure count; do not replace hard samples with easier ones.
                metrics = dict(empty_history=True, energy_targets=0)
            else:
                common, parts = ordinary_objective(model, row)
                if not torch.isfinite(common):
                    raise FloatingPointError('Non-finite original dynamics objective')
                common.backward()
                metrics = {k: float(v.detach()) for k, v in parts.items()}
                metrics['ordinary_loss'] = float(common.detach())
                # Preserve the deployed Base presence-head training on free
                # posterior prefixes. A step-keyed private stream is identical
                # across arms regardless of energy-score random consumption.
                from experiments.base48.model import free_prefix_presence_objective
                presence_rng = torch.Generator(device=device).manual_seed(args.seed+200003+step)
                presence_loss, presence_parts = free_prefix_presence_objective(model, row['history'],
                    row['reference_ids'], row['future_present'], row['history_eligible'], presence_rng)
                if presence_loss.requires_grad:
                    presence_loss.backward()
                metrics.update(presence_parts)
                metrics['free_prefix_presence_loss'] = float(presence_loss.detach())
                if args.energy_weight:
                    es, more = distribution_objective(model, row, generator)
                    if not torch.isfinite(es):
                        raise FloatingPointError('Non-finite energy score')
                    (args.energy_weight*es).backward()
                    metrics.update({k: float(v) if isinstance(v, torch.Tensor) else v for k, v in more.items()})
                    metrics['energy_score'] = float(es.detach())
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Non-finite dynamics gradients')
                metrics['gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
                optimizer.step()
            step += 1
            elapsed = time.monotonic()-begin
            metrics.update(step=step, family_id=row['metadata']['family_id'], world=row['metadata']['world'],
                           seconds=elapsed, step_seconds=elapsed, energy_weight=args.energy_weight,
                           loss=metrics.get('ordinary_loss', 0.)+metrics.get('free_prefix_presence_loss', 0.)+args.energy_weight*metrics.get('energy_score', 0.),
                           optimizer_update=not row['empty'])
            with (out/'updates.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics, allow_nan=False)+'\n')
            mark('fit', step=step, last_update=metrics)
            if step % args.save_every == 0 or step == args.steps or stopped():
                save()
        if status.get('checkpoint_step') != step:
            save()
        mark('completed' if step == args.steps else 'paused', state='completed' if step == args.steps else 'paused', completed=step == args.steps, step=step)
        return 0
    except Exception as error:
        mark('failed', state='failed', error=str(error), traceback=traceback.format_exc(), completed=False)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('paths', 'parent', 'panel', 'cache', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--energy-weight', type=float, default=.1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=17029)
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
    if not (1 <= args.steps <= 4096 and 0 <= args.energy_weight <= 1 and 0 < args.lr <= .001 and 0 < args.max_seconds <= 3600 and 1 <= args.threads <= 16 and args.save_every > 0):
        parser.error('Require finite bounded updates/lr/energy/threads/wall cap <=3600 seconds')
    if args.worker_report:
        return run(args)
    output = Path(args.output).resolve()
    protected = [Path(args.parent).resolve().parent, Path(args.cache).resolve(),
                 Path(json.loads(Path(args.panel).read_text())['source_root']).resolve()]
    if any(output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError('Output overlaps a parent checkpoint, data or cache directory')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            if not (output/'last.pt').exists():
                raise FileNotFoundError('Resume checkpoint missing')
        elif any(p.name != '.lock' for p in output.iterdir()):
            raise FileExistsError('Use a new independent output directory')
        attempt = output/'attempts'/('%d_%d' % (time.time_ns(), os.getpid()))
        attempt.mkdir(parents=True)
        command = [sys.executable, '-m', 'experiments.research_extensions.state_repair',
                   *(sys.argv[1:] if argv is None else argv), '--worker-report', str(attempt/'report.json'),
                   '--watchdog-stop', str(attempt/'STOP')]
        return supervise(command, output, attempt, args.max_seconds, 90., args.stop_file)


if __name__ == '__main__':
    raise SystemExit(main())
