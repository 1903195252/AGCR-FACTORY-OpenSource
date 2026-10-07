"""Independent Base48 training; no mutation or resumption of the preserved Pro run.

The registered Base48 protocol supplies each stage's schedule. A Pro checkpoint
can initialize weights only; it cannot confer completed-training status.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
import fcntl
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

from experiments.paired_interventions.data import (
    FamilyDataset, OrdinaryDataset, FamilySampler, to_device, file_sha256,
)
from experiments.paired_interventions.train import (
    atomic_json, export_weights, import_weights, trainable_digest, Objective,
)
from experiments.paired_interventions.validation import panel_indices

SAMPLE_SEEDS = [101, 211, 307, 401, 503, 601, 701, 809]
SCHEMA = 'agcr-base48-training-1'


def protocol_settings(protocol, phase, overrides=None):
    if protocol.get('schema') != 'agcr-base48-protocol-1':
        raise ValueError('An independent registered Base48 protocol is required')
    cfg = dict(protocol.get('stages', {}).get(phase, {}))
    for key in ('steps', 'batch', 'validate_every', 'validation_panel'):
        value = cfg.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError('Unregistered positive stage setting: ' + key)
        if (overrides or {}).get(key) not in (None, value):
            raise ValueError('CLI differs from registered protocol: ' + key)
    if protocol.get('paired_sample_seeds', SAMPLE_SEEDS) != SAMPLE_SEEDS:
        raise ValueError('Base48 uses the fixed eight paired-sample seeds')
    # Parameters, schedule and objective remain explicit in the new identity.
    for key, default in [('lr_new', 1e-4), ('lr_pretrained', 1e-5),
                         ('weight_decay', .01), ('clip_norm', 1.),
                         ('warmup_steps', 100), ('lr_final_ratio', .1)]:
        cfg.setdefault(key, default)
    if any(not math.isfinite(float(cfg[k])) or cfg[k] <= 0
           for k in ('lr_new', 'lr_pretrained', 'clip_norm')):
        raise ValueError('Invalid optimizer setting')
    if cfg['warmup_steps'] < 0 or not 0 <= cfg['lr_final_ratio'] <= 1:
        raise ValueError('Invalid learning-rate schedule')
    return cfg


def registered_variant(protocol, name):
    variants = protocol.get('variants', {})
    if isinstance(variants, list):
        variants = {v['id']: v for v in variants}
    if name not in variants:
        raise ValueError('Unknown Base48 variant: ' + name)
    return dict(variants[name], id=name)


def assert_base_paths(output, ledger, paths):
    output, ledger = Path(output).resolve(), Path(ledger).resolve()
    forbidden = [Path(paths['upstream']).resolve().parent]
    # A distinct run name or budget path cannot bypass protection of Pro assets.
    for value in (output, ledger):
        if 'agcr-intervention-runs' in value.parts:
            raise ValueError('Pro run and its budget are read-only')
        for parent in forbidden:
            if value == parent or parent in value.parents:
                raise ValueError('Cannot write within the upstream checkpoint run')


class BudgetLedger:
    """Exclusive single-job ledger, billed in allocated GPU seconds, also on error."""
    def __init__(self, path, limit, world, clock=time.monotonic):
        if limit <= 0 or world < 1:
            raise ValueError('Positive GPU-hour cap and world size required')
        self.path, self.limit, self.world, self.clock = Path(path), limit, world, clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = open(str(self.path) + '.lock', 'a+')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError('This GPU-hour ledger already has an active owner')
        try:
            old = json.loads(self.path.read_text()) if self.path.exists() else {}
            if self.path.exists() and (old.get('schema') != 'agcr-base48-budget-1' or
                        old.get('limit_gpu_hours') != limit):
                raise ValueError('Budget identity or limit changed')
            self.initial = float(old.get('used_gpu_hours', 0.))
            if not math.isfinite(self.initial) or self.initial < 0:
                raise ValueError('Invalid persisted GPU-hour usage')
            self.started = clock()
            self.closed = False
            self.commit('initializing')
        except BaseException:
            self.lock.close()
            raise

    @property
    def used(self):
        return self.initial + self.world * (self.clock() - self.started) / 3600.

    def exhausted(self):
        return self.used >= self.limit

    def commit(self, state, **extra):
        atomic_json(self.path, dict(schema='agcr-base48-budget-1',
            limit_gpu_hours=self.limit, used_gpu_hours=self.used,
            accounting='allocated_world_size_times_elapsed_including_setup_validation_and_saving',
            pid=os.getpid(), world_size=self.world, state=state,
            updated_unix=time.time(), **extra))

    def close(self, state='stopped', **extra):
        if not self.closed:
            try:
                self.commit(state, **extra)
            finally:
                self.closed = True
                self.lock.close()


def draw_plan(step, batch, paired, family_sampler, ordinary_sampler, world=1):
    """Global plan; separate committed cursors avoid skipping 3/4 of families."""
    plan = []
    pair_count = ordinary_count = 0
    for offset in range(batch):
        if paired and ((step * batch + offset) // world) % 4 == 0:
            draw = family_sampler.cursor + pair_count
            plan.append(('family', family_sampler.index(pair_count),
                         ((draw // 2) % 2, draw % 2)))
            pair_count += 1
        else:
            plan.append(('ordinary', ordinary_sampler.index(ordinary_count), None))
            ordinary_count += 1
    return plan, pair_count, ordinary_count


def atomic_checkpoint(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def source_hashes(root=None):
    """Pin actual Base/runtime dependencies; independent tools do not alter resume."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    required = [
        'experiments/base48/__init__.py', 'experiments/base48/model.py',
        'experiments/base48/train.py',
        'experiments/performance/prefetched_replay.py',
        'experiments/performance/execution_runtime.py',
    ]
    sources = {name: file_sha256(root / name) for name in required}
    # Retain the complete existing scientific/runtime coverage for these
    # shared modules. The Base model imports these, not Base sampling tools.
    for directory in ('paired_interventions', 'paper_experiments/agcr_exp'):
        for p in sorted((root / 'experiments' / directory).glob('*.py')):
            if p.name.startswith(('test_', 'check_', 'benchmark')):
                continue
            sources[str(p.relative_to(root))] = file_sha256(p)
    return sources


def audit_index_ancestry(index_path):
    """Validate a registered test subset without changing any training family.

    This permits weight initialization from a Pro checkpoint trained with the
    source index. It does not relax exact-resume identity checks.
    """
    index = json.loads(Path(index_path).read_text())
    origin = index.get('source_index_path')
    origin_hash = index.get('source_index_sha256')
    if origin is None and origin_hash is None:
        return None
    if not origin or not origin_hash:
        raise ValueError('Both source_index_path and source_index_sha256 are required')
    source_path = Path(origin)
    if not source_path.is_absolute():
        source_path = Path(index_path).resolve().parent / source_path
    if file_sha256(source_path) != origin_hash:
        raise ValueError('Base index source checksum changed')
    source = json.loads(source_path.read_text())
    for candidate in (index, source):
        if candidate.get('schema') != 'agcr-intervention-index-1' or not candidate.get('audit_passed'):
            raise ValueError('Index ancestry requires audited native indices')
        if candidate.get('development_only'):
            raise ValueError('Development index cannot establish formal ancestry')
    if Path(index['root']).resolve() != Path(source['root']).resolve():
        raise ValueError('Index ancestry cannot redirect the underlying branch root')
    def by_id(rows):
        values = {row['family_id']: row for row in rows}
        if len(values) != len(rows):
            raise ValueError('Duplicate family ids in index lineage')
        return values
    current, original = by_id(index['families']), by_id(source['families'])
    for split in ('train', 'val'):
        left = {k:v for k,v in current.items() if v['split'] == split}
        right = {k:v for k,v in original.items() if v['split'] == split}
        if not left or left != right:
            raise ValueError('Base index must preserve every original ' + split + ' family unchanged')
    for family_id, row in current.items():
        if family_id not in original or row != original[family_id]:
            raise ValueError('Base subset changed an original family/record: ' + family_id)
    if not any(row['split'] == 'test' for row in current.values()):
        raise ValueError('Base index must contain a registered test panel')
    return dict(source_index_sha256=origin_hash, source_index_path=str(source_path.resolve()),
                train_val_identical=True, whole_family_test_subset=True,
                train_families=sum(r['split']=='train' for r in current.values()),
                validation_families=sum(r['split']=='val' for r in current.values()),
                test_families=sum(r['split']=='test' for r in current.values()))


def validate_checkpoint_identity(previous, identity, mode):
    old = previous.get('identity', {})
    if mode == 'resume':
        if old != identity:
            raise ValueError('Exact resume identity differs')
        return
    if mode == 'warm_start':
        if identity['phase'] != 'state':
            raise ValueError('A Pro warm-start is allowed only for the Base48 state stage')
        if old.get('development_only', True):
            raise ValueError('Development checkpoint cannot initialize registered training')
        original_index = old.get('family_index_sha256')
        if original_index != identity.get('family_index_sha256'):
            ancestry = identity.get('family_index_ancestry') or {}
            if (not original_index or ancestry.get('source_index_sha256') != original_index
                    or ancestry.get('train_val_identical') is not True
                    or ancestry.get('whole_family_test_subset') is not True):
                raise ValueError('Warm-start family index has no verified unchanged-training ancestry')
        if old.get('ordinary_index_sha256') != identity.get('ordinary_index_sha256'):
            raise ValueError('Warm-start ordinary training index differs')
        return
    if old.get('schema') != SCHEMA or not previous.get('completed'):
        raise ValueError('Stage initialization requires a completed Base48 predecessor')
    expected = {'depth': 'state', 'video': 'depth'}.get(identity['phase'])
    if old.get('phase') != expected:
        raise ValueError('Wrong predecessor phase')
    for key in ('variant', 'seed', 'protocol_sha256', 'family_index_sha256',
                'ordinary_index_sha256', 'parent_sha256', 'source_sha256'):
        if old.get(key) != identity.get(key):
            raise ValueError('Predecessor provenance mismatch: ' + key)


@torch.no_grad()
def validate(model, dataset, indices, device, step, rank=0, world=1):
    """Eight-draw auxiliary objective; never called formal generation quality."""
    modes = {module: module.training for module in model.modules()}
    model.eval()
    rows = []
    try:
        for position in range(rank, len(indices), world):
            family = to_device(dataset[indices[position]], device)
            branch = dict(family['branches'][(position % 2, (position // 2) % 2)],
                          cohort=family['metadata']['cohort'])
            family_seed = int(hashlib.sha256(family['metadata']['family_id'].encode()).hexdigest()[:8], 16)
            values = []
            for seed in SAMPLE_SEEDS:
                generator = torch.Generator(device=device).manual_seed(family_seed + seed)
                loss, _ = model(branch, generator, step, family)
                value = float(loss)
                if not math.isfinite(value):
                    raise FloatingPointError('Nonfinite held-out validation objective')
                values.append(value)
            rows.append(dict(family=family['metadata']['family_id'],
                             value=sum(values) / len(values), draws=values))
    finally:
        for module, training in modes.items():
            module.training = training
    all_rows = [None] * world
    if world > 1:
        torch.distributed.all_gather_object(all_rows, rows)
    else:
        all_rows[0] = rows
    rows = sorted((row for part in all_rows for row in part), key=lambda r: r['family'])
    if not rows:
        raise ValueError('Empty fixed validation panel')
    return dict(step=step, mean=sum(r['value'] for r in rows) / len(rows), rows=rows,
                objective_draw_seeds=SAMPLE_SEEDS, scope='base48_auxiliary_objective_noise_average',
                generated_video_sample_count=0,
                independent_benchmark_metrics=False, formal_experiment_results=False)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('paths', 'protocol', 'output', 'phase', 'gpu-hours', 'budget-ledger'):
        kw = {'required': True}
        if key == 'phase': kw['choices'] = ['state', 'depth', 'video']
        if key == 'gpu-hours': kw['type'] = float
        p.add_argument('--' + key, **kw)
    p.add_argument('--variant', default='full')
    p.add_argument('--seed', type=int, default=17)
    for key in ('steps', 'batch', 'validate-every', 'validation-panel'):
        p.add_argument('--' + key, type=int)
    init = p.add_mutually_exclusive_group()
    for key in ('warm-start', 'init', 'resume'):
        init.add_argument('--' + key)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--prefetch', type=int, default=2)
    p.add_argument('--stop-file')
    p.add_argument('--stop-after', type=int,
                   help='Save and stop at an update boundary without changing registered schedule')
    return p.parse_args(argv)


def main(argv=None):
    a = arguments(argv)
    protocol = json.loads(Path(a.protocol).read_text())
    cfg = protocol_settings(protocol, a.phase, {key: getattr(a, key) for key in
                            ('steps', 'batch', 'validate_every', 'validation_panel')})
    variant = registered_variant(protocol, a.variant)
    paths = json.loads(Path(a.paths).read_text())
    out = Path(a.output).resolve()
    assert_base_paths(out, a.budget_ledger, paths)
    if a.gpu_hours <= 0:
        raise ValueError('Positive bounded GPU-hour budget required')
    if a.prefetch < 0 or a.prefetch > 4 or a.save_every < 1:
        raise ValueError('Invalid bounded prefetch/save setting')
    world, rank, local = (int(os.environ.get(key, default)) for key, default in
                          [('WORLD_SIZE', 1), ('RANK', 0), ('LOCAL_RANK', 0)])
    if cfg['batch'] < world or cfg['batch'] % world:
        raise ValueError('Global batch must divide by world size')
    if a.phase != 'state' and not (a.init or a.resume):
        raise ValueError('Depth/video requires its completed predecessor or an exact resume')
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise FileExistsError('A new Base48 run needs an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    # Fully audit the native family files before allocating the training models.
    families = FamilyDataset(paths['families'], 'train', verify_files=rank == 0)
    heldout = FamilyDataset(paths['families'], 'val', verify_files=rank == 0)
    ordinary = OrdinaryDataset(paths['ordinary'])
    if families.index.get('development_only') or ordinary.index.get('development_only'):
        raise ValueError('Development data cannot enter this registered training run')
    panel = panel_indices(heldout, cfg['validation_panel'])
    if len(panel) != cfg['validation_panel']:
        raise ValueError('Insufficient audited validation families')
    identity = dict(schema=SCHEMA, phase=a.phase, variant=a.variant, seed=a.seed,
        world_size=world, configuration=cfg, variant_options=variant,
        protocol_sha256=file_sha256(a.protocol), source_sha256=source_hashes(),
        family_index_sha256=file_sha256(paths['families']), family_index_ancestry=audit_index_ancestry(paths['families']),
        ordinary_index_sha256=file_sha256(paths['ordinary']),
        parent_sha256=file_sha256(paths['upstream']), statistics_sha256=file_sha256(paths['statistics']),
        empty_text_sha256=file_sha256(paths['empty_text']), paths=paths,
        validation_panel=[heldout.families[i]['family_id'] for i in panel],
        sample_seeds=SAMPLE_SEEDS, development_only=False, formal_experiment_results=False,
        scientific_version='base48_history_state_and_flow_depth_no_decoded_Euler_feedback')
    previous = None
    mode = 'resume' if a.resume else 'warm_start' if a.warm_start else 'init' if a.init else None
    init_path = a.resume or a.warm_start or a.init
    if init_path:
        previous = torch.load(init_path, map_location='cpu', weights_only=False)
        validate_checkpoint_identity(previous, identity, mode)
        if a.resume:
            if (previous.get('budget_ledger') != str(Path(a.budget_ledger).resolve())
                    or not Path(a.budget_ledger).exists()):
                raise ValueError('Exact resume must preserve its existing GPU-hour ledger')
    initialization = previous.get('initialization') if a.resume else (
        dict(mode=mode, path=str(Path(init_path).resolve()), sha256=file_sha256(init_path),
             source_step=previous.get('step'), source_completed=previous.get('completed'),
             optimizer_transferred=False) if previous else dict(mode='registered_upstream'))
    ledger = None
    started = time.monotonic()
    step = 0
    completed = False
    loader = None
    stop = [False]
    def request_stop(*_): stop[0] = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    try:
        torch.cuda.set_device(local)
        device = torch.device('cuda', local)
        if world > 1:
            torch.distributed.init_process_group('nccl', timeout=timedelta(minutes=60))
        if rank == 0:
            ledger = BudgetLedger(a.budget_ledger, a.gpu_hours, world)
            atomic_json(out / 'manifest.json', dict(identity, initialization=initialization))
        from .model import Base48System
        model = Base48System(paths, device, a.phase, variant)
        if previous:
            import_weights(model, previous['model'])
        optimizer = torch.optim.AdamW(model.trainable_groups(), betas=(.9, .999), weight_decay=cfg['weight_decay'])
        fs = FamilySampler(len(families), a.seed + 107)
        ds = FamilySampler(len(ordinary), a.seed + 311)
        generator = torch.Generator(device=device).manual_seed(a.seed + 100003 * rank)
        prior_gpu_hours = 0.
        best = {}
        if a.resume:
            optimizer.load_state_dict(previous['optimizer'])
            step = previous['step']; prior_gpu_hours = previous['gpu_hours']
            fs = FamilySampler(**previous['family_sampler']); ds = FamilySampler(**previous['ordinary_sampler'])
            rng = previous['rank_rng'][rank]
            torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state(rng['cuda'], device)
            generator.set_state(rng['generator']); np.random.set_state(rng['numpy']); random.setstate(rng['python'])
            best = previous.get('validation_best', {})
        objective = Objective(model)
        if world > 1:
            objective = DDP(objective, device_ids=[local], find_unused_parameters=True, broadcast_buffers=False)
        def save(destination='last.pt', is_complete=False):
            rng = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device),
                generator=generator.get_state(), numpy=np.random.get_state(), python=random.getstate())
            states = [None] * world
            if world > 1: torch.distributed.all_gather_object(states, rng)
            else: states[0] = rng
            if rank == 0:
                atomic_checkpoint(out / destination, dict(identity=identity, model=export_weights(model),
                    optimizer=optimizer.state_dict(), step=step, completed=is_complete,
                    validation_best=best, family_sampler=fs.state_dict(), ordinary_sampler=ds.state_dict(),
                    rank_rng=states, gpu_hours=prior_gpu_hours + world * (time.monotonic() - started) / 3600,
                    budget_ledger=str(Path(a.budget_ledger).resolve()), initialization=initialization,
                    trainable_sha256=trainable_digest(model)))
                ledger.commit('training_checkpoint', run=str(out), step=step)
            if world > 1: torch.distributed.barrier()
        loader = ThreadPoolExecutor(max_workers=a.prefetch) if a.prefetch else None
        def load(item):
            kind, index, cell = item
            raw = families[index] if kind == 'family' else ordinary[index]
            if kind == 'ordinary':
                raw['_base48_receiver'] = (index + a.seed) % 2
            if a.phase == 'state':
                from experiments.performance.prefetched_replay import prepare_replay_cpu
                if cell is not None:
                    raw['branches'][cell] = prepare_replay_cpu(raw['branches'][cell], step + 1)
                else:
                    raw = prepare_replay_cpu(raw, step + 1)
            return raw, cell
        while step < cfg['steps']:
            requested = (stop[0] or (a.stop_file and Path(a.stop_file).exists()) or
                         (a.stop_after is not None and step >= a.stop_after) or
                         (rank == 0 and ledger.exhausted()))
            stop_tensor = torch.tensor(int(bool(requested)), device=device)
            if world > 1: torch.distributed.all_reduce(stop_tensor, op=torch.distributed.ReduceOp.MAX)
            if int(stop_tensor): break
            plan, nf, nd = draw_plan(step, cfg['batch'], variant.get('paired_data', True), fs, ds, world)
            owned = plan[rank::world]
            warm = min((step + 1) / max(cfg['warmup_steps'], 1), 1.)
            progress = max(0, step - cfg['warmup_steps']) / max(cfg['steps'] - cfg['warmup_steps'], 1)
            rate = warm * (cfg['lr_final_ratio'] + (1 - cfg['lr_final_ratio']) * .5 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group['lr'] = (cfg['lr_new'] if group['name'] == 'new' else cfg['lr_pretrained']) * rate
            optimizer.zero_grad(set_to_none=True)
            metrics = {}; begin = time.monotonic(); pending = {}
            if loader:
                pending = {i: loader.submit(load, item) for i, item in enumerate(owned[:a.prefetch])}
            for micro, item in enumerate(owned):
                raw, cell = pending.pop(micro).result() if loader else load(item)
                if loader and micro + a.prefetch < len(owned):
                    pending[micro + a.prefetch] = loader.submit(load, owned[micro + a.prefetch])
                family = to_device(raw, device) if cell is not None else None
                branch = (dict(family['branches'][cell], cohort=family['metadata']['cohort'])
                          if family is not None else to_device(raw, device))
                sync = nullcontext() if world == 1 or micro == len(owned) - 1 else objective.no_sync()
                with sync:
                    loss, terms = objective(branch, generator, step + 1, family)
                    finite = torch.isfinite(loss).int()
                    if world > 1: torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
                    if not int(finite): raise FloatingPointError('Nonfinite objective before optimizer commit')
                    (loss / len(owned)).backward()
                for key, value in dict(terms, loss=loss.detach()).items():
                    value = float(value)
                    if not math.isfinite(value): raise FloatingPointError('Nonfinite metric: ' + key)
                    metrics[key] = metrics.get(key, 0.) + value / len(owned)
                del branch, family, raw, loss, terms
            norm = nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']
                                             if p.grad is not None], cfg['clip_norm'], error_if_nonfinite=True)
            optimizer.step(); step += 1; fs.advance(nf); ds.advance(nd)
            gathered = [None] * world
            if world > 1: torch.distributed.all_gather_object(gathered, metrics)
            else: gathered[0] = metrics
            if rank == 0:
                status = dict(schema=SCHEMA, state='running', phase=a.phase, step=step,
                    target_steps=cfg['steps'], world_size=world, update_seconds=time.monotonic() - begin,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(device), gradient_norm=float(norm),
                    gpu_hours=prior_gpu_hours + world * (time.monotonic() - started) / 3600,
                    family_draws=fs.cursor, ordinary_draws=ds.cursor,
                    metrics={k: sum(r.get(k, 0.) for r in gathered) / world for k in set().union(*(r.keys() for r in gathered))},
                    pid=os.getpid(), updated_unix=time.time(), formal_experiment_results=False)
                atomic_json(out / 'status.json', status)
                with (out / 'metrics.jsonl').open('a') as f: f.write(json.dumps(status) + '\n')
                ledger.commit('running', run=str(out), step=step)
                print(json.dumps(status), flush=True)
            if step % a.save_every == 0:
                # Completion is committed only after the final validation succeeds.
                save(is_complete=False)
            if step == cfg['steps']:
                final_digests = [None] * world
                if world > 1:
                    torch.distributed.all_gather_object(final_digests, trainable_digest(model))
                else:
                    final_digests[0] = trainable_digest(model)
                if len(set(final_digests)) != 1:
                    raise AssertionError('DDP divergence before final validation/completion')
            if step % cfg['validate_every'] == 0 or step == cfg['steps']:
                report = validate(model, heldout, panel, device, step, rank, world)
                if rank == 0:
                    atomic_json(out / ('validation_%06d.json' % step), report)
                    atomic_json(out / 'validation_latest.json', report)
                if report['mean'] < best.get('value', float('inf')):
                    best = dict(value=report['mean'], step=step, scope=report['scope'])
                    save('best.pt', step == cfg['steps'])
                save(is_complete=step == cfg['steps'])
        completed = step == cfg['steps']
        save(is_complete=completed)
        if rank == 0 and completed and (out / 'best.pt').exists():
            selected = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
            selected.update(completed=True, training_completed_step=step)
            atomic_checkpoint(out / 'best.pt', selected)
        digests = [None] * world
        if world > 1: torch.distributed.all_gather_object(digests, trainable_digest(model))
        else: digests[0] = trainable_digest(model)
        if len(set(digests)) != 1: raise AssertionError('Learned parameters differ between DDP ranks')
        if rank == 0:
            result = dict(completed=completed, step=step, target_steps=cfg['steps'],
                reason='complete' if completed else 'bounded_or_requested_stop',
                trainable_sha256=digests[0], ddp_parameters_identical=True,
                formal_experiment_results=False, updated_unix=time.time())
            atomic_json(out / 'completion.json', result)
            atomic_json(out / 'status.json', dict(result, phase=a.phase, state='completed' if completed else 'paused'))
    except BaseException as error:
        if rank == 0:
            atomic_json(out / 'failure.json', dict(type=type(error).__name__, message=str(error),
                                                  last_committed_step=step, completed=False))
        raise
    finally:
        if loader: loader.shutdown(wait=True)
        if ledger: ledger.close('completed' if completed else 'stopped', run=str(out), step=step)
        if world > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
