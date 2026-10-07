"""TRAIN-only refit of the two existing Bernoulli heads on frozen features.

prepare freezes the common family/configuration contract; cache runs bounded
inference; fit performs CPU-only optimization and exports an explicitly derived
checkpoint. No original checkpoint, training identity, or sampler is modified.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import torch
from torch import nn
from torch.nn import functional as F

from experiments.paired_interventions.data import file_sha256


SCHEMA = 'agcr-base-presence-refit-1'
HEADS = ('upstream.tracker.presence', 'upstream.dynamics.presence')
HEAD_KEYS = tuple(prefix + suffix for prefix in HEADS for suffix in ('.weight', '.bias'))
CONFIG = dict(families=64, selection_seed=17029, updates=256, batch_worlds=8,
              learning_rate=.001, weight_decay=.01, clip_norm=1.,
              fit_seed=17029, modes=8, horizon=16, prefix_draws_per_mode=1,
              initial_population='all_historical_trace_tracks_including_unmatched',
              future_population='unique_history_eligible_previous_present',
              world_cells=[[0, 0], [1, 0]], class_rebalancing=False,
              validation_optimization=False)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def tensor_digest(state, exclude=()):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name in exclude:
            continue
        if not isinstance(value, torch.Tensor):
            raise ValueError('Checkpoint model must contain tensors only')
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def train_families(index):
    if (index.get('schema') != 'agcr-intervention-index-1'
            or not index.get('audit_passed') or index.get('development_only')):
        raise ValueError('An audited formal family index is required')
    ids, parents, rows = set(), {}, []
    for row in index['families']:
        identity, parent, split = row['family_id'], row['scene_parent_id'], row['split']
        if identity in ids or (parent in parents and parents[parent] != split):
            raise ValueError('Duplicate family or scene parent crossing splits')
        ids.add(identity); parents[parent] = split
        if split == 'train':
            if {(r['world'], r['camera']) for r in row['records']} != {(w, c) for w in range(2) for c in range(2)}:
                raise ValueError('Training family must retain all four audited cells')
            rows.append(row)
    return rows


def select_training_panel(index, count=64, seed=17029):
    """No VAL/test sample or outcome can affect this hashed stratified selection."""
    from experiments.performance.diagnostic_panel import layout_id
    if count != 64:
        raise ValueError('The fixed supplement uses 64 TRAIN families')
    buckets = defaultdict(list)
    for row in train_families(index):
        if row['cohort'] not in ('ambiguous', 'evidence') or row['receiver'] not in (0, 1):
            raise ValueError('Unknown training stratum')
        buckets[row['cohort'], row['receiver'], layout_id(row)].append(row)
    rank = lambda value: hashlib.sha256((str(seed)+':'+str(value)).encode()).hexdigest()
    chosen = []
    for cohort in ('ambiguous', 'evidence'):
        for receiver in (0, 1):
            layouts = sorted({k[2] for k in buckets if k[:2] == (cohort, receiver)}, key=rank)
            queues = {layout: sorted(buckets[cohort, receiver, layout], key=lambda r: rank(r['family_id'])) for layout in layouts}
            local = []
            while len(local) < count//4:
                advanced = False
                for layout in layouts:
                    if queues[layout] and len(local) < count//4:
                        local.append(queues[layout].pop(0)); advanced = True
                if not advanced:
                    raise ValueError('Not enough TRAIN families in a fixed stratum')
            chosen.extend(local)
    return chosen


def prepare(index_path):
    index = json.loads(Path(index_path).read_text())
    rows = select_training_panel(index)
    return dict(schema=SCHEMA+'-panel', split='train', configuration=dict(CONFIG),
                family_index_sha256=file_sha256(index_path),
                selected_families=rows, selection_uses_validation_or_test=False,
                selection_policy='hash_cohort_receiver_layout_round_robin_v1')


def validate_panel(panel, index_path):
    if panel != prepare(index_path):
        raise ValueError('Panel differs from the fixed TRAIN-only selection/configuration')


def validate_parent(payload, paths, protocol_path):
    from .state_eval import validate_contract
    if payload.get('schema', '').startswith(SCHEMA):
        raise ValueError('Refitting an already-derived checkpoint is forbidden')
    validate_contract(payload, paths, protocol_path, paths['families'])
    for key in HEAD_KEYS:
        expected = (1, 256) if key.endswith('.weight') else (1,)
        if key not in payload['model'] or tuple(payload['model'][key].shape) != expected:
            raise ValueError('Expected existing Linear(256,1) head: '+key)


@torch.no_grad()
def future_features(dynamics, history, ids, present, eligible, generator):
    """Exact prefix law/weighting of free_prefix_presence_objective; no labels in rollout."""
    from .model import focal_history_slot
    from experiments.paired_interventions.rollout import _initial, _step
    slots = [focal_history_slot(history, ids, k)[0]
             for k in torch.unique(ids[ids >= 0]).tolist() if bool(eligible[k])]
    like = history['shared']
    if history['empty'] or not slots:
        return dict(features=like.new_empty(8, 16, 0, 256).cpu(),
                    target=torch.empty(16, 0), active=torch.empty(16, 0, dtype=torch.bool),
                    mode_weights=torch.full((8,), 1/8), selected_slots=[],
                    selected_references=[], expected_references=int(eligible.sum()))
    if present.shape[0] != 16 or history['initial'].shape[0] != 1:
        raise ValueError('Expected one 16-step training world')
    tokens, scene, initial = history['tokens'], history['shared'], history['initial']
    mixture = dynamics(tokens, scene, initial[:, None])['mode_logits'][0].softmax(-1)
    if mixture.numel() != 8:
        raise ValueError('The fixed supplement enumerates all eight scene modes')
    modes = torch.arange(8, device=initial.device)
    expand = lambda value: value.expand(8, *value.shape[1:])
    state = _initial(expand(initial), expand(history['covariance']), generator)
    states = [state]
    for _ in range(15):
        output = dynamics(expand(tokens), expand(scene), torch.stack(states, 1), modes)
        states.append(_step(dynamics, states[-1], output, generator, .2))
    captured = []
    handle = dynamics.presence.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    try:
        output = dynamics(expand(tokens), expand(scene), torch.stack(states, 1), modes)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape != (8, 1, 16, initial.shape[1], 256):
        raise ValueError('Unexpected dynamics presence-head input contract')
    selected = torch.tensor(slots, device=ids.device)
    features = captured[0][:, 0].index_select(2, selected)
    expected_logits = output['presence_logits'][:, 0].index_select(2, selected)
    # Preserve the original GEMM shape/stride before selecting reference slots.
    replayed_logits = dynamics.presence(captured[0]).squeeze(-1)[:, 0].index_select(2, selected)
    if not torch.allclose(replayed_logits, expected_logits, rtol=1e-5, atol=1e-6):
        raise ValueError('Cached features do not reproduce the original presence head')
    target = present[:, ids[selected]].bool()
    active = torch.cat([eligible[ids[selected]][None], target[:-1]], 0)
    return dict(features=features.float().cpu(), target=target.float().cpu(),
                active=active.cpu(), mode_weights=mixture.float().cpu(),
                selected_slots=slots, selected_references=ids[selected].tolist(),
                expected_references=int(eligible.sum()))


@torch.no_grad()
def cache_world(system, history, labels, present, generator):
    from experiments.paper_experiments.agcr_exp.temporal_state import historical_assignment
    maps = historical_assignment(history['observations'], labels)
    def target(track):
        t, v, q = track['birth']
        return int(maps[t][v][q])
    tracks = [track for event in history['result']['trace'] for track in event['tracks']]
    features = torch.stack([track['token'] for track in tracks]) if tracks else history['shared'].new_empty(0, 256)
    initial = dict(features=features.detach().float().cpu(),
                   target=torch.tensor([float(target(track) >= 0) for track in tracks]),
                   births=[list(track['birth']) for track in tracks])
    final = history['result']['tracks']
    ids = torch.tensor([target(track) for track in final], device=history['shared'].device, dtype=torch.long)
    eligible = labels['history_detectable'].any(0).any(0)
    future = future_features(system.dynamics, history, ids, present, eligible, generator)
    final_features = torch.stack([track['token'] for track in final]) if final else features.new_empty(0, 256)
    return dict(initial=initial, future=future, final_history=dict(
        features=final_features.detach().float().cpu(), reference_ids=ids.cpu(),
        births=[list(track['birth']) for track in final], scores=[float(track['score']) for track in final]))


def validate_row(row, allowed):
    if row.get('split') != 'train' or row.get('family_id') not in allowed or row.get('world') not in (0, 1) or row.get('camera') != 0:
        raise ValueError('Only registered TRAIN worlds can enter the feature cache')
    initial, future = row['initial'], row['future']
    x, y = initial['features'], initial['target']
    z, target, active, weights = [future[k] for k in ('features', 'target', 'active', 'mode_weights')]
    if x.ndim != 2 or x.shape[1] != 256 or y.shape != x.shape[:1]:
        raise ValueError('Invalid initial-head feature contract')
    if z.ndim != 4 or z.shape[:2] != (8, 16) or z.shape[-1] != 256 or target.shape != z.shape[1:3] or active.shape != target.shape:
        raise ValueError('Invalid future-head feature contract')
    if active.dtype != torch.bool or weights.shape != (8,) or not torch.allclose(weights.sum(), torch.tensor(1.), atol=1e-6) or (weights < 0).any():
        raise ValueError('Invalid mode weights or previous-presence domain')
    for tensor in (x, y, z, target, weights):
        if tensor.device.type != 'cpu' or tensor.requires_grad or not torch.isfinite(tensor).all():
            raise ValueError('Cache requires finite detached CPU tensors')
    if not (((y == 0) | (y == 1)).all() and ((target == 0) | (target == 1)).all()):
        raise ValueError('Existence supervision must be binary')
    if target.shape[1] and (not active[0].all() or not torch.equal(active[1:], target[:-1].bool())):
        raise ValueError('Future loss must preserve the original previous-presence domain')


def make_heads(state):
    heads = nn.ModuleList([nn.Linear(256, 1), nn.Linear(256, 1)])
    for head, prefix in zip(heads, HEADS):
        head.load_state_dict({key: state[prefix+'.'+key].detach().float().cpu() for key in ('weight', 'bias')})
    if sum(p.numel() for p in heads.parameters()) != 514:
        raise ValueError('Only two existing 257-parameter heads may be fitted')
    return heads


def row_loss(heads, row):
    initial, future = row['initial'], row['future']
    zero = sum(parameter.sum()*0 for parameter in heads.parameters())
    a = F.binary_cross_entropy_with_logits(heads[0](initial['features']).squeeze(-1), initial['target']) if len(initial['target']) else zero
    if future['target'].numel():
        logits = heads[1](future['features']).squeeze(-1)
        bce = F.binary_cross_entropy_with_logits(logits, future['target'][None].expand_as(logits), reduction='none')
        per_mode = (bce*future['active'][None]).sum((1, 2))/future['active'].sum().clamp_min(1)
        b = (per_mode*future['mode_weights']).sum()
    else:
        b = zero
    return a, b


def fit_heads(state, rows, configuration):
    if configuration != CONFIG:
        raise ValueError('CPU supplement configuration differs from the frozen amendment')
    allowed = {row['family_id'] for row in rows}
    for row in rows:
        validate_row(row, allowed)
    if not rows or not any(len(row['initial']['target']) for row in rows) or not any(row['future']['target'].numel() for row in rows):
        raise ValueError('Both heads need nonempty TRAIN supervision')
    frozen_before = tensor_digest(state, HEAD_KEYS)
    heads = make_heads(state)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=configuration['learning_rate'], weight_decay=configuration['weight_decay'])
    def means():
        with torch.no_grad():
            values = [torch.stack(row_loss(heads, row)) for row in rows]
            return torch.stack(values).mean(0).tolist()
    before = means(); log = []
    generator = torch.Generator().manual_seed(configuration['fit_seed'])
    queue = []
    for step in range(configuration['updates']):
        while len(queue) < configuration['batch_worlds']:
            queue.extend(torch.randperm(len(rows), generator=generator).tolist())
        chosen, queue = queue[:configuration['batch_worlds']], queue[configuration['batch_worlds']:]
        optimizer.zero_grad(set_to_none=True)
        parts = torch.stack([torch.stack(row_loss(heads, rows[index])) for index in chosen]).mean(0)
        loss = parts.sum(); loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in heads.parameters()):
            raise FloatingPointError('Nonfinite or missing head gradient')
        norm = torch.nn.utils.clip_grad_norm_(heads.parameters(), configuration['clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        log.append(dict(update=step+1, initial_BCE=float(parts[0]), future_BCE=float(parts[1]), gradient_norm=float(norm)))
    result = dict(state)
    for head, prefix in zip(heads, HEADS):
        for name, tensor in head.state_dict().items():
            result[prefix+'.'+name] = tensor.detach().to(state[prefix+'.'+name].dtype).clone()
    if tensor_digest(result, HEAD_KEYS) != frozen_before or tensor_digest(state, HEAD_KEYS) != frozen_before:
        raise ValueError('Frozen feature/state/motion/geometry parameters changed')
    return result, dict(updates_completed=len(log), train_BCE_before=before,
                       train_BCE_after=means(), log=log, trainable_parameters=514,
                       frozen_parameter_sha256=frozen_before, validation_used=False,
                       optimizer_state=optimizer.state_dict())


def cache(args):
    from .train import BudgetLedger, atomic_checkpoint, source_hashes, assert_base_paths
    from .model import receiver_only_branch
    from experiments.paired_interventions.train import atomic_json
    from experiments.paired_interventions.data import FamilyDataset, historical_labels, to_device
    from experiments.paired_interventions.bridge import historical_posterior
    from experiments.paired_interventions.world import load_upstream
    from experiments.paper_experiments.agcr_exp.stage2_system import FrozenImageEncoders
    paths = json.loads(Path(args.paths).read_text()); panel = json.loads(Path(args.panel).read_text())
    assert_base_paths(args.output, args.budget_ledger, paths)
    validate_panel(panel, paths['families'])
    parent = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    validate_parent(parent, paths, args.protocol)
    contract = dict(schema=SCHEMA+'-cache', split='train', parent_checkpoint_sha256=file_sha256(args.checkpoint),
                    parent_identity=parent['identity'], panel_sha256=file_sha256(args.panel),
                    family_index_sha256=file_sha256(paths['families']), helper_sha256=file_sha256(__file__),
                    runtime_source_sha256=source_hashes(), configuration=panel['configuration'],
                    families=[r['family_id'] for r in panel['selected_families']],
                    receiver_only=bool(parent['identity']['variant_options'].get('receiver_only')),
                    frozen_parameter_sha256=tensor_digest(parent['model'], HEAD_KEYS),
                    parent_model_sha256=tensor_digest(parent['model']), validation_or_test_used=False)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    manifest = out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError('Feature-cache contract changed')
    atomic_json(manifest, contract)
    ledger = BudgetLedger(args.budget_ledger, args.gpu_hours, 1); stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    rows = []
    try:
        torch.set_num_threads(4); device = torch.device(args.device)
        if device.type != 'cuda':
            raise ValueError('Native frozen-feature caching needs an explicit bounded CUDA device')
        torch.cuda.set_device(device)
        system, _ = load_upstream(paths['upstream'], paths['statistics'], device)
        system.load_state_dict({k[len('upstream.'):]:v for k,v in parent['model'].items() if k.startswith('upstream.')}, strict=True)
        system.eval().requires_grad_(False)
        del parent
        encoders = FrozenImageEncoders(paths['dino_code'], paths['dino_weights'], paths['wan']+'/vae', device)
        dataset = FamilyDataset(paths['families'], 'train', verify_files=False)
        positions = {row['family_id']:i for i,row in enumerate(dataset.families)}
        for family in panel['selected_families']:
            for world in (0, 1):
                name = family['family_id']+'_w'+str(world)+'.pt'; path = out/name
                if stop[0] or ledger.exhausted():
                    return
                if not path.exists():
                    record = next(r for r in family['records'] if (r['world'], r['camera']) == (world, 0))
                    if file_sha256(dataset.root/record['path']) != record['sha256']:
                        raise ValueError('TRAIN archive differs from its accepted hash')
                    raw, _ = dataset.branch(positions[family['family_id']], (world, 0))
                    keep = ('history_state', 'class_id', 'history_visible_fraction', 'history_projection_area', 'present')
                    branch = dict(inputs=raw['inputs'], supervision={k:raw['supervision'][k] for k in keep})
                    del raw
                    if contract['receiver_only']:
                        branch = receiver_only_branch(branch, family['receiver'])
                    branch = to_device(branch, device)
                    with torch.no_grad():
                        history = historical_posterior(system, branch['inputs'], encoders.history(branch['inputs']['rgb']))
                        seed = int(hashlib.sha256((str(CONFIG['selection_seed'])+':'+family['family_id']+':'+str(world)).encode()).hexdigest()[:15], 16)
                        generator = torch.Generator(device=device).manual_seed(seed)
                        row = cache_world(system, history, historical_labels(branch), branch['supervision']['present'], generator)
                    row.update(split='train', family_id=family['family_id'], world=world, camera=0,
                               contract_sha256=json_hash(contract), archive_sha256=record['sha256'])
                    validate_row(row, contract['families']); atomic_checkpoint(path, row)
                    del history, branch, row
                row = torch.load(path, map_location='cpu', weights_only=False)
                validate_row(row, contract['families'])
                if row['contract_sha256'] != json_hash(contract):
                    raise ValueError('Cached world belongs to a different parent/contract')
                rows.append(dict(path=name, sha256=file_sha256(path), family_id=family['family_id'], world=world))
                atomic_json(out/'progress.json', dict(complete=False, worlds=len(rows), expected=128))
        atomic_json(out/'complete.json', dict(schema=SCHEMA+'-cache-completion', complete=len(rows)==128,
            contract_sha256=json_hash(contract), rows=rows, validation_or_test_used=False))
    finally:
        ledger.close('cache_finished', cached_worlds=len(rows))


def validate_derived(payload, parent, parent_sha256):
    """Explicit admission helper; never disguise a derivative as a 320-step run."""
    if payload.get('schema') != SCHEMA+'-checkpoint' or not payload.get('refit', {}).get('completed'):
        raise ValueError('A completed, explicitly derived presence checkpoint is required')
    derivation = payload['refit']
    if derivation.get('parent_checkpoint_sha256') != parent_sha256 or payload.get('parent_training') != dict(identity=parent['identity'], step=parent['step'], completed=parent['completed'], validation_best=parent['validation_best']):
        raise ValueError('Derived checkpoint parent lineage mismatch')
    if 'identity' in payload or 'completed' in payload or 'validation_best' in payload:
        raise ValueError('Derived result must not impersonate the original completion contract')
    if derivation.get('configuration') != CONFIG or derivation.get('updates_completed') != 256 or derivation.get('validation_used') is not False:
        raise ValueError('Unregistered supplemental optimization')
    before = tensor_digest(parent['model'], HEAD_KEYS)
    if (set(payload['model']) != set(parent['model']) or tensor_digest(payload['model'], HEAD_KEYS) != before
            or derivation.get('frozen_parameter_sha256') != before):
        raise ValueError('Only the four presence-head tensors may differ from the parent')
    for key in HEAD_KEYS:
        if payload['model'][key].shape != parent['model'][key].shape or payload['model'][key].dtype != parent['model'][key].dtype or not torch.isfinite(payload['model'][key]).all():
            raise ValueError('Invalid derived presence-head tensor')
    return derivation


def fit(args):
    from .train import atomic_checkpoint
    from experiments.paired_interventions.train import atomic_json
    started = time.monotonic(); folder = Path(args.cache)
    contract = json.loads((folder/'manifest.json').read_text())
    complete = json.loads((folder/'complete.json').read_text())
    if contract.get('schema') != SCHEMA+'-cache' or contract.get('split') != 'train' or contract.get('validation_or_test_used') is not False:
        raise ValueError('Only the native TRAIN-only cache may be fitted')
    if not complete.get('complete') or complete['contract_sha256'] != json_hash(contract) or len(complete['rows']) != 128:
        raise ValueError('Incomplete feature cache')
    if contract['helper_sha256'] != file_sha256(__file__) or contract['configuration'] != CONFIG:
        raise ValueError('Supplement source or fixed configuration changed')
    parent_sha = file_sha256(args.checkpoint)
    parent = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if parent_sha != contract['parent_checkpoint_sha256'] or tensor_digest(parent['model']) != contract['parent_model_sha256'] or parent['identity'] != contract['parent_identity']:
        raise ValueError('CPU refit parent differs from the frozen-feature parent')
    rows = []; seen = set()
    for record in complete['rows']:
        path = (folder/record['path']).resolve()
        if folder.resolve() not in path.parents or file_sha256(path) != record['sha256']:
            raise ValueError('Feature artifact changed or escaped its cache')
        row = torch.load(path, map_location='cpu', weights_only=False)
        validate_row(row, contract['families'])
        key = (row['family_id'], row['world'])
        if key in seen or row['contract_sha256'] != json_hash(contract):
            raise ValueError('Duplicate/wrong-contract training world')
        seen.add(key); rows.append(row)
    if seen != {(family, world) for family in contract['families'] for world in (0, 1)}:
        raise ValueError('Every registered TRAIN world must be fitted exactly once per cache')
    output = Path(args.output)
    if 'agcr-intervention-runs' in output.resolve().parts:
        raise ValueError('Preserved Pro outputs are read-only')
    if output.exists():
        existing = torch.load(output, map_location='cpu', weights_only=False)
        refit = validate_derived(existing, parent, parent_sha)
        if (refit.get('cache_manifest_sha256') != file_sha256(folder/'manifest.json')
                or refit.get('cache_completion_sha256') != file_sha256(folder/'complete.json')
                or refit.get('helper_sha256') != file_sha256(__file__)
                or len(existing.get('refit_updates', [])) != 256):
            raise ValueError('Existing derived artifact differs; it cannot be overwritten or refitted')
        atomic_json(output.with_suffix('.json'), dict(schema=SCHEMA+'-report',
            checkpoint_sha256=file_sha256(output), refit=refit, updates=existing['refit_updates']))
        return
    torch.set_num_threads(args.threads)
    weights, report = fit_heads(parent['model'], rows, contract['configuration'])
    optimizer = report.pop('optimizer_state')
    log = report.pop('log')
    refit = dict(report, completed=True, configuration=CONFIG,
                 parent_checkpoint_sha256=parent_sha, parent_checkpoint_path=str(Path(args.checkpoint).resolve()),
                 cache_manifest_sha256=file_sha256(folder/'manifest.json'), cache_completion_sha256=file_sha256(folder/'complete.json'),
                 panel_sha256=contract['panel_sha256'], family_index_sha256=contract['family_index_sha256'],
                 training_families=contract['families'], training_worlds=128,
                 runtime_source_sha256=contract['runtime_source_sha256'], helper_sha256=file_sha256(__file__),
                 optimizer='AdamW', device='cpu', cpu_seconds=time.monotonic()-started,
                 new_architecture_parameters=0, motion_and_geometry_weights_unchanged=True,
                 status='supplemental_fit_complete_quality_not_yet_evaluated')
    payload = dict(schema=SCHEMA+'-checkpoint', model=weights, refit=refit,
                   parent_training=dict(identity=parent['identity'], step=parent['step'], completed=parent['completed'], validation_best=parent['validation_best']),
                   refit_optimizer=optimizer, refit_updates=log)
    validate_derived(payload, parent, parent_sha)
    output.parent.mkdir(parents=True, exist_ok=True); atomic_checkpoint(output, payload)
    atomic_json(output.with_suffix('.json'), dict(schema=SCHEMA+'-report', checkpoint_sha256=file_sha256(output), refit=refit, updates=log))


def verify(args):
    from experiments.paired_interventions.train import atomic_json
    parent_sha = file_sha256(args.parent)
    parent = torch.load(args.parent, map_location='cpu', weights_only=False)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    refit = validate_derived(payload, parent, parent_sha)
    panel = json.loads(Path(args.panel).read_text())
    if (refit.get('panel_sha256') != file_sha256(args.panel)
            or refit.get('helper_sha256') != file_sha256(__file__)
            or panel.get('schema') != SCHEMA+'-panel' or panel.get('split') != 'train'
            or panel.get('configuration') != CONFIG
            or len(panel.get('selected_families', [])) != 64
            or any(row.get('split') != 'train' for row in panel['selected_families'])
            or refit.get('training_families') != [row['family_id'] for row in panel['selected_families']]):
        raise ValueError('Derived checkpoint does not match the fixed TRAIN panel/source')
    if [row.get('update') for row in payload.get('refit_updates', [])] != list(range(1, 257)):
        raise ValueError('Complete supplemental update provenance is required')
    atomic_json(args.output, dict(schema=SCHEMA+'-verification', verified=True,
        checkpoint_sha256=file_sha256(args.checkpoint), parent_checkpoint_sha256=parent_sha,
        panel_sha256=file_sha256(args.panel), frozen_parameter_sha256=refit['frozen_parameter_sha256'],
        head_keys=list(HEAD_KEYS), updates_completed=256, validation_used=False,
        quality_evaluation_passed=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare'); p.add_argument('--index', required=True); p.add_argument('--output', required=True)
    p = commands.add_parser('cache')
    for key in ('paths', 'protocol', 'panel', 'checkpoint', 'output', 'budget-ledger'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--gpu-hours', type=float, required=True); p.add_argument('--device', default='cuda:0')
    p = commands.add_parser('fit')
    for key in ('cache', 'checkpoint', 'output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--threads', type=int, default=4)
    p = commands.add_parser('verify')
    for key in ('checkpoint', 'parent', 'panel', 'output'):
        p.add_argument('--'+key, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.index); output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    elif args.command == 'cache':
        cache(args)
    elif args.command == 'verify':
        verify(args)
    else:
        fit(args)


if __name__ == '__main__':
    main()
