"""Matched ConcatGRU evidence-loss replication on immutable frozen histories.

This is a separate experiment, not a modification of the original baseline.
Every seed runs both coefficients from identical weights, with the same 2048
updates, family schedule and explicit sampling streams. Geometry supervision
is not added to the GRU's original objectives. Future references score free
rollouts only; they never supply a prediction prefix.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

import torch

from .region_probe import atomic_json, sha256
from .state_prediction_baselines import ConcatGRU
from .state_repair import ordinary_objective
from .train_candidate import atomic_checkpoint, branch_index

SCHEMA = 'agcr-concat-gru-evidence-replication-1'
SEEDS = (17017, 17029, 17043)
STEPS = 2048
ROOT = Path(__file__).resolve().parents[2]
SOURCES = ('experiments/research_extensions/train_concat_gru_evidence.py',
    'experiments/research_extensions/state_prediction_baselines.py',
    'experiments/research_extensions/state_repair.py',
    'experiments/research_extensions/train_candidate.py',
    'experiments/base48/model.py',
    'experiments/paired_interventions/rollout.py',
    'experiments/paired_interventions/objectives.py',
    'experiments/paper_experiments/agcr_exp/future_prediction.py')
ROW_KEYS = ('metadata', 'empty', 'history', 'reference_ids', 'history_eligible',
    'previous_reference', 'reference', 'present', 'eligible', 'future_state',
    'future_present', 'focal_reference', 'focal_slot')
HISTORY_KEYS = ('empty', 'tokens', 'shared', 'separate', 'initial', 'covariance',
                'presence_logits', 'result')


def tensor_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare(cache_path, output):
    """CPU-only compact copy; verify every source row and retain exact tensors."""
    cache, out = Path(cache_path), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    target = out/'compact.pt'
    if target.exists():
        receipt = json.loads((out/'compact.json').read_text())
        if receipt['sha256'] != sha256(target) or receipt['cache_manifest_sha256'] != sha256(cache/'manifest.json'):
            raise ValueError('Do not replace changed compact input')
        return receipt
    manifest = json.loads((cache/'manifest.json').read_text())
    data, records = manifest['data'], manifest['data']['branches']
    if (data['test_used'] is not False or data['split'] != 'train' or len(records) != 128
            or len({r['family_id'] for r in records}) != 64
            or any(r['split'] != 'train' or r['camera'] != 0 for r in records)):
        raise ValueError('Original complete 64-family TRAIN cache required')
    rows = []
    stamps = []
    for i, record in enumerate(records):
        path = cache/('%04d.pt' % i)
        stamp = json.loads(path.with_suffix('.json').read_text())
        if stamp != dict(metadata=record, sha256=sha256(path), manifest_sha256=sha256(cache/'manifest.json')):
            raise ValueError('Frozen historical cache changed')
        row = torch.load(path, map_location='cpu', weights_only=False)
        if row['metadata'] != record:
            raise ValueError('Cache metadata differ')
        compact = {k: row[k] for k in ROW_KEYS}
        compact['history'] = {k: row['history'][k] for k in HISTORY_KEYS}
        rows.append(compact); stamps.append(dict(row=i, sha256=stamp['sha256']))
    groups = [[i for i, r in enumerate(records) if r['family_id'] == fid]
              for fid in sorted({r['family_id'] for r in records})]
    if any(len(g) != 2 or [records[i]['world'] for i in g] != [0, 1] for g in groups):
        raise ValueError('Incomplete TRAIN world pair')
    atomic_checkpoint(target, dict(schema=SCHEMA+'-cache', manifest=manifest, rows=rows, groups=groups))
    receipt = dict(schema=SCHEMA+'-cache-receipt', path=str(target), sha256=sha256(target),
        source_cache=str(cache), cache_manifest_sha256=sha256(cache/'manifest.json'),
        source_rows=stamps, families=64, worlds=128, train_only=True, tensors_transformed=False)
    atomic_json(out/'compact.json', receipt)
    return receipt


def evidence_loss(model, rows, step, seed):
    """Reset GRU view context for each history before scoring its two futures."""
    from experiments.base48.model import persistent_target_rollouts, evidence_scores
    from experiments.paired_interventions.objectives import candidate_log_density, evidence_preference
    if [r['metadata']['world'] for r in rows] != [0, 1]:
        raise ValueError('World order required')
    if any(r['metadata']['cohort'] != 'evidence' for r in rows):
        raise ValueError('Only historically informative pairs receive branch preference')
    references = torch.stack([r['future_state'][:, r['focal_reference'], :6] for r in rows]).detach()
    losses = []
    for world, row in enumerate(rows):
        history, slot = row['history'], row['focal_slot']
        if history['empty'] or slot < 0:
            losses.append(next(model.parameters()).sum()*0)
            continue
        model.condition(history)
        generator = torch.Generator(device=history['initial'].device).manual_seed(seed+500009+2*step+world)
        bank = persistent_target_rollouts(model, history['tokens'], history['shared'], history['initial'],
            history['presence_logits'], generator, slot, covariance=history['covariance'], draws=2)
        score = candidate_log_density(bank['state'][..., slot, :6], bank['mode_logits'],
            references[None], model.scale.std[:6], .2, bank['log_survival'])
        losses.append(evidence_preference(evidence_scores(score, 96, 'mean'),
            torch.tensor([world], device=score.device)))
    return torch.stack(losses).sum()/2


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    return value


def fit(args):
    from experiments.base48.model import free_prefix_presence_objective
    from .typed_evidence_repeat_test import budget_guard
    reservation = budget_guard(args.budget_ledger, args.job_id,
        dict(ledger_path=args.budget_ledger, gpu_hour_cap=2., deadline_unix=args.deadline))
    if args.max_seconds > reservation['maximum_wall_seconds']-30:
        raise ValueError('Keep outer cleanup time')
    torch.set_num_threads(2)
    torch.cuda.set_device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda:0'
    compact = Path(args.compact)
    receipt = json.loads(compact.with_suffix('.json').read_text())
    if sha256(compact) != receipt['sha256']:
        raise ValueError('Compact input changed')
    bundle = torch.load(compact, map_location='cpu', weights_only=False)
    if bundle['schema'] != SCHEMA+'-cache':
        raise ValueError('Wrong training cache')
    rows = to_device(bundle['rows'], device)
    groups = bundle['groups']
    statistics = json.loads(Path(args.statistics).read_text())
    torch.manual_seed(args.seed)
    initial_model = ConcatGRU(statistics)
    initial = {k: v.detach().clone() for k, v in initial_model.state_dict().items()}
    initial_sha = tensor_digest(initial)
    del initial_model
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    began = time.monotonic()
    models, optimizers, cursors, totals = {}, {}, {}, {}
    for arm in (0, 1):
        folder = out/f'seed{args.seed}_ev{arm}'
        folder.mkdir(exist_ok=True)
        model = ConcatGRU(statistics).to(device)
        model.load_state_dict(initial, strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
        identity = dict(schema=SCHEMA, seed=args.seed, evidence_coefficient=arm*.1,
            steps=STEPS, lr=.001, weight_decay=.01, batch_worlds=2, train_families=64,
            history_frozen=True, future_truth_conditions=False, selection_uses_test=False,
            evidence_forward_retained=True, geometry_objective=False,
            original_concat_gru_objectives=True, compact_sha256=receipt['sha256'],
            cache_manifest_sha256=receipt['cache_manifest_sha256'],
            history_checkpoint=bundle['manifest']['history_checkpoint'],
            statistics_sha256=sha256(args.statistics), initialization_sha256=initial_sha,
            sources={s: sha256(ROOT/s) for s in SOURCES},
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            horizon=16, modes=8, evidence_draws_per_mode=2, evidence_bandwidth=.2,
            evidence_score_reduction='mean', precision='float32, TF32 disabled',
            objectives='original joint likelihood + autoregressive motion Huber + free-prefix presence + coefficient*paired evidence preference')
        cursor, seconds = 0, 0.
        cp = folder/'last.pt'
        if cp.exists():
            saved = torch.load(cp, map_location='cpu', weights_only=False)
            if saved['identity'] != identity:
                raise ValueError('Resume configuration changed')
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            cursor, seconds = saved['step'], saved['training_wall_seconds']
        else:
            atomic_json(folder/'identity.json', identity)
        models[arm], optimizers[arm], cursors[arm], totals[arm] = model, optimizer, cursor, seconds
    def save(arm):
        folder = out/f'seed{args.seed}_ev{arm}'
        identity = json.loads((folder/'identity.json').read_text())
        atomic_checkpoint(folder/'last.pt', dict(schema=SCHEMA, identity=identity,
            model=models[arm].state_dict(), optimizer=optimizers[arm].state_dict(),
            step=cursors[arm], completed=cursors[arm] == STEPS,
            training_wall_seconds=totals[arm]))
        atomic_json(folder/'completion.json', dict(completed=cursors[arm] == STEPS,
            step=cursors[arm], target_steps=STEPS, checkpoint_sha256=sha256(folder/'last.pt'),
            training_wall_seconds=totals[arm], identity=identity))
    for arm in (0, 1): save(arm)
    try:
        while min(cursors.values()) < STEPS:
            if (stopped[0] or time.time() >= args.deadline-20
                    or time.monotonic()-began >= args.max_seconds-20):
                break
            for arm in (0, 1):
                if cursors[arm] >= STEPS:
                    continue
                step, model, optimizer = cursors[arm], models[arm], optimizers[arm]
                indexes = groups[branch_index(step, len(groups), args.seed)]
                pair = [rows[i] for i in indexes]
                tick = time.monotonic(); optimizer.zero_grad(set_to_none=True)
                common_value = 0.
                for world, row in enumerate(pair):
                    if row['empty']:
                        continue
                    model.condition(row['history'])
                    common, _ = ordinary_objective(model, row)
                    rng = torch.Generator(device=device).manual_seed(args.seed+200003+2*step+world)
                    presence, _ = free_prefix_presence_objective(model, row['history'], row['reference_ids'],
                        row['future_present'], row['history_eligible'], rng)
                    base = (common+presence)/2
                    if not torch.isfinite(base): raise FloatingPointError('Nonfinite base objective')
                    base.backward(); common_value += float(base.detach())
                evidence = (evidence_loss(model, pair, step, args.seed)
                    if pair[0]['metadata']['cohort'] == 'evidence' else next(model.parameters()).sum()*0)
                if not torch.isfinite(evidence): raise FloatingPointError('Nonfinite evidence objective')
                (arm*.1*evidence).backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); torch.cuda.synchronize()
                elapsed = time.monotonic()-tick
                cursors[arm] += 1; totals[arm] += elapsed
                folder = out/f'seed{args.seed}_ev{arm}'
                atomic_json(folder/'status.json', dict(state='running', step=cursors[arm], target_steps=STEPS,
                    last_base_loss=common_value, last_evidence_loss=float(evidence.detach()),
                    gradient_norm=float(norm), step_seconds=elapsed, training_wall_seconds=totals[arm],
                    updated_unix=time.time(), training_device=device))
                if cursors[arm] % 64 == 0 or cursors[arm] == STEPS:
                    save(arm)
                    print(json.dumps(dict(seed=args.seed, arm=arm, step=cursors[arm],
                        training_seconds=round(totals[arm], 2))), flush=True)
    finally:
        for arm in (0, 1):
            save(arm)
            atomic_json(out/f'seed{args.seed}_ev{arm}'/'status.json', dict(
                state='completed' if cursors[arm] == STEPS else 'paused', step=cursors[arm],
                target_steps=STEPS, updated_unix=time.time(), training_wall_seconds=totals[arm]))
        atomic_json(out/'status.json', dict(state='completed' if min(cursors.values()) == STEPS else 'paused',
            steps=cursors, seed=args.seed, updated_unix=time.time(), invocation_seconds=time.monotonic()-began))
    return 0 if min(cursors.values()) == STEPS else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--cache', required=True); prep.add_argument('--output', required=True)
    train = sub.add_parser('fit')
    for name in ('compact', 'statistics', 'output', 'budget-ledger', 'job-id'):
        train.add_argument('--'+name, required=True)
    train.add_argument('--seed', type=int, choices=SEEDS, required=True)
    train.add_argument('--max-seconds', type=float, required=True)
    train.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.cache, args.output))); return 0
    return fit(args)


if __name__ == '__main__':
    raise SystemExit(main())
