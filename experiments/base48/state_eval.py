"""Base48 state evaluation with a common paired reference population.

Reuses the corrected scorer's numerical functions, with Base checkpoint/index
contracts and explicit receiver-only input filtering. Validation readiness is
separate from successful metric computation and from scientific improvement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal

import numpy as np
import torch

from .model import receiver_only_branch
from .train import BudgetLedger, SCHEMA, SAMPLE_SEEDS, audit_index_ancestry, source_hashes
from experiments.paired_interventions.data import FamilyDataset, file_sha256, historical_labels, to_device
from experiments.paired_interventions.bridge import historical_posterior
from experiments.paired_interventions.rollout import sample_shared_future
from experiments.paired_interventions.objectives import candidate_log_density
from experiments.paired_interventions.sample import stream_seed
from experiments.paired_interventions.world import load_upstream, query_labels
from experiments.paired_interventions.train import atomic_json
from experiments.paper_experiments.agcr_exp.stage2_system import FrozenImageEncoders, matched_state_supervision
from experiments.formal_evaluation.metrics import aligned_state
from experiments.formal_evaluation.state_likelihood import (
    unique_history_assignment, align_unique_slots, innovation_nll,
    persistent_target_rollouts, HISTORY_SELECTION_POLICY,
)
from experiments.performance.diagnostic_panel import layout_id


KEEP_LABELS = ('history_state', 'state', 'class_id', 'present', 'history_visible_fraction', 'history_projection_area')


def validation_panel(rows, count=20, seed=17029):
    """Fixed complete families, balanced cohort/receiver, spread across layouts."""
    if count not in (16, 20):
        raise ValueError('Readiness uses 16 or 20 registered validation families')
    rank = lambda value: hashlib.sha256((str(seed)+':'+str(value)).encode()).hexdigest()
    layouts = sorted({layout_id(row) for row in rows}, key=rank)
    chosen = []
    per_group = count//4
    for group, (cohort, receiver) in enumerate((c, r) for c in ('ambiguous', 'evidence') for r in (0, 1)):
        for offset in range(per_group):
            layout = layouts[(group*per_group+offset) % len(layouts)]
            candidates = [i for i, row in enumerate(rows) if row['cohort'] == cohort and row['receiver'] == receiver and layout_id(row) == layout]
            candidates.sort(key=lambda i: rank(rows[i]['family_id']))
            position = next((i for i in candidates if i not in chosen), None)
            if position is None:
                raise ValueError('Incomplete validation stratum')
            chosen.append(position)
    return chosen


def evaluation_inputs(branch, receiver, receiver_only):
    """Filter S0 inputs while preserving the original paired scoring labels."""
    if not receiver_only:
        return branch
    masked = receiver_only_branch(branch, int(receiver))
    return dict(branch, inputs=masked['inputs'])


def validate_contract(payload, paths, protocol_path, index_path, development=False):
    identity = payload.get('identity', {})
    if identity.get('schema') != SCHEMA or identity.get('phase') != 'state':
        raise ValueError('A Base48 state checkpoint is required')
    if not development and (identity.get('development_only') or not payload.get('completed') or not payload.get('validation_best')):
        raise ValueError('Formal evaluation requires a completed validation-selected Base checkpoint')
    if identity.get('source_sha256') != source_hashes():
        raise ValueError('Base model/training source changed after checkpointing')
    for key, path in [('family_index_sha256', paths['families']), ('statistics_sha256', paths['statistics']), ('protocol_sha256', protocol_path)]:
        if identity.get(key) != file_sha256(path):
            raise ValueError('Evaluation provenance differs: '+key)
    requested = file_sha256(index_path)
    ancestry = audit_index_ancestry(paths['families'])
    if requested != identity['family_index_sha256']:
        if not ancestry or requested != ancestry.get('source_index_sha256') or not ancestry.get('train_val_identical'):
            raise ValueError('Evaluation index is neither registered nor its verified original source')
    return dict(checkpoint_sha256=None, index_sha256=requested, ancestry=ancestry,
                receiver_only=bool(identity.get('variant_options', {}).get('receiver_only')))


def aggregate(rows):
    """Keep family as the aggregation unit and retain missing/null denominators."""
    history_total = sum(r['reference_history_targets'] for r in rows)
    history_matched = sum(r['matched_targets'] for r in rows)
    final_total = sum(s['final_reference_count'] for r in rows for s in r['state_samples'])
    final_matched = sum(s['final_matched_count'] for r in rows for s in r['state_samples'])
    families = {}
    for row in rows:
        families.setdefault(row['family_id'], []).append(row)
    means = {}
    for key in ('ADE_m', 'FDE_m', 'state_missing_rate', 'FDE_coverage'):
        values = []
        for worlds in families.values():
            samples = [s['metrics'].get(key) for r in worlds for s in r['state_samples']]
            values.append(None if not samples or any(v is None for v in samples) else float(np.mean(samples)))
        valid = [v for v in values if v is not None]
        means[key] = dict(mean=None if len(valid) != len(values) or not valid else float(np.mean(valid)),
                          available_only_mean=None if not valid else float(np.mean(valid)),
                          valid_families=len(valid), total_families=len(values))
    return dict(families=len(families), worlds=len(rows), reference_history_targets=history_total,
        unique_history_targets=history_matched, history_unique_coverage=None if not history_total else history_matched/history_total,
        final_reference_count=final_total, final_matched_count=final_matched,
        final_reference_coverage=None if not final_total else final_matched/final_total,
        duplicate_history_slots=sum(r['duplicate_history_slots'] for r in rows),
        unassociated_history_slots=sum(r['unassociated_history_slots'] for r in rows), family_means=means)


def readiness(summary, thresholds, applicable, complete):
    if not applicable:
        return dict(applicable=False, passed=None, reason='Receiver-only S0 is not a paired-video frontend')
    if thresholds is None:
        return dict(applicable=False, passed=None, reason='No registered validation readiness protocol')
    checks = {}
    for metric, key in [('history_unique_coverage', 'history_unique_coverage_min'),
                        ('final_reference_coverage', 'final_reference_coverage_min')]:
        threshold = thresholds.get(key)
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError('A preregistered [0,1] readiness threshold is required: '+key)
        value = summary.get(metric)
        checks[metric] = dict(value=value, threshold=threshold, passed=value is not None and value >= threshold)
    return dict(applicable=True, passed=bool(complete and all(c['passed'] for c in checks.values())),
                checks=checks, scope='Minimum frontend availability only; not evidence of accurate motion or method superiority')


@torch.no_grad()
def score_world(system, encoders, branch, metadata, world, training_seed):
    # Reference eligibility is computed BEFORE receiver-only filtering.
    past = historical_labels(branch)
    y = branch['supervision']
    eligible = past['history_detectable'].any(0).any(0)
    labels = dict(past, state=y['state'], present=y['present'], history_eligible=eligible)
    history = historical_posterior(system, branch['inputs'], encoders.history(branch['inputs']['rgb']))
    ids = query_labels(history, past)
    assignment = unique_history_assignment(history, ids, eligible)
    row = dict(family_id=metadata['family_id'], cohort=metadata['cohort'], receiver=metadata['receiver'], world=world,
               reference_history_targets=int(eligible.sum()), **assignment['stats'], likelihood=None, state_samples=[])
    if not history['empty'] and bool(assignment['selected_mask'].any()):
        _, _, known, old, reference, present = matched_state_supervision(history['result']['initial_state'], ids, labels)
        row['likelihood'] = innovation_nll(system.dynamics, history['result']['tokens'], history['shared'],
            old, reference, known, present, assignment['selected_mask'])
    existence = []
    for sample_index, seed in enumerate(SAMPLE_SEEDS):
        aligned = np.full(tuple(y['state'].shape), np.nan, np.float32)
        exists = np.zeros(tuple(y['present'].shape), bool)
        extras = dict(duplicate_present_slot_steps=0, unassociated_present_slot_steps=0, unmatched_present_slot_steps=0)
        mode = None
        if not history['empty']:
            generator = torch.Generator(device=history['shared'].device).manual_seed(stream_seed(training_seed, metadata['family_id'], seed, 'state'))
            draw = sample_shared_future(system.dynamics, history['tokens'], history['shared'], history['initial'],
                history['presence_logits'], generator, covariance=history['covariance'])
            mode = int(draw['mode'][0])
            aligned, exists, extras = align_unique_slots(draw['state'][0].cpu().numpy(), draw['present'][0].cpu().numpy(), assignment, len(eligible))
        truth, present, population = y['state'].cpu().numpy(), y['present'].cpu().numpy(), eligible.cpu().numpy()
        metric = aligned_state(aligned, exists, truth, present, population)
        final = present[-1]&population
        available = exists[-1]&np.isfinite(aligned[-1]).all(-1)
        row['state_samples'].append(dict(sample_index=sample_index, sample_seed=seed, scene_mode=mode, metrics=metric,
            final_reference_count=int(final.sum()), final_matched_count=int((final&available).sum()), **extras))
        existence.append(exists)
    population = np.broadcast_to(eligible.cpu().numpy(), tuple(y['present'].shape))
    row['presence_Brier'] = float(np.square(np.mean(existence, 0)-y['present'].cpu().numpy())[population].mean()) if population.any() else None
    return row, history, assignment


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('paths', 'protocol', 'checkpoint', 'output', 'budget-ledger'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--index', help='Registered Base index or verified original source index')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--gpu-hours', required=True, type=float)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--stop-file')
    args = parser.parse_args()
    if args.development and args.split != 'val':
        parser.error('Unfinished-checkpoint diagnostics use validation only')
    output = Path(args.output)
    if any('agcr-intervention-runs' in Path(p).resolve().parts for p in (output, args.budget_ledger)):
        parser.error('Pro outputs and budgets are read-only')
    ledger = BudgetLedger(args.budget_ledger, args.gpu_hours, 1)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    report = dict(schema='agcr-base48-state-evaluation-1', complete=False, development_only=args.development,
                  split=args.split, rows=[], ready_for_renderer=False)
    try:
        paths = json.loads(Path(args.paths).read_text())
        protocol = json.loads(Path(args.protocol).read_text())
        if protocol.get('schema') != 'agcr-base48-protocol-1' or protocol.get('paired_sample_seeds') != SAMPLE_SEEDS:
            raise ValueError('Registered Base protocol and fixed eight seeds required')
        index = args.index or paths['families']
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        contract = validate_contract(payload, paths, args.protocol, index, args.development)
        contract.update(checkpoint_sha256=file_sha256(args.checkpoint), protocol_sha256=file_sha256(args.protocol),
            implementation_sha256=file_sha256(__file__), numerical_scorer_sha256=file_sha256(Path(__file__).parents[1]/'formal_evaluation/state_likelihood.py'),
            split=args.split, sample_seeds=SAMPLE_SEEDS, development_only=args.development)
        dataset = FamilyDataset(index, args.split, verify_files=False)
        gate_config = protocol.get('evaluation', {}).get('state_gate') if args.split == 'val' else None
        if args.split == 'val' and gate_config is None:
            raise ValueError('Validation readiness panel must be registered in evaluation.state_gate')
        selected = validation_panel(dataset.families, int(gate_config.get('families', 20)), int(gate_config.get('seed', 17029))) if args.split == 'val' else list(range(len(dataset)))
        contract['families'] = [dataset.families[i]['family_id'] for i in selected]
        report.update(contract=contract, checkpoint_step=payload['step'], input_policy='receiver_only' if contract['receiver_only'] else 'paired',
                      reference_population='same paired-history-eligible targets for every input policy', sample_seeds=SAMPLE_SEEDS)
        cache = output.with_suffix('.families'); cache.mkdir(parents=True, exist_ok=True)
        contract_path = cache/'contract.json'
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError('State evaluation cache contract changed')
        atomic_json(contract_path, contract)
        if ledger.exhausted():
            return
        torch.set_num_threads(4)
        device = torch.device(args.device); torch.cuda.set_device(device)
        system, _ = load_upstream(paths['upstream'], paths['statistics'], device)
        system.load_state_dict({k[len('upstream.'):]: v for k, v in payload['model'].items() if k.startswith('upstream.')}, strict=True)
        system.eval().requires_grad_(False)
        seed = payload['identity']['seed']; del payload
        encoders = FrozenImageEncoders(paths['dino_code'], paths['dino_weights'], paths['wan']+'/vae', device)
        for position in selected:
            metadata = dataset.families[position]
            marker = cache/(metadata['family_id']+'.json')
            if marker.exists():
                report['rows'].extend(json.loads(marker.read_text())['rows'])
                continue
            if stop[0] or ledger.exhausted() or (args.stop_file and Path(args.stop_file).exists()):
                break
            for record in metadata['records']:
                if file_sha256(dataset.root/record['path']) != record['sha256']:
                    raise ValueError('Audited family changed')
            family_rows, reference_paths, histories = [], [], []
            for world in range(2):
                raw, _ = dataset.branch(position, (world, 0))
                branch = dict(inputs=raw['inputs'], supervision={k: raw['supervision'][k] for k in KEEP_LABELS})
                branch = evaluation_inputs(branch, metadata['receiver'], contract['receiver_only'])
                branch = to_device(branch, device); del raw
                row, history, assignment = score_world(system, encoders, branch, metadata, world, seed)
                family_rows.append(row); histories.append((history, assignment))
                reference_paths.append(branch['supervision']['state'][:, metadata['target_index'], :6])
                del branch
            references = torch.stack(reference_paths)
            for world, (history, assignment) in enumerate(histories):
                selected_target = torch.where(assignment['ids'] == metadata['target_index'])[0]
                density = []
                if len(selected_target):
                    k = int(selected_target[0])
                    for sample_seed in SAMPLE_SEEDS:
                        generator = torch.Generator(device=device).manual_seed(stream_seed(seed, metadata['family_id'], sample_seed, 'density'))
                        draws = persistent_target_rollouts(system.dynamics, history['tokens'], history['shared'], history['initial'],
                            history['presence_logits'], generator, k, covariance=history['covariance'], draws=4)
                        score = candidate_log_density(draws['state'][..., k, :6], draws['mode_logits'], references[None],
                            system.dynamics.scale.std[:6], .2, draws['log_survival'])
                        density.append(dict(sample_seed=sample_seed, log_density=score[0].cpu().tolist()))
                family_rows[world].update(density_samples=density, kernel_bandwidth=.2,
                    kernel_smoothed_log_density=np.mean([s['log_density'] for s in density], 0).tolist() if density else None,
                    kernel_survival_scope='selected_focal_target_only')
            atomic_json(marker, dict(rows=family_rows))
            report['rows'].extend(family_rows)
            report['summary'] = aggregate(report['rows'])
            atomic_json(output, report)
            ledger.commit('state_evaluation', completed_families=len(report['rows'])//2)
            del histories, history, reference_paths, references
        report['complete'] = len(report['rows']) == 2*len(selected)
        report['summary'] = aggregate(report['rows'])
        report['gate'] = readiness(report['summary'], gate_config,
            not contract['receiver_only'] and args.split == 'val', report['complete'])
        report['ready_for_renderer'] = bool(report['complete'] and not args.development and report['gate'].get('passed'))
    except BaseException as error:
        report['error'] = type(error).__name__+': '+str(error)
        raise
    finally:
        report['allocated_gpu_hours'] = ledger.used
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, report)
        ledger.close('complete' if report['complete'] else 'stopped', output=str(output), ready_for_renderer=report['ready_for_renderer'])


if __name__ == '__main__':
    main()
