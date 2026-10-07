"""Freeze a balanced validation panel and delegate to the unchanged state scorer.

CPU preparation:
  python -m experiments.performance.diagnostic_panel prepare --paths PATHS \
      --checkpoint LAST_PT --output SNAPSHOT_DIR
Explicit bounded evaluation (never started by prepare):
  python -m experiments.performance.diagnostic_panel run --directory SNAPSHOT_DIR \
      --device cuda:0 --gpu-hours 0.25

This is development checkpoint diagnosis, not a replacement for final evaluation.
Each selected family retains all four audited cells. The existing scorer uses
historical observations for prediction; future references only score predictions
and teacher-forced innovation likelihood. Missing tracks remain in its population.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_SEEDS = (101, 211, 307, 401, 503, 601, 701, 809)
COUNT = 64


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def freeze_json(path, value):
    path = Path(path)
    with path.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def snapshot_checkpoint(source, destination):
    """An open inode survives the trainer's atomic last.pt replacement."""
    destination = Path(destination)
    with Path(source).open('rb') as incoming, destination.open('xb') as outgoing:
        before = os.fstat(incoming.fileno())
        for block in iter(lambda: incoming.read(8 * 1024 * 1024), b''):
            outgoing.write(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())
        after = os.fstat(incoming.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        destination.unlink()
        raise ValueError('Checkpoint was modified in place while copying')
    destination.chmod(0o444)


def layout_id(row):
    match = re.fullmatch(r'native_layout_(\d+)_intervention_\d+', row['scene_parent_id'])
    if not match:
        raise ValueError('Unknown layout identity: ' + row['scene_parent_id'])
    layout = int(match.group(1))
    if not row['family_id'].startswith('l%02d_' % layout):
        raise ValueError('Family and parent layout disagree')
    return layout


def select_panel(index, seed=17029):
    if index.get('schema') != 'agcr-intervention-index-1' or index.get('audit_passed') is not True:
        raise ValueError('Requires the original audited family index')
    parents, identities, buckets = {}, set(), defaultdict(list)
    for row in index['families']:
        if row['family_id'] in identities:
            raise ValueError('Duplicate family identity')
        identities.add(row['family_id'])
        parent = row['scene_parent_id']
        if parent in parents and parents[parent] != row['split']:
            raise ValueError('Scene parent leaks across dataset splits')
        parents[parent] = row['split']
        if row['split'] != 'val':
            continue
        if row['cohort'] not in ('evidence', 'ambiguous') or row['receiver'] not in (0, 1):
            raise ValueError('Unsupported validation cohort or receiver')
        records = row['records']
        if len(records) != 4 or {(r['world'], r['camera']) for r in records} != {(w, c) for w in range(2) for c in range(2)}:
            raise ValueError('Panel selection cannot split factorial families')
        buckets[(row['cohort'], row['receiver'], layout_id(row))].append(row)
    layouts = sorted({key[2] for key in buckets})
    if not layouts:
        raise ValueError('Empty validation split')
    # Rotate remainder allocations across the four cohort/direction groups.
    # For the native five layouts: each group has 16; cells have 3/4; totals 12/13.
    rank = lambda value: hashlib.sha256((str(seed) + ':' + str(value)).encode()).hexdigest()
    layout_order = sorted(layouts, key=rank)
    groups = [(cohort, receiver) for cohort in ('ambiguous', 'evidence') for receiver in (0, 1)]
    base, remainder = divmod(COUNT // len(groups), len(layouts))
    if base == 0:
        raise ValueError('64 families cannot cover every cohort/receiver/layout cell')
    selected = []
    for group_index, (cohort, receiver) in enumerate(groups):
        extras = {layout_order[(group_index * remainder + j) % len(layouts)] for j in range(remainder)}
        for layout in layouts:
            take = base + int(layout in extras)
            rows = sorted(buckets[(cohort, receiver, layout)], key=lambda row: rank(row['family_id']))
            if len(rows) < take:
                raise ValueError('Insufficient families in stratum %s' % ((cohort, receiver, layout),))
            selected.extend(rows[:take])
    assert len(selected) == COUNT
    return selected


def implementation_hashes():
    directories = ('paired_interventions', 'formal_evaluation', 'paper_experiments/agcr_exp')
    paths = [p for directory in directories for p in (ROOT / 'experiments' / directory).glob('*.py')]
    paths.append(Path(__file__).resolve())
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def prepare(paths_path, checkpoint, output, seed=17029):
    import torch
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = read(paths_path)
    index_path = Path(paths['families'])
    original_hash = digest(index_path)
    index = read(index_path)
    if digest(index_path) != original_hash:
        raise ValueError('Family index changed during preparation')
    selected = select_panel(index, seed)
    snapshot_checkpoint(checkpoint, output / 'checkpoint.pt')
    payload = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=False)
    identity = payload['identity']
    expected = dict(family_index_sha256=original_hash, statistics_sha256=digest(paths['statistics']))
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError('Checkpoint provenance mismatch: ' + key)
    if not any(name.startswith('upstream.') for name in payload['model']):
        raise ValueError('A learned intervention checkpoint with upstream weights is required')
    metadata = dict(step=payload['step'], completed=bool(payload.get('completed')),
                    training_development_only=bool(identity['development_only']),
                    phase=identity['phase'], variant=identity['variant'], seed=identity['seed'])
    del payload
    # Audit is inherited for unchanged whole families. Hash verification of all
    # selected files still occurs in the original FamilyDataset when scoring.
    subset = dict(index, families=selected, development_only=True,
                  source_index_sha256=original_hash,
                  diagnostic_subset=True, audit_scope='unchanged_complete_families_from_audited_parent')
    subset.pop('collection_quotas', None)
    subset['split_sha256'] = hashlib.sha256(json.dumps(
        [(r['family_id'], r['scene_parent_id'], r['split']) for r in selected], sort_keys=True).encode()).hexdigest()
    freeze_json(output / 'panel-index.json', subset)
    freeze_json(output / 'paths.json', paths)
    counts = Counter((r['cohort'], r['receiver'], layout_id(r)) for r in selected)
    contract = dict(schema='agcr-fixed64-state-diagnostic-1', development_only=True,
                    formal_experiment_results=False, split='val', count=COUNT, panel_seed=seed,
                    checkpoint=metadata, original_index_sha256=original_hash,
                    sample_seeds=list(SAMPLE_SEEDS), draws=8,
                    family_ids=[r['family_id'] for r in selected],
                    strata=[dict(cohort=c, receiver=r, layout=l, count=n) for (c, r, l), n in sorted(counts.items())],
                    source_sha256=implementation_hashes(),
                    conditions='history_inferred_state_only', future_reference_conditioning=False,
                    scoring='unchanged state_likelihood; teacher-forced reference used only for innovation density',
                    missing_policy='all history-eligible targets retained; null likelihood is not a dropped family',
                    assets={name: digest(paths[name]) for name in ('statistics', 'upstream')},
                    snapshots={name: digest(output / name) for name in ('checkpoint.pt', 'panel-index.json', 'paths.json')})
    freeze_json(output / 'contract.json', contract)
    return contract


def scoring_command(directory, device, gpu_hours):
    if not 0 < gpu_hours < float('inf'):
        raise ValueError('An explicit positive bounded evaluation allowance is required')
    directory = Path(directory).resolve()
    return [sys.executable, '-m', 'experiments.formal_evaluation.state_likelihood',
            '--paths', str(directory / 'paths.json'), '--checkpoint', str(directory / 'checkpoint.pt'),
            '--index', str(directory / 'panel-index.json'), '--output', str(directory / 'state-likelihood.json'),
            '--split', 'val', '--development', '--draws', '8', '--device', device,
            '--gpu-hours', str(gpu_hours)]


def run(directory, device, gpu_hours):
    directory = Path(directory).resolve()
    contract = read(directory / 'contract.json')
    if contract['schema'] != 'agcr-fixed64-state-diagnostic-1' or contract['sample_seeds'] != list(SAMPLE_SEEDS):
        raise ValueError('Unknown diagnostic contract')
    if implementation_hashes() != contract['source_sha256']:
        raise ValueError('Diagnostic implementation changed after the snapshot')
    for name, expected in contract['snapshots'].items():
        if digest(directory / name) != expected:
            raise ValueError('Immutable snapshot changed: ' + name)
    paths = read(directory / 'paths.json')
    for name, expected in contract['assets'].items():
        if digest(paths[name]) != expected:
            raise ValueError('Frozen input asset changed: ' + name)
    subprocess.run(scoring_command(directory, device, gpu_hours), check=True, cwd=ROOT)
    report = read(directory / 'state-likelihood.json')
    if report.get('development_only') is not True:
        raise ValueError('Panel diagnostics must remain development-only')
    rows = report['rows']
    observed = [(r['family_id'], r['world']) for r in rows]
    expected = {(family, world) for family in contract['family_ids'] for world in (0, 1)}
    if len(observed) != len(set(observed)) or not set(observed).issubset(expected):
        raise ValueError('Duplicate or unexpected scored worlds')
    if report['complete'] and set(observed) != expected:
        raise ValueError('Complete report lost a family/world, including possible missing tracks')
    for row in rows:
        if [sample['sample_seed'] for sample in row['state_samples']] != list(SAMPLE_SEEDS):
            raise ValueError('A scored world lost fixed sampling seeds')
    return dict(complete=report['complete'], development_only=True, scored_worlds=len(rows),
                expected_worlds=len(expected), contract_sha256=digest(directory / 'contract.json'),
                report_sha256=digest(directory / 'state-likelihood.json'),
                reference_history_targets=sum(r['reference_history_targets'] for r in rows),
                matched_targets=sum(r['matched_targets'] for r in rows),
                worlds_without_likelihood=sum(r['likelihood'] is None for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    create = actions.add_parser('prepare')
    for name in ('paths', 'checkpoint', 'output'):
        create.add_argument('--' + name, required=True)
    create.add_argument('--seed', type=int, default=17029)
    execute = actions.add_parser('run')
    execute.add_argument('--directory', required=True)
    execute.add_argument('--device', default='cuda:0')
    execute.add_argument('--gpu-hours', type=float, required=True)
    args = parser.parse_args()
    result = (prepare(args.paths, args.checkpoint, args.output, args.seed) if args.action == 'prepare'
              else run(args.directory, args.device, args.gpu_hours))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
