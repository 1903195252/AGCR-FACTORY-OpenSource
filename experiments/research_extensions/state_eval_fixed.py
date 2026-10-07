"""Fixed whole-family VAL state comparison; references are scoring-only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import signal
import time
import statistics

import torch

from .region_probe import atomic_json, load_state_overlay, sha256


def selected_families(panel, index):
    if panel.get('test_used') is not False or panel.get('source_index_sha256') != sha256(index):
        raise ValueError('A locked native panel without TEST is required')
    source = json.loads(Path(index).read_text())
    if source.get('schema') != 'agcr-intervention-index-1' or not source.get('audit_passed'):
        raise ValueError('Audited native index required')
    indexed = {r['family_id']: r for r in source['families']}
    if any(row != indexed.get(row['family_id']) or row['split'] not in ('train', 'val') for row in panel['families']):
        raise ValueError('Panel changed a registered family')
    rows = [r for r in panel['families'] if r['split'] == 'val']
    if not rows or len({r['family_id'] for r in rows}) != len(rows):
        raise ValueError('Nonempty unique whole-family validation required')
    return rows


class EncodedHistory:
    def __init__(self, rgb, encoded):
        self.rgb, self.encoded = rgb, encoded

    def history(self, rgb):
        if rgb is not self.rgb:
            raise ValueError('Feature cache belongs to a different history')
        return self.encoded


def initial_errors(history, assignment, truth):
    rows = []
    if not history['empty']:
        for slot, reference in enumerate(assignment['ids'].tolist()):
            if reference < 0:
                continue
            delta = history['initial'][0, slot, :6]-truth[-1, reference, :6]
            rows.append(dict(reference=reference, slot=slot,
                position_m=float(delta[:3].norm()), velocity_mps=float(delta[3:6].norm())))
    return rows


def detailed_summary(rows, aggregate):
    result = aggregate(rows)
    result['strata'] = {cohort+'_r'+str(receiver): aggregate([r for r in rows
        if r['cohort'] == cohort and r['receiver'] == receiver])
        for cohort, receiver in sorted({(r['cohort'], r['receiver']) for r in rows})}
    extras = {}
    readers = dict(presence_Brier=lambda r: r.get('presence_Brier'),
        teacher_forced_innovation_NLL=lambda r: (r.get('likelihood') or {}).get('per_active_step_NLL'),
        initial_position_m=lambda r: statistics.mean(e['position_m'] for e in r['initial_errors']) if r['initial_errors'] else None,
        initial_velocity_mps=lambda r: statistics.mean(e['velocity_mps'] for e in r['initial_errors']) if r['initial_errors'] else None)
    for name, reader in readers.items():
        families = defaultdict(list)
        for row in rows:
            families[row['family_id']].append(reader(row))
        values = [statistics.mean(v) for v in families.values() if v and all(x is not None for x in v)]
        extras[name] = dict(mean=statistics.mean(values) if len(values) == len(families) and values else None,
            available_only_mean=statistics.mean(values) if values else None,
            valid_families=len(values), total_families=len(families))
    result['additional_family_means'] = extras
    return result


@torch.no_grad()
def main():
    from experiments.base48.state_eval import KEEP_LABELS, score_world, aggregate, SAMPLE_SEEDS
    from experiments.paired_interventions.data import load_branch, to_device
    from experiments.paired_interventions.world import load_upstream
    from experiments.paper_experiments.agcr_exp.stage2_system import FrozenImageEncoders
    from experiments.base48 import state_eval
    from experiments.paired_interventions import rollout
    from experiments.formal_evaluation import state_likelihood, metrics

    p = argparse.ArgumentParser(description=__doc__)
    for name in ('paths', 'panel', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--checkpoints', nargs='+', required=True, help='Unique label=/absolute/checkpoint.pt')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-seconds', type=float, default=1800.)
    args = p.parse_args()
    if not 0 < args.max_seconds <= 3600:
        raise ValueError('Bounded <=3600 second diagnostic required')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists():
        raise FileExistsError('Use a fresh state comparison directory')
    paths = json.loads(Path(args.paths).read_text())
    panel = json.loads(Path(args.panel).read_text())
    families = selected_families(panel, paths['families'])
    root = Path(panel['source_root']).resolve()
    if root != Path(json.loads(Path(paths['families']).read_text())['root']).resolve():
        raise ValueError('Panel source root changed')
    checkpoints = dict(item.split('=', 1) for item in args.checkpoints)
    if len(checkpoints) != len(args.checkpoints) or any(not k for k in checkpoints):
        raise ValueError('Unique named checkpoint comparison required')
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    started = time.monotonic(); stop = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    def boundary():
        if stop[0] or time.monotonic()-started >= args.max_seconds-20:
            raise InterruptedError('State comparison wall budget reached')
    report = dict(schema='peerdream-direction-state-comparison-1', completed=False,
        development_only=True, formal_experiment_result=False, split='val',
        panel_sha256=sha256(args.panel), source_sha256=sha256(__file__),
        scorer_sources={str(Path(m.__file__).resolve()): sha256(m.__file__)
                        for m in (state_eval, rollout, state_likelihood, metrics)},
        sample_seeds=SAMPLE_SEEDS, families=[r['family_id'] for r in families], methods={},
        future_reference_scope='scoring_only; prediction inputs are historical observations',
        likelihood_scope='teacher-forced reference-prefix innovation density; distinct from free sampled trajectory errors',
        selection_uses_test=False, selected_worlds=2*len(families))
    models = {}
    try:
        seed = None
        for label, path in checkpoints.items():
            payload, lineage = load_state_overlay(path)
            if lineage['identity']['family_index_sha256'] != sha256(paths['families']):
                raise ValueError('Checkpoint native index differs')
            current_seed = lineage['identity']['seed']
            if seed is not None and current_seed != seed:
                raise ValueError('Native inference stream seed must match all arms')
            seed = current_seed
            model, _ = load_upstream(paths['upstream'], paths['statistics'], device)
            model.load_state_dict({k[len('upstream.'):]: v for k, v in payload['model'].items()
                                   if k.startswith('upstream.')}, strict=True)
            models[label] = model.eval().requires_grad_(False)
            report['methods'][label] = dict(checkpoint_sha256=sha256(path), lineage=lineage, rows=[])
            del payload
        encoders = FrozenImageEncoders(paths['dino_code'], paths['dino_weights'], paths['wan']+'/vae', device)
        for family in families:
            for world in (0, 1):
                boundary()
                record = next(r for r in family['records'] if r['world'] == world and r['camera'] == 0)
                source = (root/record['path']).resolve()
                if root not in source.parents or sha256(source) != record['sha256']:
                    raise ValueError('Audited validation branch changed')
                raw = load_branch(source)
                branch = to_device(dict(inputs=raw['inputs'], supervision={k: raw['supervision'][k] for k in KEEP_LABELS}), device)
                del raw
                encoded = EncodedHistory(branch['inputs']['rgb'], encoders.history(branch['inputs']['rgb']))
                for label, model in models.items():
                    boundary()
                    row, history, assignment = score_world(model, encoded, branch, family, world, seed)
                    row['initial_errors'] = initial_errors(history, assignment, branch['supervision']['history_state'])
                    row['native_sha256'] = record['sha256']
                    report['methods'][label]['rows'].append(row)
                    report['methods'][label]['summary'] = detailed_summary(report['methods'][label]['rows'], aggregate)
                    del history, assignment
                report['completed_worlds'] = len(next(iter(report['methods'].values()))['rows'])
                atomic_json(out/'report.json', report)
                atomic_json(out/'status.json', dict(state='running', completed=False, updated_unix=time.time(),
                    completed_clips=report['completed_worlds'], selected_clips=report['selected_worlds']))
                del encoded, branch
        report['completed'] = True
    except BaseException as error:
        report['error'] = type(error).__name__+': '+str(error)
        raise
    finally:
        report['updated_unix'] = time.time()
        report['wall_seconds'] = time.monotonic()-started
        atomic_json(out/'report.json', report)
        atomic_json(out/'completion.json', dict(completed=report['completed'], report_sha256=sha256(out/'report.json')))
        atomic_json(out/'status.json', dict(state='completed' if report['completed'] else 'failed',
            completed=report['completed'], updated_unix=time.time(), completed_clips=report.get('completed_worlds', 0),
            selected_clips=report['selected_worlds']))


if __name__ == '__main__':
    main()
