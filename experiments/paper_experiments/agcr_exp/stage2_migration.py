"""Checked execution-only migration; preserve samples, optimizer and lineage."""
import json
import shutil
from pathlib import Path

import torch

from .data import read_json
from .train_observation import sha256


def migration_lineage(previous, current, checkpoint_path, report_path):
    proof = read_json(report_path)
    required = {'paired_batch16_equivalence', 'fixed_validation_equivalence', 'optimized_checkpoint_resume'}
    if proof.get('passed') is not True or proof.get('scope') != 'stage2_execution_migration':
        raise ValueError('Execution migration has not passed its numerical checks')
    if not required.issubset(proof.get('verified_features', [])):
        raise ValueError('Incomplete execution migration checks')
    if proof.get('origin_checkpoint_sha256') != sha256(checkpoint_path) or proof.get('origin_manifest') != previous:
        raise ValueError('Migration report checked different source weights or manifest')
    if proof.get('new_source_sha256') != current['source_sha256']:
        raise ValueError('Candidate source changed after migration validation')
    exempt = {'source_sha256', 'execution'}
    if {k:v for k,v in previous.items() if k not in exempt} != {k:v for k,v in current.items() if k not in exempt}:
        raise ValueError('Migration changed model, data, precision, protocol or validation panel')
    expected = dict(previous['execution'], policy='optimized_v1', voxel_chunk=65536, dynamics_checkpoint_above_tracks=8)
    if current['execution'] != expected or previous['execution'].get('policy', 'standard') != 'standard':
        raise ValueError('Unapproved execution-policy change')
    return dict(origin_checkpoint=str(Path(checkpoint_path).resolve()), origin_checkpoint_sha256=sha256(checkpoint_path),
                origin_step=proof['origin_step'], report=str(Path(report_path).resolve()), report_sha256=sha256(report_path))


def inherit_records(checkpoint_path, output, manifest, saved):
    origin, output = Path(checkpoint_path).parent, Path(output)
    if origin.resolve() == output.resolve():
        raise ValueError('Execution migration must preserve the original run directory')
    log = origin/'training.jsonl'
    if log.exists():
        lines = [line for line in log.read_text().splitlines() if json.loads(line)['step'] <= saved['step']]
        (output/'training.jsonl').write_text('\n'.join(lines)+'\n')
    for path in origin.glob('validation-*.json'):
        if path.stem[len('validation-'):].isdigit() and int(path.stem[len('validation-'):]) <= saved['step']:
            shutil.copy2(path, output/path.name)
    if saved['best_step'] is not None:
        best = torch.load(origin/'best.pt', map_location='cpu', weights_only=False)
        if best['step'] != saved['best_step'] or best['manifest'] != saved['manifest']:
            raise ValueError('Original best checkpoint does not match migration state')
        best['manifest'] = manifest
        torch.save(best, output/'best.pt')
