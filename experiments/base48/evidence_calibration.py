"""Fit evidence-loss units from paired TRAIN targets without changing the model.

This CPU-only calibration opens only state/presence arrays from camera-0 cells;
the alternate camera is not an additional world sample. Validation and test
records never contribute. The artifact records its exact input-array digest;
it relies on the existing dataset audit rather than rehashing large RGB arrays.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibrate(index_path, statistics_path):
    index_path, statistics_path = Path(index_path), Path(statistics_path)
    index = json.loads(index_path.read_text())
    statistics = json.loads(statistics_path.read_text())
    if (index.get('schema') != 'agcr-intervention-index-1' or not index.get('audit_passed')
            or index.get('development_only')):
        raise ValueError('Evidence calibration requires an audited formal family index')
    if statistics.get('role') != 'train':
        raise ValueError('Existing normalization must also have training-only provenance')
    floor = np.asarray(statistics['state']['std'][:6], dtype=np.float64)
    if floor.shape != (6,) or not np.isfinite(floor).all() or (floor <= 0).any():
        raise ValueError('Six positive finite original state scales are required')
    root = Path(index['root']).resolve()
    count, worlds, families = 0, 0, []
    mean, centered = np.zeros(6), np.zeros(6)
    digest = hashlib.sha256()
    for family in sorted(index['families'], key=lambda f: f['family_id']):
        if family['split'] != 'train':
            continue
        records = sorted((r for r in family['records'] if r['camera'] == 0), key=lambda r: r['world'])
        if [r['world'] for r in records] != [0, 1]:
            raise ValueError('Each training family requires exactly two camera-0 worlds')
        families.append(family['family_id'])
        for record in records:
            path = (root / record['path']).resolve()
            if root not in path.parents:
                raise ValueError('Training array path escapes the audited dataset')
            with np.load(path, allow_pickle=False) as arrays:
                states, present = arrays['world_state'], arrays['present']
            target = int(family['target_index'])
            if (states.ndim != 3 or states.shape[0] != 22 or states.shape[-1] != 11
                    or present.shape != states.shape[:2] or not 0 <= target < states.shape[1]):
                raise ValueError('Expected a 6+16 frame native training state/presence array')
            values = np.asarray(states[6:, target, :6], dtype=np.float64)
            available = np.asarray(present[6:, target], dtype=bool)
            digest.update(json.dumps([family['family_id'], record['world'], record['path']],
                                     separators=(',', ':')).encode())
            digest.update(np.asarray(values, dtype='<f8').tobytes())
            digest.update(available.tobytes())
            values = values[available]
            if not len(values) or not np.isfinite(values).all():
                raise ValueError('Training calibration world needs finite present target states')
            # Merge population moments without cancellation for constant axes.
            size, local_mean = len(values), values.mean(0)
            delta = local_mean-mean
            centered += ((values-local_mean)**2).sum(0)+delta**2*count*size/(count+size)
            mean += delta*size/(count+size)
            count += size
            worlds += 1
    if not families:
        raise ValueError('No training families for evidence calibration')
    std = np.sqrt(np.maximum(centered/count, 0))
    scales = np.maximum(floor, std)
    return dict(schema='agcr-base-evidence-calibration-1', split='train',
        policy='coordinatewise_max_original_training_scale_and_paired_training_population_std',
        coordinates=['p_x_m', 'p_y_m', 'p_z_m', 'u_x_mps', 'u_y_mps', 'u_z_mps'],
        family_index_sha256=sha256(index_path), statistics_sha256=sha256(statistics_path),
        training_families=families, training_worlds=worlds, present_target_steps=count,
        input_arrays_sha256=digest.hexdigest(), existing_archive_audit=bool(index['audit_passed']),
        original_scales=floor.tolist(), paired_training_mean=mean.tolist(),
        paired_training_std=std.tolist(), evidence_loss_scales=scales.tolist(),
        evidence_score_reduction='mean', model_state_normalization_changed=False,
        validation_or_test_targets_used=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'statistics', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    report = calibrate(args.index, args.statistics)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Calibration is an immutable protocol artifact; never overwrite a run.
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('training_worlds', 'present_target_steps',
                                          'evidence_loss_scales', 'evidence_score_reduction')}))


if __name__ == '__main__':
    main()
