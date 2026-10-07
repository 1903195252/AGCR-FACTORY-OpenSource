"""Fail-closed audit of four-cell MuJoCo intervention families.

CLI: python3 -m experiments.paired_interventions.audit manifest.json
Each manifest record names one NPZ with full 6+16 frames. Pixels are native
rendered outputs, never copied between branches to satisfy this audit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HISTORY = 6
REQUIRED = ('rgb', 'depth_mm', 'valid', 'K', 'camera_to_world', 'time_s',
            'world_state', 'present', 'instance', 'visible_fraction', 'projection_area',
            'class_id', 'instance_id', 'world_from_factory')


def digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str((array.shape, array.dtype.str)).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def equal(a, b, label):
    if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
        raise ValueError('Control invariant violated: ' + label)


def _history(data, view=None):
    result = {}
    for key in ('rgb', 'depth_mm', 'valid', 'K', 'camera_to_world'):
        array = data[key][:HISTORY]
        result[key] = array if view is None else array[:, view]
    result['time_s'] = data['time_s'][:HISTORY]
    return result


def _same_history(a, b, view=None):
    for key, array in _history(a, view).items():
        equal(array, _history(b, view)[key], 'history/' + key)


def audit_family(records, root):
    if len(records) != 4 or {(x['world'], x['camera']) for x in records} != {(a,b) for a in range(2) for b in range(2)}:
        raise ValueError('A family must contain the full 2-world x 2-camera factorial')
    fixed = ('family_id', 'scene_parent_id', 'split', 'cohort', 'receiver', 'target_index',
             'environment_sha256', 'nuisance_seed', 'passive_camera')
    for key in fixed:
        if any(x[key] != records[0][key] for x in records):
            raise ValueError('Family metadata differs: ' + key)
    first = records[0]
    if first['passive_camera'] is not True:
        raise ValueError('Camera path must have no feedback into world dynamics')
    if first['cohort'] not in ('evidence', 'ambiguous') or first['receiver'] not in (0,1):
        raise ValueError('Invalid cohort or receiver')
    k = first['target_index']
    data, hashes = {}, {}
    for record in records:
        key = (record['world'], record['camera'])
        with np.load(Path(root) / record['path'], allow_pickle=False) as archive:
            data[key] = {name: archive[name] for name in REQUIRED}
        item = data[key]
        if item['rgb'].shape[0:2] != (22,2) or item['rgb'].dtype != np.uint8:
            raise ValueError('Expected 22 synchronized paired RGB frames, uint8')
        if item['world_state'].shape[0] != 22 or item['world_state'].shape[-1] != 11:
            raise ValueError('Expected [22,K,11] physical states')
        if not np.all(item['present']) or k < 0 or k >= item['world_state'].shape[1]:
            raise ValueError('Core diagnostic requires persistent, indexed targets')
        for name in ('depth_mm', 'K', 'camera_to_world', 'world_state', 'time_s'):
            if not np.isfinite(item[name]).all():
                raise ValueError('Nonfinite data: ' + name)
        if not np.allclose(item['time_s'], (np.arange(22)-5)*.2, atol=1e-10, rtol=0):
            raise ValueError('Expected 6 historical and 16 future timestamps at 5 Hz')
        if not np.all(item['depth_mm'][~item['valid'].astype(bool)] == 0):
            raise ValueError('Invalid depth must be zero')
        target_count=item['world_state'].shape[1]
        if item['instance_id'].shape!=(target_count,) or not np.array_equal(item['instance_id'],np.arange(1,target_count+1)):
            raise ValueError('Instance IDs must match stable per-family target indices')
        if item['class_id'].shape!=(target_count,):
            raise ValueError('Class labels must cover the stable target set')
        pixel_counts=np.stack([(item['instance']==target+1).sum((-1,-2)) for target in range(target_count)],-1)
        area=item['projection_area']
        if area.shape!=pixel_counts.shape or (pixel_counts>area).any() or (area<0).any():
            raise ValueError('Visible pixels exceed isolated projection support')
        if not np.allclose(item['visible_fraction'],pixel_counts/np.maximum(area,1),atol=1e-6,rtol=0):
            raise ValueError('Visibility labels disagree with rendered instance pixels')
        hashes[str(key)] = {name: digest(array) for name,array in _history(item).items()}
    for world in range(2):
        a,b = data[world,0],data[world,1]
        _same_history(a,b)
        for field in ('world_state', 'present', 'K', 'time_s', 'class_id', 'instance_id', 'world_from_factory'):
            equal(a[field],b[field], 'camera intervention/' + field)
        if np.array_equal(a['camera_to_world'][HISTORY:],b['camera_to_world'][HISTORY:]):
            raise ValueError('Camera intervention has no changed future pose')
        if np.array_equal(a['instance'][HISTORY:] == k+1, b['instance'][HISTORY:] == k+1):
            raise ValueError('Camera intervention changes no target visible pixels')
        va=(a['visible_fraction'][HISTORY:,:,k]>=.1)&(a['projection_area'][HISTORY:,:,k]>=144)
        vb=(b['visible_fraction'][HISTORY:,:,k]>=.1)&(b['projection_area'][HISTORY:,:,k]>=144)
        if not (va&vb).any():
            raise ValueError('Camera pair has no jointly visible readout support')
    for camera in range(2):
        a,b = data[0,camera],data[1,camera]
        for field in ('K', 'camera_to_world', 'time_s', 'present', 'class_id', 'instance_id', 'world_from_factory'):
            equal(a[field],b[field], 'world intervention/' + field)
        others = np.arange(a['world_state'].shape[1]) != k
        equal(a['world_state'][:,others],b['world_state'][:,others], 'nonintervened actors')
        if np.array_equal(a['world_state'][HISTORY:,k],b['world_state'][HISTORY:,k]):
            raise ValueError('World intervention has no distinct target future')
        if first['cohort'] == 'ambiguous':
            _same_history(a,b)
            equal(a['world_state'][:HISTORY],b['world_state'][:HISTORY], 'ambiguous history dynamics')
        else:
            receiver, partner = first['receiver'],1-first['receiver']
            _same_history(a,b,receiver)
            changed = np.any(a['world_state'][:HISTORY,k] != b['world_state'][:HISTORY,k], axis=-1)
            if not changed.any() or not changed[-1]:
                raise ValueError('Evidence intervention must already differ DURING history')
            if np.array_equal(a['rgb'][:HISTORY,partner],b['rgb'][:HISTORY,partner]):
                raise ValueError('No changed partner RGB evidence')
            common_seen = (a['visible_fraction'][:HISTORY,receiver,k] >= .1) & (a['projection_area'][:HISTORY,receiver,k] >= 144) & ~changed
            if not common_seen.any():
                raise ValueError('Target must be established in receiver history before divergence')
            for item in (a,b):
                if (item['visible_fraction'][:HISTORY,receiver,k][changed] != 0).any():
                    raise ValueError('Receiver must be fully occluded during changed historical motion')
                visible = (item['visible_fraction'][:HISTORY,partner,k] >= .1) & (item['projection_area'][:HISTORY,partner,k] >= 144)
                if not (visible & changed).any():
                    raise ValueError('Partner does not see intervened historical motion')
                if camera==0:
                    future=(item['visible_fraction'][HISTORY:,receiver,k]>=.1)&(item['projection_area'][HISTORY:,receiver,k]>=144)
                    if not any(future[h:h+3].all() for h in range(len(future)-2)):
                        raise ValueError('Canonical camera lacks a three-frame target reappearance')
    return dict(family_id=first['family_id'], passed=True, history_sha256=hashes,
                scope='pixel/calibration/state controls; not a model-performance result')


def audit_manifest(path):
    path = Path(path)
    records = json.loads(path.read_text())['records']
    groups, splits, paths = {}, {}, set()
    for item in records:
        if item['path'] in paths:
            raise ValueError('Duplicate branch file')
        paths.add(item['path'])
        if item['split'] not in ('train','val','test'):
            raise ValueError('Invalid split')
        # scene_parent includes all near variants, not merely an episode number.
        for kind in ('scene_parent_id','family_id'):
            key = (kind,item[kind])
            if key in splits and splits[key] != item['split']:
                raise ValueError('Train/test lineage leakage: ' + str(key))
            splits[key] = item['split']
        groups.setdefault(item['family_id'],[]).append(item)
    if not groups:
        raise ValueError('Empty paired dataset')
    results=[]; histories={}
    for group in groups.values():
        result=audit_family(group,path.parent)
        for signatures in result['history_sha256'].values():
            signature=tuple(sorted(signatures.items()))
            if signature in histories and histories[signature]!=group[0]['split']:
                raise ValueError('Identical complete history appears across splits')
            histories[signature]=group[0]['split']
        results.append(result)
    return dict(passed=True, families=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = audit_manifest(args.manifest)
    encoded = json.dumps(result,indent=2)
    if args.output:
        Path(args.output).write_text(encoded+'\n')
    print(encoded)


if __name__ == '__main__':
    main()
