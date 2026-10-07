#!/usr/bin/env python3
"""Validate completed episode contracts and export nuScenes tables."""
import argparse,json
from pathlib import Path
import h5py
import numpy as np
from paper_dataset.common import file_hash,atomic_json,episode_identity
from paper_dataset.events import audit_event,classify
from paper_dataset.export import export
from paper_dataset.reader import read_window

def validate_tables(directory):
    names=['category','sensor','calibrated_sensor','ego_pose','scene','sample','sample_data','sample_annotation','instance','visibility']
    tables={t:json.loads((directory/(t+'.json')).read_text()) for t in names}
    index={t:{row['token']:row for row in rows} for t,rows in tables.items()}
    for t,rows in tables.items():assert len(rows)==len(index[t]),'Duplicate tokens: '+t
    references={'calibrated_sensor':{'sensor_token':'sensor'},'sample':{'scene_token':'scene'},
        'sample_data':{'sample_token':'sample','ego_pose_token':'ego_pose','calibrated_sensor_token':'calibrated_sensor'},
        'sample_annotation':{'sample_token':'sample','instance_token':'instance','visibility_token':'visibility'},
        'instance':{'category_token':'category'}}
    for table,refs in references.items():
        for row in tables[table]:
            for field,target in refs.items():assert row[field] in index[target],(table,field)
            for key,back in [('next','prev'),('prev','next')]:
                if row.get(key):assert index[table][row[key]][back]==row['token'],(table,key)
    return dict(passed=True,table_sizes={t:len(rows) for t,rows in tables.items()},official_sdk_tested=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--export',action='store_true')
    p.add_argument('--require-complete',action='store_true');p.add_argument('--full-hashes',action='store_true');a=p.parse_args();root=Path(a.root)
    manifest=json.loads((root/'manifest.json').read_text());c=manifest['config'];checked=[]
    layouts=json.loads((root/'layouts.json').read_text());assert len({l['geometry_sha256'] for l in layouts})==len(layouts)
    plans={p['name']:p for p in json.loads((root/'plans.json').read_text())}
    for marker in sorted((root/'episodes').glob('*/complete.json')):
        meta=json.loads((marker.parent/'episode.json').read_text());file=marker.parent/'frames.h5'
        assert meta['name'] in plans
        assert meta['identity']==episode_identity(manifest,meta['name'])
        for key in ('layout','path','repeat','split','pool','event','camera_seed','motion_seed'):
            assert meta[key]==plans[meta['name']][key],(meta['name'],key)
        if a.full_hashes:assert file_hash(file)==meta['frames_sha256']
        with h5py.File(file,'r') as h:
            assert h.attrs['identity']==meta['identity']
            T=meta['frames'];assert h['depth_mm'].shape==(T,2,c['height'],c['width'])
            assert np.allclose(h['ego_to_world'][:]@h['camera_to_ego'][:],h['camera_to_world'][:],atol=1e-8)
            assert np.allclose(np.diff(h['time_s']),1/c['fps'])
            assert np.allclose(h['camera_to_ego'][:],h['camera_to_ego'][0],atol=1e-8)
            assert audit_event(classify(h['visible_fraction'][:],h['full_area'][:],c),h['full_area'][:],meta['spec'],c)['passed']
            for t in range(T):
                depth=h['depth_mm'][t];valid=h['valid'][t].astype(bool)
                assert np.isfinite(depth).all() and np.all(depth[~valid]==0)
                assert np.all((depth[valid]>=c['depth_min_mm'])&(depth[valid]<=c['depth_max_mm']))
                for v in range(2):
                    image=marker.parent/'rgb'/('%s_%04d.jpg'%(c['camera_channels'][v],t))
                    assert image.exists()
                    if a.full_hashes:assert file_hash(image)==h['rgb_sha256'][t,v].decode()
            assert np.all(h['visible_area'][:]<=h['full_area'][:])
        window=read_window(root,meta['name'],25)
        assert window['inputs']['rgb'].shape[0]==6 and window['supervision']['rgb'].shape[0]==16
        assert 'state' not in window['inputs'] and 'instance' not in window['inputs']
        checked.append(meta['name'])
    report=dict(checked=len(checked),episodes=checked,passed=True,purpose=manifest['purpose'])
    if a.export:
        report['export']=export(root)
        report['nuScenes_relations']=validate_tables(root/c['nuscenes_version'])
    if a.require_complete:report['passed']=set(checked)==set(plans) and len(checked)==manifest['planned_episodes']
    atomic_json(root/'validation.json',report);print(json.dumps(report))
    if not report['passed']:raise SystemExit(2)
if __name__=='__main__':main()
