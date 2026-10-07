"""Validate exported tables, masks and transforms with the official nuScenes SDK.

Run in an isolated environment with nuscenes-devkit==1.2.0 installed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataroot',type=Path,required=True)
p.add_argument('--full-masks',action='store_true',help='Read every RGB and label file instead of a deterministic subset')
a=p.parse_args();root=a.dataroot
config=json.loads((root/'capture_config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
assert manifest['status']=='complete'
catalog=json.loads((root/'extensions/object_catalog.json').read_text())
colors={c['name']:tuple(c['color']) for c in catalog['categories']}
nusc=NuScenes(version=config['version'],dataroot=str(root),verbose=False,colormap=colors)
expected=config['scenes']*config['samples_per_scene']
assert len(nusc.sample)==expected and len(nusc.sample_data)==5*expected
assert len(nusc.scene)==config['scenes']
for name in nusc.table_names:
    rows=getattr(nusc,name);assert len({r['token'] for r in rows})==len(rows),name
for sc in nusc.scene:
    current=sc['first_sample_token'];visited=[]
    while current:
        r=nusc.get('sample',current);assert r['scene_token']==sc['token'] and current not in visited
        if visited:assert r['prev']==visited[-1]
        else:assert r['prev']==''
        visited.append(current);current=r['next']
    assert len(visited)==sc['nbr_samples'] and visited[-1]==sc['last_sample_token']
for table,key in [('sample_data','calibrated_sensor_token'),('sample_annotation','instance_token')]:
    for r in getattr(nusc,table):
        for direction,reverse in [('next','prev'),('prev','next')]:
            if r[direction]:
                other=nusc.get(table,r[direction]);assert other[reverse]==r['token'] and other[key]==r[key]
                sr=nusc.get('sample',r['sample_token']);so=nusc.get('sample',other['sample_token'])
                assert sr['scene_token']==so['scene_token']
                assert (so['timestamp']-sr['timestamp'])*(1 if direction=='next' else -1)>0
for sample in nusc.sample:
    assert set(sample['data'])==set(manifest['channels'])
    for sd_token in sample['data'].values():
        sd=nusc.get('sample_data',sd_token)
        assert sd['timestamp']==sample['timestamp']
        ego=nusc.get('ego_pose',sd['ego_pose_token']);assert ego['timestamp']==sample['timestamp']
        assert (root/sd['filename']).is_file()
for ann in nusc.sample_annotation:
    assert np.isfinite(ann['translation']).all() and np.min(ann['size'])>0
    assert abs(np.linalg.norm(ann['rotation'])-1)<1e-8
    assert ann['visibility_token']=='' and ann['num_lidar_pts']==ann['num_radar_pts']==0
counts=Counter(r['instance_token'] for r in nusc.sample_annotation)
for inst in nusc.instance:
    assert counts[inst['token']]==inst['nbr_annotations']
    assert nusc.get('sample_annotation',inst['first_annotation_token'])['prev']==''
    assert nusc.get('sample_annotation',inst['last_annotation_token'])['next']==''
splits=json.loads((root/'splits.json').read_text());flat=sum(splits.values(),[])
assert len(flat)==len(set(flat))==len(nusc.scene)
assert set(flat)=={s['name'] for s in nusc.scene}
map_mask=nusc.map[0]['mask']
assert bool(map_mask.is_on_mask(205.,155.)[0])
assert not bool(map_mask.is_on_mask(0.,0.)[0])
assert not bool(map_mask.is_on_mask(409.,155.)[0])
by_id={o['id']:o for o in catalog['objects']};verified=0;projection_error=0.;pose_error=0.;containment_checks=0
instance_classes=np.full(max(by_id)+1,65535,np.uint16)
for oid,obj in by_id.items():instance_classes[oid]=obj['category_id']
for scene_index,scene in enumerate(nusc.scene):
    entries=json.loads((root/'extensions'/f'{scene["name"]}.json').read_text())
    assert len(entries)==5*scene['nbr_samples']
    for i,e in enumerate(entries):
        sd=nusc.get('sample_data',e['sample_data_token']);cs=nusc.get('calibrated_sensor',sd['calibrated_sensor_token']);ego=nusc.get('ego_pose',sd['ego_pose_token'])
        r=Quaternion(ego['rotation']).rotation_matrix@Quaternion(cs['rotation']).rotation_matrix
        t=np.asarray(ego['translation'])+Quaternion(ego['rotation']).rotation_matrix@cs['translation']
        pose_error=max(pose_error,float(np.max(np.abs(r-Quaternion(e['camera_to_global_rotation']).rotation_matrix))),float(np.max(np.abs(t-e['camera_to_global_translation']))))
        assert pose_error<1e-9
        assert (root/e['semantic_filename']).is_file() and (root/e['instance_filename']).is_file()
        if not a.full_masks and i not in list(range(5))+list(range(len(entries)//2,len(entries)//2+5))+list(range(len(entries)-5,len(entries))):continue
        with Image.open(root/sd['filename']) as image:assert image.size==(config['width'],config['height']);image.load()
        with Image.open(root/e['semantic_filename']) as image:sem=np.asarray(image)
        with Image.open(root/e['instance_filename']) as image:ins=np.asarray(image)
        assert sem.shape==ins.shape==(config['height'],config['width'])
        assert sem.min()>=0 and sem.max()<len(catalog['categories'])
        assert ins.min()>=0 and ins.max()<len(instance_classes)
        assert np.all((ins==0)|(sem==instance_classes[ins]))
        for obj in e['objects']:
            ann=nusc.get('sample_annotation',obj['annotation_token'])
            assert ann['instance_token']==obj['instance_token'] and ann['sample_token']==sd['sample_token']
            yy,xx=np.nonzero(ins==obj['instance_id'])
            assert len(xx)==obj['visible_pixels'] and len(xx)>=config['min_instance_pixels']
            assert obj['bbox_xyxy']==[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
            _,sdk_boxes,K=nusc.get_sample_data(sd['token'],box_vis_level=BoxVisibility.NONE,selected_anntokens=[ann['token']])
            sdk=sdk_boxes[0].corners()
            world=nusc.get_box(ann['token']).corners()
            manual=r.T@(world-t[:,None]);projection_error=max(projection_error,float(np.max(np.abs(sdk-manual))))
            if np.min(sdk[2])>.05:
                uv=K@sdk;uv=uv[:2]/uv[2]
                # Pixel rasterization can cover half a pixel beyond projected bounds.
                assert xx.min()>=uv[0].min()-1.5 and xx.max()<=uv[0].max()+1.5,(sd['token'],obj,uv)
                assert yy.min()>=uv[1].min()-1.5 and yy.max()<=uv[1].max()+1.5,(sd['token'],obj,uv)
                containment_checks+=1
        verified+=1
    if scene_index%5==0:print(f'Validated {scene_index+1}/{len(nusc.scene)} scenes, {verified} mask pairs',flush=True)
assert projection_error<1e-8
report=dict(passed=True,official_devkit='nuscenes-devkit 1.2.0',samples=len(nusc.sample),images=len(nusc.sample_data),
    annotations=len(nusc.sample_annotation),instances=len(nusc.instance),scenes=len(nusc.scene),
    exact_timestamp_sync=True,foreign_keys_and_chains=True,scene_disjoint_splits=True,map_mask_coordinates=True,
    checked_rgb_semantic_instance_triplets=verified,full_masks=a.full_masks,
    camera_pose_max_abs_error=pose_error,sdk_box_transform_max_abs_error_m=projection_error,
    projected_box_mask_containment_checks=containment_checks)
(root/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
