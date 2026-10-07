"""Read the factory dataset using the official SDK (no MuJoCo dependency).

Run with the isolated nuscenes-devkit environment, after generation completes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataroot',type=Path,required=True)
p.add_argument('--sample',type=int,default=0)
p.add_argument('--channel',default='CAM_FRONT')
a=p.parse_args();root=a.dataroot
config=json.loads((root/'capture_config.json').read_text())
catalog=json.loads((root/'extensions/object_catalog.json').read_text())
colors={c['name']:tuple(c['color']) for c in catalog['categories']}
nusc=NuScenes(version=config['version'],dataroot=str(root),colormap=colors,verbose=False)
sample=nusc.sample[a.sample];scene=nusc.get('scene',sample['scene_token'])
sd_token=sample['data'][a.channel]
path,boxes,K=nusc.get_sample_data(sd_token)
meta=next(e for e in json.loads((root/'extensions'/f'{scene["name"]}.json').read_text()) if e['sample_data_token']==sd_token)
with Image.open(path) as im:rgb=np.asarray(im)
with Image.open(root/meta['semantic_filename']) as im:semantic=np.asarray(im)
with Image.open(root/meta['instance_filename']) as im:instance=np.asarray(im)
print(json.dumps(dict(scene=scene['name'],sample_token=sample['token'],timestamp_us=sample['timestamp'],
    channel=a.channel,rgb_shape=list(rgb.shape),semantic_dtype=str(semantic.dtype),
    visible_classes=[catalog['categories'][int(i)]['name'] for i in np.unique(semantic)],
    instance_ids=np.unique(instance).tolist(),sdk_camera_boxes=len(boxes),camera_intrinsic=K.tolist(),
    camera_to_global_translation=meta['camera_to_global_translation'],camera_to_global_rotation=meta['camera_to_global_rotation']),indent=2))
