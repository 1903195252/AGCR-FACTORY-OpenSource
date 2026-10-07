#!/usr/bin/env python3
"""Critical contract checks without rendering or GPUs."""
import json,sys
from pathlib import Path
import numpy as np
from collections import Counter
from paper_dataset.planning import plans,episode_spec,actor_state,camera_poses
from paper_dataset.events import audit_event,classify,frustum_presence
from paper_dataset.common import load_config
c=load_config(Path(__file__).parents[1]/'configs/dataset_paper_rgbd.json');jobs=plans(c)
assert len(jobs)==2550 and len({j['name'] for j in jobs})==2550
for split in ['train','val','test']:
    group=[j for j in jobs if j['pool']=='main' and j['split']==split]
    counts=Counter(j['event'] for j in group);assert max(counts.values())-min(counts.values())<=2
    assert len({j['focal_class'] for j in group})==4
    assert len({j['occluder'] for j in group})==5
for p in jobs[:80:2]:
    s=episode_spec(p,c);r=episode_spec(jobs[p['index']+1],c)
    for t in [0,5,6,7,10,12.4,15]:
        C,E,X=camera_poses(s,t)[0];assert np.allclose(C,E@X)
        assert np.allclose(np.linalg.det(C[:3,:3]),1)
        pos,v,_,_=actor_state(s['actors'][0],t,s['duration'])
        eps=1e-5;dv=(actor_state(s['actors'][0],t+eps,s['duration'])[0]-actor_state(s['actors'][0],t-eps,s['duration'])[0])/(2*eps)
        assert np.allclose(v,dv,atol=1e-4),(p['motion'],t,v,dv)
        assert np.allclose(camera_poses(s,t)[0][0],camera_poses(r,t)[0][0])
# Correct pair alone does not pass an occlusion event when outside the image.
s=episode_spec(jobs[6],c);areas=np.zeros((100,2,1),int);fraction=np.zeros_like(areas,float)
assert not audit_event(classify(fraction,areas,c),areas,s,c)['passed']
assert frustum_presence([0,100,140,150,0]).tolist()==[False,True,True,True,False]
print(json.dumps({'passed':True,'planned_episodes':len(jobs),'checks':['quota_balance','factor_coverage','unique_ids','pose_chain','analytic_velocity','matched_camera_paths','frustum_vs_occlusion']}))
