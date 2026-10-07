"""nuScenes camera tables plus explicit paired RGB-D / visibility extensions."""
import json
from pathlib import Path
from collections import Counter
import h5py
import numpy as np
from .common import token, quaternion, atomic_json, CATEGORIES, episode_identity

TABLES=['category','attribute','visibility','sensor','calibrated_sensor','ego_pose','log','map',
        'scene','sample','sample_data','sample_annotation','instance']

def export(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());c=manifest['config']
    tables={k:[] for k in TABLES};extensions=[];windows=[];counts=Counter();cross=Counter();inspection_counts=Counter()
    for i,(name,desc) in enumerate(CATEGORIES):
        tables['category'].append(dict(token=token('category',i),name=name,description=desc,index=i))
    for i,name in enumerate(['moving','stationary']):
        tables['attribute'].append(dict(token=token('attribute',name),name='motion.'+name,description=name))
    for i,desc in enumerate(['0--40%','40--60%','60--80%','80--100%'],1):
        tables['visibility'].append(dict(token=str(i),level='v'+str(i),description=desc))
    for ch in c['camera_channels']:tables['sensor'].append(dict(token=token('sensor',ch),channel=ch,modality='camera'))
    mapfile=root/'maps'/'factory.png';mapfile.parent.mkdir(exist_ok=True)
    # The required map table stores a semantic-free map placeholder; actual floor
    # polygons and geometry are layouts.json, not a fictitious georeferenced map.
    from PIL import Image
    Image.new('L',(1,1),0).save(mapfile)
    layout_ids=sorted({p['layout'] for p in json.loads((root/'plans.json').read_text())})
    tables['map']=[dict(token=token('map'),category='semantic_prior',filename='maps/factory.png',
                       log_tokens=[token('log',i) for i in layout_ids])]
    for i in layout_ids:tables['log'].append(dict(token=token('log',i),logfile='layout_%02d'%i,
        vehicle='air-ground-camera-pair',date_captured='2026-09-30',location='synthetic_factory_layout_%02d'%i))
    for marker in sorted((root/'episodes').glob('*/complete.json')):
        d=marker.parent;meta=json.loads((d/'episode.json').read_text());name=meta['name'];T=meta['frames'];N=meta['targets']
        if meta['identity']!=episode_identity(manifest,name):raise ValueError('Unregistered episode identity')
        if not meta['metrics']['script_event']['passed']:raise ValueError('Unqualified event in completed episodes')
        events=json.loads((d/'events.json').read_text()) if (d/'events.json').exists() else {}
        for kind,rows in events.get('inspection',{}).items():
            for row in rows:
                inspection_counts[(meta['split'],meta['pool'],kind,str(row.get('view','both')))]+=1
        scene=token('scene',name);samples=[token('sample',name,t) for t in range(T)]
        tables['scene'].append(dict(token=scene,name=name,description=meta['event']+' / '+meta['occluder'],
            log_token=token('log',meta['layout']),nbr_samples=T,first_sample_token=samples[0],last_sample_token=samples[-1]))
        with h5py.File(d/'frames.h5','r') as h:
            K=h['K'][:];timestamps=h['timestamp_us'][:];poses=h['ego_to_world'][:]
            X=h['camera_to_ego'][0];states=h['state'][:];vis=h['visible_fraction'][:];classes=h['class_id'][:]
            for v,ch in enumerate(c['camera_channels']):
                tables['calibrated_sensor'].append(dict(token=token('calib',name,ch),sensor_token=token('sensor',ch),
                    translation=X[v,:3,3].tolist(),rotation=quaternion(X[v,:3,:3]),camera_intrinsic=K.tolist()))
            for k in range(N):
                tables['instance'].append(dict(token=token('instance',name,k),category_token=token('category',int(classes[k])),
                    nbr_annotations=T,first_annotation_token=token('annotation',name,0,k),last_annotation_token=token('annotation',name,T-1,k)))
            for t in range(T):
                tables['sample'].append(dict(token=samples[t],timestamp=int(timestamps[t]),scene_token=scene,
                    prev=samples[t-1] if t else '',next=samples[t+1] if t+1<T else ''))
                for v,ch in enumerate(c['camera_channels']):
                    ep=token('ego',name,t,ch);sd=token('sample_data',name,t,ch);E=poses[t,v]
                    tables['ego_pose'].append(dict(token=ep,timestamp=int(timestamps[t]),translation=E[:3,3].tolist(),rotation=quaternion(E[:3,:3])))
                    tables['sample_data'].append(dict(token=sd,sample_token=samples[t],ego_pose_token=ep,
                        calibrated_sensor_token=token('calib',name,ch),timestamp=int(timestamps[t]),is_key_frame=True,
                        height=c['height'],width=c['width'],fileformat='jpg',
                        filename=str((d/'rgb'/('%s_%04d.jpg'%(ch,t))).relative_to(root)),
                        prev=token('sample_data',name,t-1,ch) if t else '',next=token('sample_data',name,t+1,ch) if t+1<T else ''))
                for k in range(N):
                    s=states[t,k];fraction=float(vis[t,:,k].max());bucket=1+int(fraction>.4)+int(fraction>.6)+int(fraction>.8)
                    tables['sample_annotation'].append(dict(token=token('annotation',name,t,k),sample_token=samples[t],
                        instance_token=token('instance',name,k),visibility_token=str(bucket),
                        attribute_tokens=[token('attribute','moving' if np.linalg.norm(s[3:6])>.05 else 'stationary')],
                        translation=s[:3].tolist(),size=s[[9,8,10]].tolist(),rotation=[float(np.cos(s[6]/2)),0.,0.,float(np.sin(s[6]/2))],
                        num_lidar_pts=0,num_radar_pts=0,prev=token('annotation',name,t-1,k) if t else '',
                        next=token('annotation',name,t+1,k) if t+1<T else ''))
            # The principal benchmark uses event-centred windows. Context windows
            # remain available but never count as extra events or independent scenes.
            starts=range(T-c['history_frames']-c['future_frames']+1) if meta['split']=='train' else c['evaluation_window_starts']
            for start in starts:
                a=start+c['history_frames']-1;end=a+c['future_frames']
                event_window=(a<=meta['spec']['before_frames'][-1] and end>=meta['spec']['after_frames'][-1]
                              and a>=meta['spec']['before_frames'][0])
                windows.append(dict(episode=name,start=start,anchor=a,split=meta['split'],pool=meta['pool'],
                    layout=meta['layout'],event=meta['event'],event_window=event_window,matched_camera_path='l%02d_p%03d'%(meta['layout'],meta['path'])))
        extensions.append(dict(scene_token=scene,episode=name,hdf5=str((d/'frames.h5').relative_to(root)),
            metadata=str((d/'episode.json').relative_to(root)),split=meta['split'],pool=meta['pool'],
            event=meta['event'],focal_instance_token=token('instance',name,0),
            frames=list(range(T)),channels=c['camera_channels'],world_unit='m',depth_unit='mm'))
        counts[(meta['split'],meta['pool'],meta['event'])]+=1
        cross[(meta['split'],meta['pool'],meta['event'],meta['occluder'],meta['focal_class'],meta['motion'])]+=1
    destination=root/c['nuscenes_version']
    for name,rows in tables.items():atomic_json(destination/(name+'.json'),rows)
    atomic_json(root/'rgbd_episodes.json',extensions);atomic_json(root/'windows.json',windows)
    splits={}
    for e in extensions:
        key=e['split'] if e['pool']=='main' else e['pool']+'_'+e['split']
        splits.setdefault(key,[]).append(e['episode'])
    atomic_json(root/'splits.json',splits)
    planned=Counter((p['split'],p['pool'],p['event']) for p in json.loads((root/'plans.json').read_text()))
    report=dict(completed_episodes=len(extensions),planned_episodes=manifest['planned_episodes'],
        quotas=[dict(split=k[0],pool=k[1],event=k[2],planned=v,accepted=counts[k],missing=v-counts[k]) for k,v in planned.items()],
        coverage=[dict(split=k[0],pool=k[1],event=k[2],occluder=k[3],target=k[4],motion=k[5],accepted=v) for k,v in cross.items()],
        inspection_coverage=[dict(split=k[0],pool=k[1],kind=k[2],view=k[3],count=v) for k,v in inspection_counts.items()],
        complete=len(extensions)==manifest['planned_episodes'])
    atomic_json(root/'coverage.json',report)
    return report
