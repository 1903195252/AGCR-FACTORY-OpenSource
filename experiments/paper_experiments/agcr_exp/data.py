"""Frozen split/index creation and leakage-separated, egocentric RGB-D windows."""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from .protocol import DEFAULT,digest

CLASS_MAP={3:1,4:2,13:3,14:4} # zero is background; full class order is fixed.

def read_json(path):return json.loads(Path(path).read_text())

def plan_role(plan, roles):
    # Supplement paths are new training groups, never re-rank the original
    # main-pool groups and consequently never move an ID-test/reader path.
    if plan['pool']=='training_supplement':
        if plan['split']!='train':raise ValueError('Training supplement outside train layouts')
        return 'train'
    if plan['pool']=='main':
        return roles[(plan['layout'],plan['path'])] if plan['split']=='train' else plan['split']
    return plan['pool']+'_'+plan['split']
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def assignments(root,heldout_class=None):
    plans=read_json(Path(root)/'plans.json');by_layout=defaultdict(set)
    for p in plans:
        if p['pool']=='main' and p['split']=='train':by_layout[p['layout']].add(p['path'])
    role={}
    policy=DEFAULT['split_policy']
    id_end=policy['id_groups_per_layout']
    annotation_end=id_end+policy['annotation_groups_per_layout']
    reader_end=annotation_end+policy['evaluation_reader_groups_per_layout']
    for layout,paths in by_layout.items():
        ordered=sorted(paths,key=lambda p:digest(['split-v1',layout,p]))
        for rank,path in enumerate(ordered):
            role[(layout,path)]=('id_test' if rank<id_end else 'annotation_train' if rank<annotation_end else 'evaluation_reader_train' if rank<reader_end else 'train')
    return role

def build_index(root,output,allow_partial=False,heldout_class=None):
    root=Path(root);manifest=read_json(root/'manifest.json');plans=read_json(root/'plans.json')
    markers=sorted((root/'episodes').glob('*/complete.json'))
    if not allow_partial and len(markers)!=len(plans):raise ValueError('Collection incomplete: %d/%d'%(len(markers),len(plans)))
    if manifest['purpose']!='formal' and not allow_partial:raise ValueError('Pilot data cannot form a formal split')
    if manifest.get('revision') and not allow_partial:
        validation=read_json(root/'validation.json') if (root/'validation.json').exists() else {}
        if not validation.get('passed') or set(validation.get('episodes',[]))!={p['name'] for p in plans}:
            raise ValueError('Revised collection has not passed final validation for the current plans')
    roles=assignments(root)
    planned_capacity=Counter()
    for plan in plans:
        planned_role=plan_role(plan,roles)
        planned_capacity[planned_role]+=len(DEFAULT['evaluation_starts'])
    rows=[];counts=Counter();episode_roles={};class_sets={};coordinate_frames=set()
    for marker in markers:
        meta=read_json(marker.parent/'episode.json');name=meta['name']
        expected_identity=manifest.get('inherited_episode_identities',{}).get(name,manifest['identity'])
        if meta['identity']!=expected_identity:raise ValueError('Unregistered capture identity')
        if not meta['metrics']['script_event']['passed']:raise ValueError('Unqualified script event')
        role=plan_role(meta,roles)
        with h5py.File(marker.parent/'frames.h5','r') as h:
            fraction=h['visible_fraction'][:];area=h['full_area'][:];visible_area=h['visible_area'][:]
            b=(fraction>=DEFAULT['v_min'])&(area>=DEFAULT['area_min']);classes=h['class_id'][:]
            has_new=heldout_class is not None and heldout_class in classes
            if has_new:
                if role in ('train','annotation_train','evaluation_reader_train','augmentation_train'):continue
                if role=='id_test':role='ood_class'
                elif role=='test':role='ood_layout_class'
                elif role=='val':role='val_heldout_class' # never used for seen-class model selection.
            episode_roles[name]=role;class_sets[name]=classes.tolist()
            T=len(h['time_s']);starts=range(T-21) if role in ('train','annotation_train','evaluation_reader_train') else DEFAULT['evaluation_starts']
            for start in starts:
                anchor=start+5;stop=anchor+17;eligible=b[start:anchor+1].any(axis=(0,1))
                # Any visible future moving object must be history-covered. Dynamic
                # occluders with pixels in front of evaluated targets are visible too.
                future_visible=(visible_area[anchor+1:stop]>0).any(axis=(0,1))
                history_covered=not np.any(future_visible & ~eligible)
                pair=(2*b[anchor,0]+b[anchor,1]).astype(int)
                uid=name+':%03d'%start
                row=dict(uid=uid,episode=name,start=start,anchor=anchor,role=role,layout=meta['layout'],
                    camera_group='l%02d_p%03d'%(meta['layout'],meta['path']),repeat=meta['repeat'],
                    pool=meta['pool'],script_event=meta['event'],anchor_pair=pair.tolist(),
                    history_eligible=eligible.tolist(),history_covered=bool(history_covered),
                    focal_history_eligible=bool(eligible[0]),
                    event_window=anchor in (29,30),target_classes=classes.tolist(),focal_class_id=int(classes[0]))
                rows.append(row);counts[(role,'all')]+=1;counts[(role,'history_covered')]+=int(history_covered)
    policy=dict(version='layout-and-camera-group-v2',dataset_identity=manifest['identity'],heldout_class=heldout_class,
                roles=episode_roles,classes=class_sets,allow_partial=allow_partial,
                split_policy=DEFAULT['split_policy'],
                window_protocol={k:DEFAULT[k] for k in ['history','future','fps','evaluation_starts','v_min','area_min']},
                windows_sha256=digest(rows))
    index=dict(schema='agcr-window-index-1',dataset_root=str(root.resolve()),dataset_identity=manifest['identity'],
        split_sha256=digest(policy),development_only=allow_partial or manifest['purpose']!='formal',
        heldout_class=heldout_class,roles=episode_roles,windows=rows,
        planned_fixed_window_capacity_upper_bound=dict(planned_capacity),
        counts=[dict(role=k[0],cohort=k[1],windows=v) for k,v in counts.items()],
        limitations=['All reference targets remain physically present; disappearance/survival calibration is not covered.',
                    'No independent planned-versus-realized camera stream; physical trajectory error requires rerendered data.',
                    'Current camera trajectories are straight; smooth orbit generalization requires additional capture.'])
    atomic_json(output,index);return index

def reference_reason(area,visible_area,v_min=.1,area_min=144):
    # 0 visible; 1 full occlusion; 2 partial/subthreshold visibility; 3 too small; 4 out of image.
    fraction=visible_area/np.maximum(area,1)
    return np.where(area==0,4,np.where(area<area_min,3,np.where(visible_area==0,1,np.where(fraction<v_min,2,0)))).astype('u1')

def world_to_anchor(ground_pose):
    yaw=np.arctan2(ground_pose[1,0],ground_pose[0,0]);c,s=np.cos(yaw),np.sin(yaw)
    R=np.array([[c,s,0],[-s,c,0],[0,0,1.]])
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=-R@ground_pose[:3,3]
    return T,yaw

def transform_states(states,T,yaw):
    out=states.copy();out[...,:3]=states[...,:3]@T[:3,:3].T+T[:3,3]
    out[...,3:6]=states[...,3:6]@T[:3,:3].T
    out[...,6]=(states[...,6]-yaw+np.pi)%(2*np.pi)-np.pi
    return out

class Dataset:
    def __init__(self,index,role='train',primary=True,load_surfaces=False):
        self.index=read_json(index) if not isinstance(index,dict) else index
        self.root=Path(self.index['dataset_root']);self.load_surfaces=load_surfaces
        self.rows=[r for r in self.index['windows'] if r['role']==role and (r['history_covered'] or not primary)]
        if not self.rows:raise ValueError('No eligible windows for role='+role)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return read_window(self.root,self.rows[i],self.load_surfaces)

def read_window(root,row,load_surfaces=False):
    d=Path(root)/'episodes'/row['episode'];start=row['start'];a=start+6;end=a+16
    with h5py.File(d/'frames.h5','r') as h:
        transform,yaw=world_to_anchor(h['ego_to_world'][a-1,1])
        poses=transform@h['camera_to_world'][start:end]
        ego=transform@h['ego_to_world'][start:a]
        rgb=np.stack([np.stack([np.asarray(Image.open(d/'rgb'/('%s_%04d.jpg'%(ch,t))))
            for ch in ['CAM_AIR','CAM_GROUND']]) for t in range(start,end)])
        depth=h['depth_mm'][start:end];valid=h['valid'][start:end].astype(bool)
        time=h['time_s'][start:end]-h['time_s'][a-1]
        inputs=dict(rgb=rgb[:6],depth_mm=depth[:6],depth_valid=valid[:6],K=h['K'][:],
                    camera_to_world=poses[:6],ego_to_world=ego,planned_camera_to_world=poses[6:],
                    time_s=time[:6],future_time_s=time[6:],observation_available=np.ones((6,2),bool))
        states=transform_states(h['state'][start:end],transform,yaw)
        instance=h['instance'][start:end];n=states.shape[1]
        masks=instance[:,:,None]==np.arange(1,n+1)[None,None,:,None,None]
        area=h['full_area'][start:end];vis=h['visible_area'][start:end]
        labels=dict(rgb=rgb[6:],depth_mm=depth[6:],depth_valid=valid[6:],
            camera_to_world=poses[6:],state=states[6:],history_state=states[:6],
            present=np.ones((16,n),bool),history_present=np.ones((6,n),bool),
            class_id=np.array([CLASS_MAP[int(x)] for x in h['class_id'][:]]),
            visible_mask=masks[6:],history_visible_mask=masks[:6],visible_fraction=h['visible_fraction'][a:end],
            history_visible_fraction=h['visible_fraction'][start:a],full_area=area[6:],
            reason=reference_reason(area[6:],vis[6:]),
            history_eligible=np.array(row['history_eligible']),track_id=np.arange(1,n+1),
            static_depth_mm=h['static_depth_mm'][a:end])
        if load_surfaces:
            layers=[]
            for t in range(start,end):
                paired=[]
                for v in range(2):
                    surfaces=np.zeros((n,480,720),np.float32)
                    for k in range(n):
                        x0,y0,x1,y1=h['isolated/boxes'][t,v,k];off,count=h['isolated/offsets'][t,v,k]
                        if count:surfaces[k,y0:y1,x0:x1]=h['isolated/depth_mm'][off:off+count].reshape(y1-y0,x1-x0)
                    paired.append(surfaces)
                layers.append(paired)
            layers=np.array(layers)
            labels.update(isolated_depth_mm=layers[6:],history_full_mask=layers[:6]>0,full_mask=layers[6:]>0)
        return dict(inputs=inputs,labels=labels,metadata=dict(row,world_to_anchor=transform.tolist(),depth_unit='mm'))

def assert_no_future_leak(inputs):
    required={'rgb','depth_mm','depth_valid','K','camera_to_world','ego_to_world','planned_camera_to_world',
              'time_s','future_time_s','observation_available'}
    if set(inputs)!=required:raise ValueError('Unexpected inference input fields: '+str(set(inputs)^required))
    assert inputs['rgb'].shape[:2]==(6,2) and inputs['planned_camera_to_world'].shape==(16,2,4,4)
    assert np.all(inputs['time_s']<=0) and np.all(inputs['future_time_s']>0)


def read_observation_pair(root,row,history_offset):
    """Read only one synchronized HISTORY pair for observation-head pretraining."""
    if not 0<=history_offset<6:raise ValueError('Not a history frame')
    d=Path(root)/'episodes'/row['episode'];t=row['start']+history_offset;anchor=row['anchor']
    with h5py.File(d/'frames.h5','r') as h:
        transform,yaw=world_to_anchor(h['ego_to_world'][anchor,1]);state=transform_states(h['state'][t],transform,yaw)
        n=len(state);full=np.zeros((2,n,480,720),bool)
        for v in range(2):
            for k in range(n):
                x0,y0,x1,y1=h['isolated/boxes'][t,v,k];off,count=h['isolated/offsets'][t,v,k]
                if count:full[v,k,y0:y1,x0:x1]=h['isolated/depth_mm'][off:off+count].reshape(y1-y0,x1-x0)>0
        inst=h['instance'][t]
        masks=inst[:,None]==np.arange(1,n+1)[None,:,None,None]
        rgb=np.stack([np.asarray(Image.open(d/'rgb'/('%s_%04d.jpg'%(ch,t)))) for ch in ['CAM_AIR','CAM_GROUND']])
        return dict(inputs=dict(rgb=rgb[None],depth_mm=h['depth_mm'][t][None],depth_valid=h['valid'][t][None].astype(bool),
                                K=h['K'][:],camera_to_world=(transform@h['camera_to_world'][t])[None],
                                time_s=np.array([h['time_s'][t]-h['time_s'][anchor]])),
                    labels=dict(history_state=state[None],class_id=np.array([CLASS_MAP[int(x)] for x in h['class_id'][:]]),
                                history_full_mask=full[None],history_visible_mask=masks[None],
                                history_visible_fraction=h['visible_fraction'][t][None]))
