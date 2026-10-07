"""Labels remain separate from admissible RGB-D observations at every boundary."""
import json
from pathlib import Path
import hashlib
import h5py
import numpy as np
import torch
from PIL import Image

CLASS_MAP={3:1,4:2,13:3,14:4}
ROLES={'annotation':'annotation_train','video_evaluation':'evaluation_reader_train','downstream':'train'}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def native_clip(root,episode,start,frames=16):
    d=Path(root)/'episodes'/episode;stop=start+frames
    with h5py.File(d/'frames.h5','r') as h:
        rgb=np.stack([[np.asarray(Image.open(d/'rgb'/('%s_%04d.jpg'%(ch,t)))) for ch in ('CAM_AIR','CAM_GROUND')] for t in range(start,stop)])
        inputs=dict(rgb=rgb,depth_mm=h['depth_mm'][start:stop],valid=h['valid'][start:stop].astype(bool),
            K=h['K'][:],camera_to_world=h['camera_to_world'][start:stop],time_s=h['time_s'][start:stop])
        labels=dict(instance=h['instance'][start:stop],state=h['state'][start:stop],
            class_id=np.array([CLASS_MAP[int(c)] for c in h['class_id'][:]],np.int64),track_id=np.arange(1,len(h['class_id'])+1))
    return inputs,labels


def camera_K(K,t,v):
    if K.ndim==2:return K
    if K.ndim==3:return K[v]
    return K[t,v]


def make_targets(inputs,labels,episode,frames=None):
    rgb=inputs['rgb'];H,W=rgb.shape[-3:-1];targets=[]
    frames=list(range(len(rgb))) if frames is None else frames
    for t in frames:
        for v in range(2):
            K=camera_K(inputs['K'],t,v);T=inputs['camera_to_world'][t,v];R=T[:3,:3];origin=T[:3,3]
            boxes=[];masks=[];contacts=[];classes=[];centers=[];dims=[];headings=[];identities=[]
            for j,track in enumerate(labels['track_id']):
                mask=labels['instance'][t,v]==track
                yy,xx=np.where(mask)
                # Detection supervision uses actual visible pixels, never hidden projections.
                if len(xx)==0:continue
                x0,x1=float(xx.min()),float(xx.max()+1);y0,y1=float(yy.min()),float(yy.max()+1)
                boxes.append([(x0+x1)/2/W,(y0+y1)/2/H,(x1-x0)/W,(y1-y0)/H]);masks.append(mask)
                s=labels['state'][t,j];center=R.T@(s[:3]-origin)
                contact=s[:3].copy();contact[2]-=s[10]/2;point=R.T@(contact-origin)
                pixel=K@point;pixel=pixel[:2]/max(pixel[2],.01)/np.array([W,H])
                contacts.append(pixel);classes.append(int(labels['class_id'][j])-1);centers.append(center)
                dims.append(s[8:11]);direction=R.T@np.array([np.cos(s[6]),np.sin(s[6]),0.]);headings.append(direction)
                identities.append(episode+':'+str(int(track)))
            n=len(boxes)
            target=dict(boxes=np.asarray(boxes,np.float32).reshape(n,4),class_labels=np.asarray(classes,np.int64),
                masks=np.asarray(masks,bool).reshape(n,H,W),contact=np.asarray(contacts,np.float32).reshape(n,2),
                center_camera_m=np.asarray(centers,np.float32).reshape(n,3),dimensions_m=np.asarray(dims,np.float32).reshape(n,3),
                heading_camera=np.asarray(headings,np.float32).reshape(n,3),identity_keys=identities)
            targets.append({k:torch.from_numpy(a) if isinstance(a,np.ndarray) else a for k,a in target.items()})
    return targets


def pack_images(inputs,frames=None):
    frames=list(range(len(inputs['rgb']))) if frames is None else frames
    rgb=inputs['rgb'][frames].reshape(-1,*inputs['rgb'].shape[2:]).transpose(0,3,1,2)
    return dict(images=torch.from_numpy(np.ascontiguousarray(rgb)),
        depth_mm=torch.from_numpy(inputs['depth_mm'][frames].reshape(-1,*inputs['depth_mm'].shape[2:]).astype(np.float32)),
        valid=torch.from_numpy(inputs['valid'][frames].reshape(-1,*inputs['valid'].shape[2:]).astype(bool)),
        K=torch.from_numpy(np.stack([camera_K(inputs['K'],t,v) for t in frames for v in range(2)]).astype(np.float32)))


def move(value,device):
    if isinstance(value,torch.Tensor):return value.to(device,non_blocking=True)
    if isinstance(value,list):return [move(v,device) for v in value]
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    return value


class ClipDataset:
    def __init__(self,index,role='downstream',cohort=None,modality='rgbd'):
        self.index=json.loads(Path(index).read_text());self.root=Path(self.index['dataset_root']);self.role=role;self.modality=modality
        if role not in ROLES:raise ValueError('Unknown independent learner role')
        if cohort:
            if role!='downstream':raise ValueError('Readers must train on their own rendered episode subset')
            self.cohort=json.loads(Path(cohort).read_text());self.rows=self.cohort['clips']
            if self.cohort['split_sha256']!=self.index['split_sha256']:raise ValueError('Cohort split mismatch')
            for row in self.rows:
                if row['source'] not in ('generated','additional_rendered') and self.index['roles'].get(row['episode'])!='train':
                    raise ValueError('Downstream training episode leakage')
                if row['source']=='generated' and (row.get('role')!='train' or row.get('acceptance_schema') not in ('agcr-acceptance-2','agcr-acceptance-3-rgb')):
                    raise ValueError('Unaccepted or non-training generated clip')
                if row['source']=='generated' and self.index['roles'].get(row['episode'])!='train':
                    raise ValueError('Generated proposal episode is outside training split')
                if row['source']=='additional_rendered' and self.index['roles'].get(row['episode'])!='augmentation_train':
                    raise ValueError('Additional rendered cohort is outside independent augmentation split')
                if row.get('acceptance_schema')=='agcr-acceptance-3-rgb' and modality!='rgb':
                    raise ValueError('Trajectory-free RGB labels cannot supervise RGB-D box heads')
        else:
            self.cohort=None
            self.rows=[r for r in self.index['windows'] if r['role']==ROLES[role] and r['start'] in (0,25,50,75) and r['history_covered']]
        if not self.rows:raise ValueError('No eligible training clips')

    def __len__(self):return len(self.rows)

    def __getitem__(self,i):
        row=self.rows[i]
        if row.get('source')=='generated':
            source=Path(row['rgbd_path']);annotation=Path(row['annotation_path'])
            if sha256(annotation)!=row['annotation_sha256']:raise ValueError('Changed accepted annotations')
            with np.load(source,allow_pickle=False) as h:
                inputs={k:h[k] for k in ('rgb','K','camera_to_world')}
                if 'depth_mm' in h:
                    inputs['depth_mm']=h['depth_mm'];inputs['valid']=h['valid'] if 'valid' in h else h['validity_probability']>=.5
                elif self.modality=='rgb':
                    inputs['depth_mm']=np.zeros(inputs['rgb'].shape[:-1],np.float32);inputs['valid']=np.zeros(inputs['rgb'].shape[:-1],bool)
                else:raise ValueError('An RGB-only generator cannot be used for RGB-D training')
                inputs['time_s']=h['time_s'] if 'time_s' in h else np.arange(1,17,dtype=np.float32)/5
            with np.load(annotation,allow_pickle=False) as h:
                labels={k:h[k] for k in ('instance','state','class_id','track_id')}
            if len(inputs['rgb'])!=16:raise ValueError('Generated training clips must contain only 16 future frames')
        else:inputs,labels=native_clip(self.root,row['episode'],row['start']+6)
        return inputs,make_targets(inputs,labels,row['episode'])
