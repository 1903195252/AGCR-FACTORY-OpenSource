"""Independent image annotation, validation calibration and generated-data acceptance.

Candidate state/geometry is used only for checking image annotations. It is
never passed to the detector/tracker and never substituted for a missing readout.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from .data import sha256
from .inference import load_reader,IndependentReader,finite_position
from .train import atomic
from .runtime import ReaderRuntime
from experiments.paper_experiments.agcr_exp.downstream import fit_thresholds,accept_generated


def resize(array,H,W):
    x=torch.as_tensor(np.asarray(array).copy()).float()
    return F.interpolate(x.reshape(-1,1,*x.shape[-2:]),size=(H,W),mode='bilinear',align_corners=False).reshape(*x.shape[:-2],H,W).numpy()


def mask_iou(a,b):
    return float(np.logical_and(a,b).sum()/max(np.logical_or(a,b).sum(),1))


def annotate_arrays(model,history,generated,candidates,metadata,reader_sha,association):
    required=('initial_state','class_id','internal_state','predicted_visible_layers','predicted_full_layers','predicted_surface_depth_mm')
    for key in required:
        if key not in candidates:raise ValueError('Missing independently checkable candidate '+key)
    H,W=generated['rgb'].shape[-3:-1];N=len(candidates['class_id']);floor=float(np.asarray(metadata.get('world_to_anchor',np.eye(4)))[2,3]);reader=IndependentReader(model,association,floor_height_m=floor)
    for t in range(6):
        K=history['K'] if history['K'].ndim<4 else history['K'][t]
        reader.step(history['rgb'][t],history['depth_mm'][t],history['valid'][t],K,history['camera_to_world'][t],float(history['time_s'][t]))
    keys=list(reader.tracker.tracks);cost=np.full((N,len(keys)),1e6)
    for k in range(N):
        for j,key in enumerate(keys):
            track=reader.tracker.tracks[key]
            if track['mean'] is not None and track['class_id']==int(candidates['class_id'][k]):cost[k,j]=np.linalg.norm(track['mean'][:3]-candidates['initial_state'][k,:3])
    mapping={}
    if cost.size:
        i,j=linear_sum_assignment(cost)
        mapping={int(k):keys[v] for k,v in zip(i,j) if cost[k,v]<=5.}
    descriptors={k:reader.tracker.tracks[v]['descriptor'].copy() for k,v in mapping.items()}
    def geometry(name):
        a=candidates[name]
        if a.shape[2]==N+1:a=a[:,:,1:]
        if a.shape[-2:]==(60,92) and (H,W)==(480,720):return resize(a,480,736)[...,8:-8]
        return resize(a,H,W)
    P=geometry('predicted_full_layers');M=geometry('predicted_visible_layers');Z=geometry('predicted_surface_depth_mm')
    if P.shape!=(16,2,N,H,W) or M.shape!=P.shape or Z.shape!=P.shape:raise ValueError('Candidate geometry shape mismatch')
    intended=(M.sum((-2,-1))/np.maximum(P.sum((-2,-1)),1e-6)>=.1)&(P.sum((-2,-1))>=144)
    row=dict(metadata,annotation_reader_sha256=reader_sha,error_reference='sampled_state',visibility_source='conditioned_geometry',
        frames=16,views=['air','ground'],includes_history=False,conditioned_target_ids=list(range(N)),
        history_identity_map={str(k):str(v) for k,v in mapping.items()},reader_frame_coverage=[],intended_visibility=[],observations=[])
    instances=np.zeros((16,2,H,W),np.int32);state=np.full((16,N,11),np.nan,np.float32)
    for t in range(16):
        K=generated['K'] if generated['K'].ndim<4 else generated['K'][t]
        paired=reader.step(generated['rgb'][t],generated['depth_mm'][t],generated['valid'][t],K,
            generated['camera_to_world'][t],float(generated['time_s'][t]))
        frame_states={k:[] for k in range(N)}
        for v,view in enumerate(('air','ground')):
            row['reader_frame_coverage'].append(dict(frame=t,view=view))
            for k in range(N):row['intended_visibility'].append(dict(frame=t,view=view,target_id=k,visible=bool(intended[t,v,k])))
            for d in sorted(paired[view],key=lambda x:x['score']):
                k=next((k for k,value in mapping.items() if value==d['track_id']),None)
                if k is None:
                    row['observations'].append(dict(frame=t,view=view,target_id='unmatched:'+str(d['track_id']),class_confidence=d['score']))
                    continue
                if not finite_position(d):
                    # The detection remains visible to image metrics; it cannot
                    # fabricate a state label for a geometry-required cohort.
                    row['observations'].append(dict(frame=t,view=view,target_id=k,class_confidence=d['score'],
                        reader_identity=str(d['track_id']),geometry_available=False))
                    continue
                mask=d['mask']>=.5;expected=M[t,v,k]>=.5
                valid=mask&expected&generated['valid'][t,v]&(Z[t,v,k]>0)
                surface=float(np.abs(generated['depth_mm'][t,v][valid]-Z[t,v,k][valid]).mean()) if valid.any() else 1e9
                position=np.asarray(d['position_world_m']);similar=float(np.clip((np.dot(d['identity_embedding'],descriptors[k])+1)/2,0,1))
                T=generated['camera_to_world'][t,v];kv=K if K.ndim==2 else K[v]
                candidate=candidates['internal_state'][t,k];bottom=candidate[:3].copy();bottom[2]-=candidate[10]/2
                camera=T[:3,:3].T@(bottom-T[:3,3]);contact=kv@camera;contact=contact[:2]/max(contact[2],1e-6)
                obs=dict(frame=t,view=view,target_id=k,reader_identity=str(d['track_id']),class_confidence=d['score'],annotation_confidence=d['score'],
                    center_world_m=position.tolist(),state_error_m=float(np.linalg.norm(position-candidate[:3])),surface_error_mm=surface,
                    mask_iou=mask_iou(mask,expected),identity_similarity=similar,ground_contact_error_px=float(np.linalg.norm(np.asarray(d['contact_xy'])-contact)))
                row['observations'].append(obs);instances[t,v][mask]=k+1
                measured=np.zeros(11,np.float32);measured[:3]=position;measured[3:6]=d['velocity_world_mps'];measured[6]=d.get('yaw_rad',0)
                measured[8:11]=d.get('dimensions_m',[0,0,0]);frame_states[k].append(measured)
        for k,observed in frame_states.items():
            # Missing readout stays NaN. There is no future-state label fallback.
            if observed:
                state[t,k]=np.mean(observed,axis=0)
                headings=np.asarray(observed)[:,6]
                state[t,k,6]=np.arctan2(np.sin(headings).mean(),np.cos(headings).mean())
    annotation=dict(instance=instances,state=state,class_id=np.asarray(candidates['class_id']),track_id=np.arange(1,N+1))
    return row,annotation


def calibration_rows(proposal):
    """Calibration proposal must come from val original renderings, not synthesis."""
    if proposal.get('role')!='val' or proposal.get('source')!='rendered':raise ValueError('Calibration uses rendered validation only')
    observations={(r['frame'],r['view'],str(r['target_id'])):r for r in proposal['observations'] if 'state_error_m' in r}
    rows=[]
    for key,obs in observations.items():
        if not any(r['frame']==key[0] and r['view']==key[1] and str(r['target_id'])==key[2] and r['visible'] for r in proposal['intended_visibility']):continue
        # Match correctness is established on rendered labels for calibration only.
        if not obs.get('correctly_associated',False):continue
        other=observations.get((key[0],'ground' if key[1]=='air' else 'air',key[2]))
        rows.append(dict(obs,uid=proposal['uid']+':%d:%s:%s'%key,role='val',source='rendered',
            split_sha256=proposal['split_sha256'],annotation_reader_sha256=proposal['annotation_reader_sha256'],correctly_associated=True,
            jointly_visible=other is not None,paired_center_error_m=float(np.linalg.norm(np.asarray(obs['center_world_m'])-other['center_world_m'])) if other else None))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['annotate','calibrate','accept'],required=True);p.add_argument('--input',required=True)
    p.add_argument('--output',required=True);p.add_argument('--checkpoint');p.add_argument('--assets');p.add_argument('--association');p.add_argument('--thresholds')
    p.add_argument('--modality',choices=['rgb','rgbd'],default='rgbd');p.add_argument('--development',action='store_true');p.add_argument('--gpu-hours',type=float,default=24);a=p.parse_args()
    document=json.loads(Path(a.input).read_text())
    if document.get('diagnostic_only') or document.get('downstream_augmentation') is False:
        raise ValueError('Diagnostic/oracle generations cannot enter downstream augmentation')
    if not a.development and document.get('development_only'):raise ValueError('Development data cannot supply formal annotation')
    if a.mode=='calibrate':
        rows=document['records'];result=fit_thresholds(rows,document['split_sha256'],a.modality);atomic(a.output,result);return
    if a.mode=='accept':
        thresholds=json.loads(Path(a.thresholds).read_text());accepted,report=accept_generated(document['proposals'],thresholds,document['annotation_reader_sha256'])
        atomic(a.output,dict(accepted=accepted,report=report,completed=True));return
    if not a.association:raise ValueError('Use validation-calibrated association JSON')
    calibration=json.loads(Path(a.association).read_text())
    if not a.development and calibration.get('fit_role')!='val':raise ValueError('Association parameters must be fixed on validation')
    model,_=load_reader(a.checkpoint,a.assets,required_role='annotation',allow_development=a.development)
    if model.modality!=a.modality:raise ValueError('Annotation reader modality mismatch')
    reader_sha=sha256(a.checkpoint);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);proposals=[]
    runtime=ReaderRuntime(a.output,a.gpu_hours,[a.input,a.checkpoint,a.association],dict(mode=a.mode,modality=a.modality,development=a.development))
    for row in document['clips']:
        for pathkey,hashkey in [('history_path','history_sha256'),('rgbd_path','rgbd_sha256')]:
            if hashkey in row and sha256(row[pathkey])!=row[hashkey]:raise ValueError('Generated input changed: '+pathkey)
        cached=runtime.get(row['uid'])
        if cached is not None:
            if sha256(cached['annotation_path'])!=cached['annotation_sha256']:raise ValueError('Cached annotation changed')
            proposals.append(cached);continue
        if runtime.pause():return
        with np.load(row['history_path'],allow_pickle=False) as h:
            history={k:h[k][:6] if k!='K' else h[k] for k in ('rgb','depth_mm','valid','K','camera_to_world','time_s')}
        with np.load(row['rgbd_path'],allow_pickle=False) as h:
            generated={k:h[k] for k in ('rgb','K','camera_to_world')}
            if 'depth_mm' in h:
                generated['depth_mm']=h['depth_mm'];generated['valid']=h['valid'] if 'valid' in h else h['validity_probability']>=.5
            elif a.modality=='rgb':
                generated['depth_mm']=np.zeros(generated['rgb'].shape[:-1],np.float32);generated['valid']=np.zeros(generated['rgb'].shape[:-1],bool)
            else:raise ValueError('Missing metric depth in RGB-D annotation')
            generated['time_s']=h['time_s'] if 'time_s' in h else np.arange(1,17,dtype=np.float32)/5
        with np.load(row.get('candidate_path',row['rgbd_path']),allow_pickle=False) as h:
            candidates={k:h[k] for k in ('initial_state','class_id','internal_state','predicted_visible_layers','predicted_full_layers','predicted_surface_depth_mm')}
        result,annotation=annotate_arrays(model,history,generated,candidates,row,reader_sha,calibration['parameters'])
        path=out.parent/'annotations'/('%s.npz'%row['uid'].replace('/','_').replace(':','_'));path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**annotation);result.update(annotation_path=str(path.resolve()),annotation_sha256=sha256(path),annotation_modality=a.modality)
        runtime.put(row['uid'],result);proposals.append(result)
        if runtime.pause():return
    atomic(out,dict(proposals=proposals,annotation_reader_sha256=reader_sha,completed=True))

if __name__=='__main__':main()
