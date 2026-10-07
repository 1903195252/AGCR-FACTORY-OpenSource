"""Trajectory-free common RGB annotation for ALL external-generator comparisons.

This protocol never compares a native generator against another model's sampled
world state. Missing-actor/occlusion quality is measured on UNFILTERED outputs;
automatic retention and the separately pending human audit are reported apart.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .data import sha256
from .inference import IndependentReader,load_reader,finite_position
from .train import atomic
from .runtime import ReaderRuntime
from .annotations import mask_iou
from experiments.paper_experiments.agcr_exp.protocol import digest

RULES={'annotation_confidence':'min','identity_similarity':'min','paired_center_error_m':'max','contact_reprojection_error_px':'max'}


def annotate_rgb(model,history,generated,metadata,association,reader_sha):
    if model.modality!='rgb':raise ValueError('Common RGB annotation requires an RGB-only independent reader')
    floor=float(np.asarray(metadata.get('world_to_anchor',np.eye(4)))[2,3]);reader=IndependentReader(model,association,floor_height_m=floor)
    for t in range(6):
        K=history['K'] if history['K'].ndim<4 else history['K'][t]
        reader.step(history['rgb'][t],history['depth_mm'][t],history['valid'][t],K,history['camera_to_world'][t],float(history['time_s'][t]))
    historical={k:dict(class_id=v['class_id'],descriptor=v['descriptor'].copy()) for k,v in reader.tracker.tracks.items()}
    ids=sorted(historical);slot={k:i for i,k in enumerate(ids)};H,W=generated['rgb'].shape[-3:-1]
    instance=np.zeros((16,2,H,W),np.int32);state=np.full((16,len(ids),11),np.nan,np.float32);observations=[];unmatched=[]
    for t in range(16):
        K=generated['K'] if generated['K'].ndim<4 else generated['K'][t]
        # RGB model ignores depth and validity. No depth is synthesized for this protocol.
        z=np.zeros((2,H,W),np.float32)
        detected=reader.step(generated['rgb'][t],z,z.astype(bool),K,generated['camera_to_world'][t],float(generated['time_s'][t]))
        by_view={v:{d['track_id']:d for d in detected[v]} for v in ('air','ground')}
        measurements={k:[] for k in ids}
        for v,view in enumerate(('air','ground')):
            for d in sorted(detected[view],key=lambda a:a['score']):
                key=d['track_id']
                if key not in historical:
                    unmatched.append(dict(frame=t,view=view,track_id=key,confidence=d['score']));continue
                j=slot[key];mask=d['mask']>=.5;instance[t,v][mask]=j+1
                geometry_available=finite_position(d)
                if geometry_available:
                    measured=np.zeros(11,np.float32);measured[:3]=d['position_world_m'];measured[3:6]=d['velocity_world_mps'];measurements[key].append(measured)
                other=by_view['ground' if view=='air' else 'air'].get(key)
                obs=dict(frame=t,view=view,track_id=key,annotation_slot=j+1,class_id=d['class_id'],annotation_confidence=d['score'],
                    identity_similarity=float(np.clip((np.dot(d['identity_embedding'],historical[key]['descriptor'])+1)/2,0,1)),
                    jointly_visible=other is not None,center_world_m=d['position_world_m'],geometry_available=geometry_available,
                    localization_scope=d.get('localization_scope'))
                if other is not None and geometry_available and finite_position(other):
                    obs['paired_center_error_m']=float(np.linalg.norm(np.asarray(d['position_world_m'])-other['position_world_m']))
                    T=generated['camera_to_world'][t,v];kv=K if K.ndim==2 else K[v]
                    camera=T[:3,:3].T@(np.asarray(other['position_world_m'])-T[:3,3]);uv=kv@camera
                    obs['contact_reprojection_error_px']=float(np.linalg.norm(uv[:2]/max(uv[2],1e-6)-d['contact_xy']))
                observations.append(obs)
        for key,values in measurements.items():
            if values:state[t,slot[key]]=np.mean(values,0)
    row=dict(metadata,annotation_reader_sha256=reader_sha,annotation_modality='rgb',frames=16,views=['air','ground'],includes_history=False,
        annotation_protocol='independent_rgb_trajectory_free_v1',historical_targets=len(ids),observations=observations,unmatched=unmatched,
        scope='image-derived labels; no sampled-state/visibility truth used for retention')
    return row,dict(instance=instance,state=state,class_id=np.array([historical[k]['class_id'] for k in ids],np.int64),track_id=np.arange(1,len(ids)+1))


def fit(rows,split_sha256,reader_sha):
    if not rows or any(r.get('role')!='val' or not r.get('correctly_associated') for r in rows):raise ValueError('Only correctly associated original validation observations may calibrate')
    thresholds={}
    for name,direction in RULES.items():
        values=[r[name] for r in rows if name in r and (name not in ('paired_center_error_m','contact_reprojection_error_px') or r.get('jointly_visible'))]
        if not values or not np.isfinite(values).all():raise ValueError('No finite validation measurements for '+name)
        thresholds[name]=dict(direction=direction,value=float(np.quantile(values,.05 if direction=='min' else .95)),observations=len(values))
    out=dict(schema='agcr-rgb-acceptance-thresholds-1',fit_role='val',split_sha256=split_sha256,annotation_reader_sha256=reader_sha,
        thresholds=thresholds,trajectory_filter=False,future_truth_filter=False)
    out['threshold_sha256']=digest(out);return out


def accept(proposals,thresholds):
    if thresholds.get('schema')!='agcr-rgb-acceptance-thresholds-1' or thresholds['fit_role']!='val':raise ValueError('Wrong RGB quality threshold protocol')
    if digest({k:v for k,v in thresholds.items() if k!='threshold_sha256'})!=thresholds['threshold_sha256']:raise ValueError('Changed thresholds')
    accepted=[];rejected=[]
    if len({r['uid'] for r in proposals})!=len(proposals):raise ValueError('Duplicate proposals')
    for row in proposals:
        failures=[]
        if row.get('role')!='train':failures.append('not_training_origin')
        if row.get('annotation_protocol')!='independent_rgb_trajectory_free_v1':failures.append('wrong_protocol')
        if row['split_sha256']!=thresholds['split_sha256'] or row['annotation_reader_sha256']!=thresholds['annotation_reader_sha256']:raise ValueError('Split/reader mismatch')
        if not row['observations']:failures.append('no_history_matched_detection')
        if row['unmatched']:failures.append('unmatched_high_confidence_actor')
        for obs in row['observations']:
            if obs.get('geometry_available') is False:failures.append('missing_world_geometry')
            for key,rule in thresholds['thresholds'].items():
                if key not in obs:
                    if key in ('paired_center_error_m','contact_reprojection_error_px') and not obs['jointly_visible']:continue
                    failures.append('missing_'+key);continue
                x=obs[key]
                if not np.isfinite(x) or (x<rule['value'] if rule['direction']=='min' else x>rule['value']):failures.append(key)
        if failures:rejected.append(dict(uid=row['uid'],reasons=sorted(set(failures))))
        else:accepted.append(dict(row,acceptance_schema='agcr-acceptance-3-rgb',acceptance_threshold_sha256=thresholds['threshold_sha256']))
    return dict(completed=True,accepted=accepted,report=dict(proposal_count=len(proposals),accepted_count=len(accepted),
        acceptance_rate=len(accepted)/max(len(proposals),1),rejected=rejected,trajectory_filter=False,
        missing_actor_evaluation='unfiltered independent metrics plus separately pending 200-clip human audit'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['annotate','calibrate','accept'],required=True)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--index');p.add_argument('--checkpoint');p.add_argument('--assets')
    p.add_argument('--association');p.add_argument('--thresholds');p.add_argument('--limit',type=int,default=64);p.add_argument('--development',action='store_true');p.add_argument('--gpu-hours',type=float,default=24);a=p.parse_args()
    document=json.loads(Path(a.input).read_text())
    if document.get('diagnostic_only') or document.get('downstream_augmentation') is False:
        raise ValueError('Diagnostic/oracle generations cannot enter downstream augmentation')
    if not a.development and document.get('development_only'):raise ValueError('Development data cannot supply formal annotation')
    if a.mode=='accept':atomic(a.output,accept(document['proposals'],json.loads(Path(a.thresholds).read_text())));return
    model,_=load_reader(a.checkpoint,a.assets,required_role='annotation',allow_development=a.development);reader_sha=sha256(a.checkpoint)
    config=json.loads(Path(a.association).read_text())
    if config.get('fit_role')!='val':raise ValueError('Association thresholds must be fitted on validation')
    parameters=config['parameters']
    runtime=ReaderRuntime(a.output,a.gpu_hours,[a.input,a.checkpoint,a.association],dict(mode=a.mode,limit=a.limit,development=a.development))
    if a.mode=='calibrate':
        from experiments.paper_experiments.agcr_exp.data import read_window
        index=document;rows=[r for r in index['windows'] if r['role']=='val' and r['history_covered']][:a.limit]
        if not a.development and len(rows)<64:raise ValueError('At least64 validation clips required')
        records=[]
        for metadata in rows:
            cached=runtime.get(metadata['uid'])
            if cached is not None:records.extend(cached);continue
            if runtime.pause():return
            start_record=len(records)
            sample=read_window(index['dataset_root'],metadata);x,y=sample['inputs'],sample['labels'];history=dict(x,valid=x['depth_valid'])
            generated=dict(rgb=y['rgb'],K=x['K'],camera_to_world=x['planned_camera_to_world'],time_s=x['future_time_s'])
            metadata=dict(metadata,world_to_anchor=sample['metadata']['world_to_anchor'])
            row,labels=annotate_rgb(model,history,generated,metadata,parameters,reader_sha)
            # Ground truth appears ONLY here, establishing correct validation matches.
            pred_ids=labels['track_id'];gt_ids=y['track_id'];score=np.zeros((len(pred_ids),len(gt_ids)))
            for i,pid in enumerate(pred_ids):
                for j,gid in enumerate(gt_ids):score[i,j]=np.logical_and(labels['instance']==pid,y['visible_mask'][:,:,j]).sum()
            mapping={}
            if score.size:
                ii,jj=linear_sum_assignment(-score);mapping={int(pred_ids[i]):j for i,j in zip(ii,jj) if score[i,j]>0}
            for obs in row['observations']:
                slot=obs['annotation_slot'];t=obs['frame'];v=('air','ground').index(obs['view']);j=mapping.get(slot)
                if j is not None and obs['class_id']==int(y['class_id'][j]) and mask_iou(labels['instance'][t,v]==slot,y['visible_mask'][t,v,j])>=.5:
                    records.append(dict(obs,role='val',correctly_associated=True))
            runtime.put(metadata['uid'],records[start_record:])
            if runtime.pause():return
        atomic(a.output,fit(records,index['split_sha256'],reader_sha));return
    proposals=[];out=Path(a.output)
    for metadata in document['clips']:
        for pathkey,hashkey in [('history_path','history_sha256'),('rgbd_path','rgbd_sha256')]:
            if hashkey in metadata and sha256(metadata[pathkey])!=metadata[hashkey]:raise ValueError('Generated input changed: '+pathkey)
        cached=runtime.get(metadata['uid'])
        if cached is not None:
            if sha256(cached['annotation_path'])!=cached['annotation_sha256']:raise ValueError('Cached annotation changed')
            proposals.append(cached);continue
        if runtime.pause():return
        with np.load(metadata['history_path'],allow_pickle=False) as z:
            history={k:z[k][:6] if k!='K' else z[k] for k in ('rgb','depth_mm','valid','K','camera_to_world','time_s')}
        with np.load(metadata['rgbd_path'],allow_pickle=False) as z:
            generated={k:z[k] for k in ('rgb','K','camera_to_world')};generated['time_s']=z['time_s'] if 'time_s' in z else np.arange(1,17,dtype=np.float32)/5
        row,labels=annotate_rgb(model,history,generated,metadata,parameters,reader_sha)
        path=out.parent/'annotations_rgb'/(hashlib.sha256(metadata['uid'].encode()).hexdigest()+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**labels);row.update(annotation_path=str(path.resolve()),annotation_sha256=sha256(path));proposals.append(row)
        runtime.put(metadata['uid'],row)
        if runtime.pause():return
    atomic(out,dict(proposals=proposals,completed=True))

if __name__=='__main__':main()
