"""Fit association and annotation thresholds on rendered validation only."""
import argparse,json,itertools
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .association import CooperativeTracker
from .data import native_clip,sha256
from .inference import load_reader,predict_pair
from .annotations import annotate_arrays,calibration_rows
from .train import atomic
from .runtime import ReaderRuntime
from experiments.paper_experiments.agcr_exp.downstream import fit_thresholds


def iou(a,b):
    a=np.asarray(a);b=np.asarray(b)
    intersection=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    return intersection/max((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-intersection,1e-6)


def identity_score(episodes,parameters):
    tp=gt_count=pred_count=0
    for episode in episodes:
        tracker=CooperativeTracker(parameters);pairs={};gids=set();pids=set();gc=pc=0
        for frame in episode:
            prediction=tracker.update(frame['detections'],frame['time_s'])
            for view in ('air','ground'):
                gt=frame['reference'][view];pr=prediction[view];gc+=len(gt);pc+=len(pr)
                gids.update(g['track_id'] for g in gt);pids.update(p['track_id'] for p in pr)
                cost=np.ones((len(gt),len(pr)))
                for i,g in enumerate(gt):
                    for j,p in enumerate(pr):
                        if g['class_id']==p['class_id']:cost[i,j]=1-iou(g['box_xyxy'],p['box_xyxy'])
                if cost.size:
                    ii,jj=linear_sum_assignment(cost)
                    for i,j in zip(ii,jj):
                        if cost[i,j]<=.5:
                            key=(gt[i]['track_id'],pr[j]['track_id']);pairs[key]=pairs.get(key,0)+1
        gids=sorted(gids);pids=sorted(pids);counts=np.array([[pairs.get((g,p),0) for p in pids] for g in gids])
        if counts.size:
            i,j=linear_sum_assignment(-counts);tp+=int(counts[i,j].sum())
        gt_count+=gc;pred_count+=pc
    return dict(idf1=2*tp/max(gt_count+pred_count,1),identity_true_positives=tp,gt_detections=gt_count,predicted_detections=pred_count)


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--assets',required=True)
    p.add_argument('--output',required=True);p.add_argument('--mode',choices=['association','annotation'],required=True)
    p.add_argument('--association');p.add_argument('--modality',choices=['rgb','rgbd'],default='rgbd');p.add_argument('--development',action='store_true')
    p.add_argument('--limit',type=int,default=64);p.add_argument('--gpu-hours',type=float,default=24);a=p.parse_args();index=json.loads(Path(a.index).read_text())
    model,saved=load_reader(a.checkpoint,a.assets,required_role='annotation' if a.mode=='annotation' else None,allow_development=a.development)
    if not a.development and a.limit<64:raise ValueError('Formal validation panel requires at least64 episodes/windows')
    runtime=ReaderRuntime(a.output,a.gpu_hours,[a.index,a.checkpoint,a.association],dict(mode=a.mode,modality=a.modality,limit=a.limit,development=a.development))
    if a.mode=='association':
        episodes=sorted({r['episode'] for r in index['windows'] if r['role']=='val'})[:a.limit];cache=[]
        for episode in episodes:
            cached=runtime.get(episode)
            if cached is not None:cache.append(cached);continue
            if runtime.pause():return
            inputs,labels=native_clip(index['dataset_root'],episode,0,100);frames=[]
            for t in range(100):
                detected=predict_pair(model,inputs['rgb'][t],inputs['depth_mm'][t],inputs['valid'][t],inputs['K'],inputs['camera_to_world'][t],confidence=.3)
                for view in detected:
                    for d in detected[view]:d.pop('mask',None)
                reference={}
                for v,view in enumerate(('air','ground')):
                    reference[view]=[]
                    for k,track in enumerate(labels['track_id']):
                        yy,xx=np.where(labels['instance'][t,v]==track)
                        if len(xx):reference[view].append(dict(track_id=int(track),class_id=int(labels['class_id'][k]),box_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]))
                frames.append(dict(detections=detected,reference=reference,time_s=float(inputs['time_s'][t])))
            cache.append(frames);runtime.put(episode,frames)
            if runtime.pause():return
        scores=[]
        for weight,gate,confidence in itertools.product([.3,.6,.8],[9.21,16.27],[.3,.5]):
            parameters=dict(appearance_weight=weight,mahalanobis_gate=gate,confidence=confidence)
            scores.append(dict(parameters=parameters,**identity_score(cache,parameters)))
        best=max(scores,key=lambda r:(r['idf1'],r['parameters']['confidence']))
        atomic(a.output,dict(fit_role='val',split='val',parameters=best['parameters'],association=best['parameters'],selection='global_idf1',candidates=scores,
            checkpoint_sha256=sha256(a.checkpoint),split_sha256=index['split_sha256'],episodes=episodes,development_only=a.development,
            learner_role=saved['identity']['role'],modality=saved['identity']['modality'],seed=saved['identity']['seed'],training_partition_sha256=saved['identity']['training_partition_sha256']))
    else:
        from experiments.paper_experiments.agcr_exp.data import read_window
        parameters=json.loads(Path(a.association).read_text())['parameters'];records=[];reader_sha=sha256(a.checkpoint)
        rows=[r for r in index['windows'] if r['role']=='val' and r['history_covered']][:a.limit]
        for row in rows:
            cached=runtime.get(row['uid'])
            if cached is not None:records.extend(cached);continue
            if runtime.pause():return
            sample=read_window(index['dataset_root'],row,load_surfaces=True);x,y=sample['inputs'],sample['labels']
            history=dict(x,valid=x['depth_valid'])
            generated=dict(rgb=y['rgb'],depth_mm=y['depth_mm'],valid=y['depth_valid'],K=x['K'],camera_to_world=x['planned_camera_to_world'],time_s=x['future_time_s'])
            candidates=dict(initial_state=y['history_state'][-1],class_id=y['class_id'],internal_state=y['state'],
                predicted_visible_layers=y['visible_mask'],predicted_full_layers=y['full_mask'],predicted_surface_depth_mm=y['isolated_depth_mm'])
            meta=dict(row,source='rendered',split_sha256=index['split_sha256'],world_to_anchor=sample['metadata']['world_to_anchor'])
            proposal,_=annotate_arrays(model,history,generated,candidates,meta,reader_sha,parameters)
            for observation in proposal['observations']:
                observation['correctly_associated']=observation.get('mask_iou',0)>=.5
            current=calibration_rows(proposal);records.extend(current);runtime.put(row['uid'],current)
            if runtime.pause():return
        thresholds=fit_thresholds(records,index['split_sha256'],a.modality)
        thresholds['development_only']=a.development
        # Digest covers all fields used in acceptance, including development provenance.
        from experiments.paper_experiments.agcr_exp.protocol import digest
        thresholds['threshold_sha256']=digest({k:v for k,v in thresholds.items() if k!='threshold_sha256'})
        atomic(a.output,thresholds)
        atomic(str(a.output)+'.records.json',dict(records=records,split_sha256=index['split_sha256']))

if __name__=='__main__':main()
