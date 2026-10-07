"""Continuous 100-frame tracking evaluation of independent downstream learners."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import read_json,write_json,read_jsonl,write_jsonl,sha256,VIEWS
from .readout import infer_arrays,native_reference
from .tracking import evaluate_tracking,combine_tracking,event_identity_metrics,coco_map,matched_indices,visible_detections
from .metrics import ratio


def inside_polygon(point,polygon):
    x,y=point;inside=False
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:inside=not inside
    return inside


def zone_events(rows,zones,reference=False):
    tracks=defaultdict(lambda:defaultdict(list));times={}
    for row in rows:
        t=int(row['frame']);times[t]=float(row['time_s'])
        for d in row['detections']:
            if not d.get('present',True):continue
            position=d.get('position_world_m')
            if position is None or np.shape(position)!=(3,) or not np.isfinite(position).all():continue
            tracks[(str(d['track_id']),int(d['class_id']))][t].append((np.asarray(position)[:2],bool(d.get('visible',True))))
    events=[]
    for (track,cls),frames in tracks.items():
        points={t:np.mean([v[0] for v in items],axis=0) for t,items in frames.items()}
        for zone in zones:
            flags={t:inside_polygon(point,zone['polygon']) for t,point in points.items()}
            for t in sorted(flags):
                if t-1 in flags and t+1 in flags and not flags[t-1] and flags[t] and flags[t+1]:
                    events.append(dict(track_id=track,class_id=cls,zone=zone['id'],frame=t,time_s=times[t],position_m=points[t].tolist(),
                        occluded_one_view=reference and any(not visible for f in (t-1,t,t+1) for _,visible in frames.get(f,[]))))
    return events


def zone_scores(reference,prediction):
    from scipy.optimize import linear_sum_assignment
    cost=np.full((len(reference),len(prediction)),1e6)
    for i,r in enumerate(reference):
        for j,p in enumerate(prediction):
            dt=abs(r['time_s']-p['time_s']);distance=np.linalg.norm(np.asarray(r['position_m'])-p['position_m'])
            if r['class_id']==p['class_id'] and r['zone']==p['zone'] and dt<=.400001 and distance<=1.:cost[i,j]=dt+distance
    matched=[]
    if cost.size:
        a,b=linear_sum_assignment(cost);matched=[(int(i),int(j)) for i,j in zip(a,b) if cost[i,j]<1e5]
    tp=len(matched);fn=len(reference)-tp;fp=len(prediction)-tp;occluded=[i for i,r in enumerate(reference) if r['occluded_one_view']]
    found={i for i,j in matched}
    return dict(tp=tp,fp=fp,fn=fn,precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),F1=ratio(2*tp,2*tp+fp+fn),
        occluded_entry_miss_rate=ratio(sum(i not in found for i in occluded),len(occluded)),occluded_reference_events=len(occluded))


def visibility_strata(reference,prediction,native):
    from .common import frame_index
    refs=frame_index(reference);preds=frame_index(prediction);episode=reference[0]['episode'];ids=native['instance_id']
    v=(native['visible_fraction']>=.1)&(native['projection_area']>=144);membership={name:defaultdict(set) for name in ('both_visible','aerial_only','ground_only','neither_visible')}
    names=('neither_visible','ground_only','aerial_only','both_visible')
    for start in (0,25,50,75):
        anchor=start+5
        eligible=v[start:anchor+1].any((0,1))
        for target,identity in enumerate(ids):
            if not eligible[target]:continue
            name=names[int(v[anchor,0,target])*2+int(v[anchor,1,target])]
            for t in range(anchor+1,min(anchor+17,len(v))):membership[name][t].add(str(identity))
    scores={}
    for name,selection in membership.items():
        rr=[];pp=[];targets=0
        for t,selected in selection.items():
            for view in VIEWS:
                key=(episode,t,view);r=refs[key];p=preds[key]
                included=[d for d in r['detections'] if str(d['track_id']) in selected];ignored=[d for d in visible_detections(r) if str(d['track_id']) not in selected]
                # Standard ignored-GT handling: unrelated correctly located
                # detections are removed, unmatched false positives remain.
                pairs,_=matched_indices(ignored,p['detections']);remove={j for i,j in pairs}
                rr.append(dict(r,detections=included));pp.append(dict(p,detections=[d for j,d in enumerate(p['detections']) if j not in remove]))
                targets+=sum(d.get('visible',True) for d in included)
        if rr and targets:
            stat=list(evaluate_tracking(rr,pp).values())[0];combined=combine_tracking([stat]);scores[name]=dict(HOTA=combined['HOTA'],IDF1=combined['IDF1'],ID_switches=combined['ID_switches'],reference_visible_detections=targets,tracking_statistics=stat)
        else:scores[name]=dict(HOTA=None,IDF1=None,reference_visible_detections=0,status='no_reference_visible_detections')
    return scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('index','checkpoint','assets','association','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--role',choices=['val','id_test','test','ood_class','ood_layout_class'],default='test');p.add_argument('--device',default='cuda:0')
    p.add_argument('--development',action='store_true');p.add_argument('--limit-episodes',type=int);p.add_argument('--zones');p.add_argument('--gpu-hours',type=float,default=24.)
    a=p.parse_args()
    from .runtime import Budget
    budget=Budget(a.gpu_hours)
    if a.limit_episodes and not a.development:raise ValueError('Formal downstream evaluation uses all registered complete episodes')
    from experiments.downstream_tracking.inference import load_reader
    from experiments.downstream_tracking.data import native_clip
    import h5py,torch
    torch.set_num_threads(4);index=read_json(a.index);association=read_json(a.association)
    model,saved=load_reader(a.checkpoint,a.assets,a.device,required_role='downstream',allow_development=a.development);identity=saved['identity']
    if not a.development:
        if association.get('split',association.get('fit_role'))!='val' or association.get('split_sha256')!=index['split_sha256']:raise ValueError('Freeze association on rendered validation of this split')
        if association.get('modality')!=identity['modality'] or association.get('seed')!=identity['seed']:raise ValueError('Use the matched R-only calibration across all data regimes')
    config=association.get('association',association.get('parameters',association));root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    windows=[r for r in index['windows'] if r['role']==a.role];episodes=sorted({r['episode'] for r in windows})
    if not episodes:raise ValueError('No episodes for role '+a.role)
    if a.limit_episodes:episodes=episodes[:a.limit_episodes]
    layoutfile=Path(index['dataset_root'])/'layouts.json';layouts=read_json(layoutfile) if layoutfile.exists() else None
    rows=[];allrefs=[];allpreds=[]
    manifest=dict(schema='agcr-downstream-evaluation-1',role=a.role,checkpoint_sha256=sha256(a.checkpoint),association_sha256=sha256(a.association),
        training_partition_sha256=identity['training_partition_sha256'],training_seed=identity['seed'],modality=identity['modality'],split_sha256=index['split_sha256'],
        development_only=a.development,frame_policy='continuous_100_frame_episode',calibration_reference_checkpoint=association.get('checkpoint_sha256'))
    if (root/'manifest.json').exists() and read_json(root/'manifest.json')!=manifest:raise ValueError('Mixed downstream evaluation provenance')
    write_json(root/'manifest.json',manifest)
    for episode in episodes:
        if budget.expired():
            write_json(root/'status.json',dict(completed_episodes=len(rows),total_episodes=len(episodes),budget_paused=True,elapsed_gpu_hours=budget.elapsed_hours()));return
        row=next(r for r in windows if r['episode']==episode);meta=dict(episode=episode,layout=row['layout'],scene_parent_id=row['camera_group'])
        inputs,labels=native_clip(index['dataset_root'],episode,0,frames=100)
        native=dict(instance=labels['instance'],instance_id=labels['track_id'],class_id=np.array([{1:3,2:4,3:13,4:14}[int(c)] for c in labels['class_id']]),
            world_state=labels['state'],present=np.ones(labels['state'].shape[:2],bool),time_s=inputs['time_s'])
        with h5py.File(Path(index['dataset_root'])/'episodes'/episode/'frames.h5') as h:
            native.update(visible_fraction=h['visible_fraction'][:],projection_area=h['full_area'][:])
        refs,events=native_reference(native,meta,anchors=(5,30,55,80));predfile=root/episode/'prediction.jsonl';marker=predfile.with_suffix('.complete.json')
        if marker.exists():
            if read_json(marker)['sha256']!=sha256(predfile):raise ValueError('Changed tracker output')
            preds=read_jsonl(predfile)
        else:
            preds=infer_arrays(model,config,inputs['rgb'],inputs['depth_mm'],inputs['valid'],inputs['K'],inputs['camera_to_world'],inputs['time_s'],meta)
            write_jsonl(predfile,preds);write_json(marker,dict(sha256=sha256(predfile)))
        write_jsonl(predfile.parent/'reference.jsonl',refs);write_json(predfile.parent/'events.json',events)
        stats=list(evaluate_tracking(refs,preds).values())[0];tracking=combine_tracking([stats]);event=event_identity_metrics(refs,preds,events)
        zones=None
        if a.zones:zones=read_json(a.zones)['zones']
        elif layouts is not None:
            if isinstance(layouts,list):zones=next(z['zones'] for z in layouts if z.get('index')==row['layout'])
            elif 'layouts' in layouts:zones=next(z['zones'] for z in layouts['layouts'] if z.get('index')==row['layout'])
        entry=None if zones is None else zone_scores(zone_events(refs,zones,True),zone_events(preds,zones))
        result=dict(meta,unit_id=episode,training_seed=identity['seed'],metrics={k:tracking[k] for k in ('HOTA','IDF1','ID_switches')},tracking_statistics=stats,
            events=event,entry=entry,visibility_strata=visibility_strata(refs,preds,native))
        result['metrics'].update({k:event[k] for k in ('R3','cross_view_ID_agreement','joint_detection_coverage')})
        rows.append(result);allrefs.extend(refs);allpreds.extend(preds)
        write_json(root/'status.json',dict(completed_episodes=len(rows),total_episodes=len(episodes)))
    pooled=combine_tracking([r['tracking_statistics'] for r in rows]);map_result=coco_map(allrefs,allpreds)
    # Layout-balanced primary HOTA/IDF1 recombines official statistics within
    # layout, not a mean of episode scores. Pooled values remain supplemental.
    bylayout=defaultdict(list)
    for r in rows:bylayout[r['layout']].append(r)
    primary={}
    for metric in ('HOTA','IDF1'):primary[metric]=float(np.mean([combine_tracking([r['tracking_statistics'] for r in v])[metric] for v in bylayout.values()]))
    primary['R3']=None if any(sum(r['events']['R3_events'] for r in v)==0 for v in bylayout.values()) else float(np.mean([sum(r['events']['R3_success'] for r in v)/sum(r['events']['R3_events'] for r in v) for v in bylayout.values()]))
    write_json(root/'report.json',dict(manifest,complete=True,rows=rows,primary_equal_layout=primary,pooled_tracking=pooled,coco=map_result,
        zones_status='loaded' if all(r['entry'] is not None for r in rows) else 'missing_native_zone_polygons'))
    write_json(root/'completion.json',dict(completed=True,development_only=a.development,episodes=len(rows),report_sha256=sha256(root/'report.json')))
if __name__=='__main__':main()
