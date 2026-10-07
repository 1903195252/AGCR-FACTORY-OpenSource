"""Official TrackEval/COCO backends and identity events over continuous episodes."""
from collections import defaultdict,Counter
import numpy as np
from .common import frame_index,VIEWS
from .metrics import box_iou,matched_indices,ratio,metric_position


def _backend():
    # TrackEval's pinned official revision uses removed NumPy scalar aliases.
    # Restore only their exact historical dtypes; no metric equations change.
    if 'float' not in np.__dict__:np.float=np.float64
    if 'int' not in np.__dict__:np.int=np.int64
    if 'bool' not in np.__dict__:np.bool=np.bool_
    import trackeval
    return trackeval


def visible_detections(frame):return [d for d in frame['detections'] if d.get('present',True) and d.get('visible',True)]

def official_data(reference,prediction):
    from experiments.paper_experiments.agcr_exp.official_metrics import trackeval_sequence
    gids=[];pids=[];similarities=[]
    for r,p in zip(reference,prediction):
        g=visible_detections(r);d=p['detections']
        sim=box_iou([x['box_xyxy'] for x in g],[x['box_xyxy'] for x in d])
        sim*=np.equal.outer([x['class_id'] for x in g],[x['class_id'] for x in d])
        gids.append([str(x['track_id']) for x in g]);pids.append([str(x['track_id']) for x in d]);similarities.append(sim)
    # Arbitrary string identities are valid; map once over the whole episode.
    gm={v:i for i,v in enumerate(sorted({x for row in gids for x in row}))}
    pm={v:i for i,v in enumerate(sorted({x for row in pids for x in row}))}
    return trackeval_sequence([[gm[x] for x in row] for row in gids],[[pm[x] for x in row] for row in pids],similarities)


def evaluate_tracking(reference_rows,prediction_rows):
    trackeval=_backend()
    ref=frame_index(reference_rows);pred=frame_index(prediction_rows)
    if set(ref)!=set(pred):raise ValueError('Missing prediction frames: empty frames must be explicitly exported')
    episodes=sorted({k[0] for k in ref});output={}
    for episode in episodes:
        per_view={}
        for view in VIEWS:
            keys=sorted(k for k in ref if k[0]==episode and k[2]==view)
            d=official_data([ref[k] for k in keys],[pred[k] for k in keys])
            per_view[view]=dict(HOTA=trackeval.metrics.HOTA().eval_sequence(d),
                CLEAR=trackeval.metrics.CLEAR({'THRESHOLD':.5,'PRINT_CONFIG':False}).eval_sequence(d))
        keys=sorted(k for k in ref if k[0]==episode)
        joint=official_data([ref[k] for k in keys],[pred[k] for k in keys])
        identity=trackeval.metrics.Identity({'THRESHOLD':.5,'PRINT_CONFIG':False}).eval_sequence(joint)
        first=ref[keys[0]]
        output[episode]=dict(views=per_view,Identity=identity,layout=first.get('layout'),scene_parent_id=first.get('scene_parent_id'))
    return output


def combine_tracking(episodes):
    trackeval=_backend()
    if not episodes:raise ValueError('No tracking episodes')
    h=[];switches=0;stats={}
    for view in VIEWS:
        v=trackeval.metrics.HOTA().combine_sequences({str(i):x['views'][view]['HOTA'] for i,x in enumerate(episodes)})
        c=trackeval.metrics.CLEAR({'THRESHOLD':.5,'PRINT_CONFIG':False}).combine_sequences({str(i):x['views'][view]['CLEAR'] for i,x in enumerate(episodes)})
        h.append(float(np.mean(v['HOTA'])));switches+=int(c['IDSW']);stats[view]=dict(HOTA=v,CLEAR=c)
    ids=trackeval.metrics.Identity({'THRESHOLD':.5,'PRINT_CONFIG':False}).combine_sequences({str(i):x['Identity'] for i,x in enumerate(episodes)})
    return dict(HOTA=float(np.mean(h)),IDF1=float(ids['IDF1']),ID_switches=switches,per_view_HOTA=dict(zip(VIEWS,h)),sufficient_statistics=dict(views=stats,Identity=ids))


def restore_tracking_arrays(episode):
    # JSON serializes TrackEval's 19 thresholds as lists. Restore before its
    # combine_sequences does elementwise numeric reductions in bootstrap.
    out=dict(episode,views={})
    for view,groups in episode['views'].items():
        out['views'][view]={name:{k:np.asarray(v) if isinstance(v,list) else v for k,v in fields.items()} for name,fields in groups.items()}
    return out


def historical_identity_map(reference_rows,prediction_rows,history_frames=6):
    """Evaluation-only ID alignment fixed using history; never a model input."""
    from scipy.optimize import linear_sum_assignment
    r=frame_index(reference_rows);p=frame_index(prediction_rows);counts=Counter()
    for key in sorted(r):
        if key[1]>=history_frames:continue
        if key not in p:raise ValueError('Reader must export conditioning history')
        g=visible_detections(r[key]);d=p[key]['detections'];pairs,_=matched_indices(g,d)
        for i,j in pairs:counts[(str(g[i]['track_id']),str(d[j]['track_id']))]+=1
    gids=sorted({k[0] for k in counts});pids=sorted({k[1] for k in counts})
    if not gids or not pids:return {}
    c=np.array([[counts[g,q] for q in pids] for g in gids]);a,b=linear_sum_assignment(-c)
    return {gids[i]:pids[j] for i,j in zip(a,b) if c[i,j]>0}


def event_identity_metrics(reference_rows,prediction_rows,events,score_from_frame=0):
    """R3 retains missing pre-occlusion IDs as failures; matching is visual IoU."""
    ref=frame_index(reference_rows);pred=frame_index(prediction_rows);matches={}
    for key in ref:
        if key not in pred:raise ValueError('Missing frame export')
        g=visible_detections(ref[key]);d=pred[key]['detections'];pairs,_=matched_indices(g,d)
        matches[key]={str(g[i]['track_id']):str(d[j]['track_id']) for i,j in pairs}
    success=0;established=0;rows=[]
    for event in events:
        ep=str(event['episode']);view=event['view'];target=str(event['track_id']);start=int(event['hidden_start']);h=int(event['reappearance'])
        previous=[]
        # Global identity can have been established in either platform, but
        # only BEFORE the hidden interval (not by seeing its future reappearance).
        for key,value in matches.items():
            if key[0]==ep and key[1]<start and target in value:previous.append((key[1],value[target]))
        identity=max(previous,key=lambda x:x[0])[1] if previous else None
        recovered=identity is not None and any(matches.get((ep,frame,view),{}).get(target)==identity for frame in range(h,h+3))
        success+=int(recovered);established+=int(identity is not None)
        rows.append(dict(event,R3_success=bool(recovered),pre_occlusion_id_established=identity is not None))
    joint=0;detected=0;agree=0;errors=[]
    for ep,frame in sorted({(k[0],k[1]) for k in ref}):
        if frame<score_from_frame:continue
        a,g=(ep,frame,'air'),(ep,frame,'ground')
        ga={str(x['track_id']):x for x in visible_detections(ref[a])};gg={str(x['track_id']):x for x in visible_detections(ref[g])}
        for target in set(ga)&set(gg):
            joint+=1
            if target in matches[a] and target in matches[g]:
                detected+=1;agree+=int(matches[a][target]==matches[g][target])
                da=next(x for x in pred[a]['detections'] if str(x['track_id'])==matches[a][target]);dg=next(x for x in pred[g]['detections'] if str(x['track_id'])==matches[g][target])
                pa,pg=metric_position(da),metric_position(dg)
                if pa is not None and pg is not None:errors.append(float(np.linalg.norm(pa-pg)))
    return dict(R3=ratio(success,len(events)),R3_success=success,R3_events=len(events),pre_occlusion_ID_coverage=ratio(established,len(events)),
        cross_view_ID_agreement=ratio(agree,detected),joint_detection_coverage=ratio(detected,joint),
        jointly_visible_count=joint,jointly_detected_count=detected,identity_agreement_count=agree,
        cross_view_center_error_m=None if not errors else float(np.mean(errors)),center_count=len(errors),events=rows)


def coco_map(reference_rows,prediction_rows):
    """Official COCO AP@[.50:.95], visible-instance boxes, full corpus."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ref=frame_index(reference_rows);pred=frame_index(prediction_rows)
    if set(ref)!=set(pred):raise ValueError('Frame populations differ')
    images=[];annotations=[];detections=[];classes=set()
    for image_id,key in enumerate(sorted(ref),1):
        frame=ref[key];h,w=frame['canvas_size'];images.append(dict(id=image_id,width=w,height=h))
        for d in visible_detections(frame):
            x,y,x2,y2=d['box_xyxy'];box=[x,y,x2-x,y2-y];classes.add(int(d['class_id']))
            annotations.append(dict(id=len(annotations)+1,image_id=image_id,category_id=int(d['class_id']),bbox=box,area=box[2]*box[3],iscrowd=0))
        for d in pred[key]['detections']:
            x,y,x2,y2=d['box_xyxy'];classes.add(int(d['class_id']))
            detections.append(dict(image_id=image_id,category_id=int(d['class_id']),bbox=[x,y,x2-x,y2-y],score=float(d['score'])))
    gt=COCO();gt.dataset=dict(images=images,annotations=annotations,categories=[dict(id=c,name=str(c)) for c in sorted(classes)],info={});gt.createIndex()
    if detections:dt=gt.loadRes(detections)
    else:
        dt=COCO();dt.dataset=dict(images=images,annotations=[],categories=gt.dataset['categories'],info={});dt.createIndex()
    ev=COCOeval(gt,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
    return dict(mAP_50_95=float(ev.stats[0]) if ev.stats[0]>=0 else None,AP50=float(ev.stats[1]) if ev.stats[1]>=0 else None,
                reference_detections=len(annotations),prediction_detections=len(detections))
