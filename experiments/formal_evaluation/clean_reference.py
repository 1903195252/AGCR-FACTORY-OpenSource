"""Clean-sensor control for the independent video reader, never generated data.

Use metadata-selected train/val panels for diagnosis. The fixed test comparison
uses precisely the registered generated families and draw zero, with the same
checkpoint and validation-selected association thresholds. No labels select
network queries, detections or tracks. Reference targets only score predictions.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .common import read_json,write_json,read_jsonl,write_jsonl,sha256,VIEWS,frame_index
from .metrics import matched_indices
from .tracking import (visible_detections,event_identity_metrics,evaluate_tracking,
                       combine_tracking,restore_tracking_arrays)
from .readout import metadata,native_reference,mask_rle,evaluation_geometry

SCHEMA='agcr-clean-reference-reader-diagnostic-3'


def check_reader_policy(saved,current,rerun_generated=False):
    if not rerun_generated and any(saved.get(key)!=current.get(key) for key in
            ('reader_postprocessing','reader_source_sha256')):
        raise ValueError('Reader implementation differs: rerun retained generated pixels into a new diagnostic version')


def paired_source_policy(predictions,readouts,rerun,current,index_sha256):
    """Rereading pixels changes the reader explicitly; reuse never does."""
    if readouts and not predictions:raise ValueError('Readouts require their generated corpus')
    if rerun and not predictions:raise ValueError('Re-reading requires a fixed paired generated corpus')
    if predictions and not rerun and not readouts:raise ValueError('Reuse requires existing independent readouts')
    if not predictions:return None,None
    pair=read_json(Path(predictions)/'manifest.json')
    if pair.get('index_sha256')!=index_sha256:raise ValueError('Paired generator reference index mismatch')
    previous=None
    if readouts:
        path=Path(readouts)/'manifest.json';saved=read_json(path)
        if saved.get('generator_manifest_sha256')!=sha256(Path(predictions)/'manifest.json'):
            raise ValueError('Changed paired generated corpus')
        if not rerun:
            for key in ('checkpoint_sha256','association_sha256'):
                if saved.get(key)!=current[key]:raise ValueError('Reused readout differs from current reader: '+key)
            check_reader_policy(saved,current)
        previous=dict(manifest_sha256=sha256(path),checkpoint_sha256=saved.get('checkpoint_sha256'),
            association_sha256=saved.get('association_sha256'),reader_postprocessing=saved.get('reader_postprocessing'),
            reader_source_sha256=saved.get('reader_source_sha256'),reused_predictions=not rerun)
    return pair,previous


def infer_generated(model,config,arrays,generated,meta,reference,threshold):
    """Use exactly the current clean reader with actual retained future pixels."""
    K,poses=evaluation_geometry(generated,arrays['rgb'].shape,arrays['K'],arrays['camera_to_world'],arrays['time_s'])
    if 'depth_mm' not in generated and model.modality!='rgb':raise ValueError('RGB-D reader needs generated depth')
    depth=generated.get('depth_mm',np.zeros_like(arrays['depth_mm'][6:]))
    valid=generated.get('validity_probability',np.zeros_like(arrays['valid'][6:]))>=.5
    observed=dict(rgb=np.concatenate([arrays['rgb'][:6],generated['rgb']]),
        depth_mm=np.concatenate([arrays['depth_mm'][:6],depth]),valid=np.concatenate([arrays['valid'][:6],valid]),
        K=K,camera_to_world=poses,time_s=arrays['time_s'])
    if 'world_from_factory' in arrays:observed['world_from_factory']=arrays['world_from_factory']
    return infer_clean(model,config,observed,meta,reference,threshold)


def select_families(index,split,per_stratum=1,paired_manifest=None):
    rows=[r for r in index['families'] if r['split']==split]
    if split=='test':
        if paired_manifest is None or paired_manifest.get('split')!='test':
            raise ValueError('Test is fixed to a registered generated corpus, never panel tuning')
        ids=paired_manifest.get('selected_family_ids')
        if ids is None:ids=[r['family_id'] for r in rows]
        if not ids or len(ids)!=len(set(ids)):raise ValueError('Invalid fixed test family population')
        by_id={r['family_id']:r for r in rows}
        if set(ids)-set(by_id):raise ValueError('Generated test families are outside reference test split')
        return [by_id[i] for i in ids]
    if per_stratum<1:raise ValueError('Positive fixed panel size required')
    strata=defaultdict(list)
    for row in rows:strata[(row['cohort'],int(row['receiver']))].append(row)
    result=[]
    for key in sorted(strata):
        ordered=sorted(strata[key],key=lambda r:hashlib.sha256(('reader-clean-panel-v1:'+r['family_id']).encode()).hexdigest())
        result.extend(ordered[:per_stratum])
    if not result:raise ValueError('Empty reference panel')
    return result


def detection_counts(reference,prediction):
    """Class-compatible IoU=.5 matching; empty predictions preserve denominator."""
    matched,_=matched_indices(reference,prediction)
    return dict(reference=len(reference),prediction=len(prediction),matched=len(matched))


def count_summary(counts):
    n,g,p=(sum(c[k] for c in counts) for k in ('matched','reference','prediction'))
    return dict(reference=g,prediction=p,matched=n,recall=None if not g else n/g,
                precision=None if not p else n/p)


def summarize(reference,prediction,events):
    rr,pp=frame_index(reference),frame_index(prediction)
    if set(rr)!=set(pp):raise ValueError('Incomplete frame/view population')
    counts=defaultdict(list)
    for key,ref in rr.items():
        period='history' if key[1]<6 else 'future'
        gt=visible_detections(ref);pred=pp[key]['detections']
        counts[period].append(detection_counts(gt,pred))
        counts[period+'_'+key[2]].append(detection_counts(gt,pred))
        for category in (1,2,3,4):
            counts[period+'_class%d'%category].append(detection_counts(
                [g for g in gt if g['class_id']==category],[d for d in pred if d['class_id']==category]))
    identity=event_identity_metrics(reference,prediction,events,score_from_frame=6)
    established=sum(e['pre_occlusion_id_established'] for e in identity['events'])
    identity.update(pre_occlusion_ID_established=established,
                    R3_conditional_on_pre_ID=None if not established else identity['R3_success']/established,
                    R3_undefined_reason='no_reference_reappearance_events' if not events else None,
                    conditional_R3_undefined_reason='no_pre_occlusion_ID' if not established else None,
                    missing_pre_ID_policy='failure in primary R3; separately counted, never excluded')
    future=lambda rows:[r for r in rows if r['frame']>=6]
    track=evaluate_tracking(future(reference),future(prediction))
    return dict(detection={k:count_summary(v) for k,v in counts.items()},identity=identity,
                tracking_statistics=next(iter(track.values())))


def summarize_cells(cells,arm):
    selected=[c[arm] for c in cells if arm in c]
    if not selected:return None
    result={'cells':len(selected)}
    result['detection']={key:count_summary([x['detection'][key] for x in selected]) for key in selected[0]['detection']}
    keys=('R3_success','R3_events','pre_occlusion_ID_established','jointly_visible_count','jointly_detected_count','identity_agreement_count')
    counts={k:sum(x['identity'][k] for x in selected) for k in keys}
    divide=lambda a,b:None if not b else a/b
    result['identity']=dict(counts,R3=divide(counts['R3_success'],counts['R3_events']),
        pre_occlusion_ID_coverage=divide(counts['pre_occlusion_ID_established'],counts['R3_events']),
        R3_conditional_on_pre_ID=divide(counts['R3_success'],counts['pre_occlusion_ID_established']),
        cross_view_ID_agreement=divide(counts['identity_agreement_count'],counts['jointly_detected_count']),
        joint_detection_coverage=divide(counts['jointly_detected_count'],counts['jointly_visible_count']),
        R3_undefined_reason='no_reference_reappearance_events' if not counts['R3_events'] else None,
        conditional_R3_undefined_reason='no_pre_occlusion_ID' if not counts['pre_occlusion_ID_established'] else None)
    tracking=combine_tracking([restore_tracking_arrays(x['tracking_statistics']) for x in selected])
    result['tracking']={k:v for k,v in tracking.items() if k not in ('sufficient_statistics',)}
    return result


def raw_detections(output,view,height,width,confidence):
    """Only network boxes/classes/scores, before RGB floor-ray filtering."""
    import torch
    from experiments.downstream_tracking.model import xyxy
    scores,classes=output['logits'][view].sigmoid().max(-1)
    keep=torch.where(scores>=confidence)[0]
    boxes=(xyxy(output['boxes'][view,keep]).clamp(0,1)*output['boxes'].new_tensor([width,height,width,height])).cpu().tolist()
    return [dict(query_index=q,class_id=int(classes[q])+1,score=float(scores[q]),box_xyxy=b) for q,b in zip(keep.tolist(),boxes)]


def infer_clean(model,config,arrays,meta,refs,diagnostic_threshold=None):
    """One network pass/frame; instrument unchanged production postprocessing."""
    import torch
    from torch import nn
    from experiments.downstream_tracking.data import pack_images,move
    from experiments.downstream_tracking.inference import predict_pair
    from experiments.downstream_tracking.association import CooperativeTracker
    class Replay(nn.Module):
        def __init__(self,out):
            super().__init__();self.anchor=nn.Parameter(torch.empty(0,device=device),requires_grad=False)
            self.output=out;self.modality=model.modality
        def forward(self,**kwargs):return self.output
    device=next(model.parameters()).device
    tracker=CooperativeTracker(config);rgb=arrays['rgb'];height,width=rgb.shape[2:4]
    K=arrays['K'];poses=arrays['camera_to_world'];times=arrays['time_s']
    floor=float(arrays['world_from_factory'][2,3]) if 'world_from_factory' in arrays else 0.
    rr=frame_index(refs);rows=[];stages=defaultdict(list);drops=[];timings=[]
    for t in range(len(rgb)):
        started=time.monotonic();k=K[t] if K.ndim==4 else K
        inputs=dict(rgb=rgb[t:t+1],depth_mm=arrays['depth_mm'][t:t+1],valid=arrays['valid'][t:t+1],K=k)
        with torch.no_grad():
            output=model(**move(pack_images(inputs),device))
            count=output['boxes'].shape[1]
            # Carry query indices through production geometry decoding, then
            # restore the actual learned embedding before association.
            indexed=dict(output,identity=torch.arange(count,device=device,dtype=output['identity'].dtype)[None,:,None].expand(2,-1,-1))
            decoded=predict_pair(Replay(indexed),rgb[t],arrays['depth_mm'][t],arrays['valid'][t],k,poses[t],
                                 tracker.config['confidence'],floor)
            for v,view in enumerate(VIEWS):
                for det in decoded[view]:
                    query=int(det['identity_embedding'][0]);det['query_index']=query
                    det['identity_embedding']=output['identity'][v,query].cpu().tolist()
            tracked=tracker.update(decoded,float(times[t]))
            for v,view in enumerate(VIEWS):
                reference=visible_detections(rr[(meta['episode'],t,view)])
                period='history' if t<6 else 'future'
                raw=raw_detections(output,v,height,width,tracker.config['confidence'])
                after=decoded[view]
                for label,dets in (('raw_at_fixed_association_threshold',raw),('after_geometry_filter',after),('after_tracking',tracked[view])):
                    stages[period+':'+view+':'+label].append(detection_counts(reference,dets))
                if diagnostic_threshold is not None:
                    stages[period+':'+view+':raw_at_registered_validation_gate'].append(
                        detection_counts(reference,raw_detections(output,v,height,width,diagnostic_threshold)))
                surviving={d['query_index'] for d in after}
                for det in raw:
                    if det['query_index'] not in surviving:
                        q=det['query_index'];contact=output['contact'][v,q].cpu().numpy()*[width,height]
                        kv=k[v] if k.ndim==3 else k;ray=poses[t,v,:3,:3]@np.linalg.solve(kv,np.r_[contact,1.])
                        scale=(floor-poses[t,v,2,3])/ray[2] if abs(ray[2])>1e-6 else -1.
                        drops.append(dict(frame=t,view=view,**det,contact_xy=contact.tolist(),ray_z=float(ray[2]),floor_scale=float(scale)))
                packed=[]
                for det in tracked[view]:
                    item=dict(det);item['mask_rle']=mask_rle(item.pop('mask')>=.5);packed.append(item)
                rows.append(dict(meta,frame=t,time_s=float(times[t]),view=view,canvas_size=[height,width],detections=packed))
        timings.append(time.monotonic()-started)
    return rows,dict(stages={k:count_summary(v) for k,v in stages.items()},geometry_dropped_queries=drops,
        frame_pair_seconds=timings,floor_height_m=floor,registered_association_confidence=tracker.config['confidence'],
        registered_validation_confidence=diagnostic_threshold,threshold_selection_performed=False,
        input=dict(rgb_shape=list(rgb.shape),rgb_dtype=str(rgb.dtype),rgb_min=int(rgb.min()),rgb_max=int(rgb.max()),
                   time_s=times.tolist(),camera_shape=list(poses.shape),K_shape=list(K.shape)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('index','checkpoint','assets','association','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--split',choices=('train','val','test'),default='val')
    p.add_argument('--panel-per-stratum',type=int,default=1)
    p.add_argument('--paired-predictions');p.add_argument('--paired-readouts')
    p.add_argument('--family-panel',help='Immutable availability-only partial test panel; requires generated pixel rerun')
    p.add_argument('--rerun-generated',action='store_true',help='Re-read retained draw zero using this same reader implementation')
    p.add_argument('--device',default='cpu');p.add_argument('--gpu-hours',type=float,default=.25)
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    from .runtime import Budget
    import torch
    from experiments.downstream_tracking.inference import load_reader,reader_implementation
    budget=Budget(a.gpu_hours);torch.set_num_threads(a.threads)
    index=read_json(a.index);association=read_json(a.association)
    current=dict(checkpoint_sha256=sha256(a.checkpoint),association_sha256=sha256(a.association),**reader_implementation())
    pair,previous_readout=paired_source_policy(a.paired_predictions,a.paired_readouts,a.rerun_generated,current,sha256(a.index))
    families=select_families(index,a.split,a.panel_per_stratum,pair)
    registered_cells=4*len(families);panel=None
    if a.family_panel:
        from .reader_availability_panel import validate_panel
        panel=validate_panel(a.family_panel,a.paired_predictions,a.index,a.split,a.rerun_generated)
        wanted=set(panel['selected_family_ids']);families=[f for f in families if f['family_id'] in wanted]
        if [f['family_id'] for f in families]!=panel['selected_family_ids']:raise ValueError('Panel order differs from original registration')
    if association.get('split')!='val' or association.get('checkpoint_sha256')!=sha256(a.checkpoint):raise ValueError('Require checkpoint-matched validation association')
    model,saved=load_reader(a.checkpoint,a.assets,a.device,required_role='video_evaluation')
    config=association.get('association',association.get('parameters'))
    if not isinstance(config,dict):raise ValueError('Missing frozen association parameters')
    threshold=saved['identity'].get('frontend_gate',{}).get('confidence') if a.split!='test' else None
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    contract=dict(schema=SCHEMA,diagnostic_only=True,clean_reference_control=True,split=a.split,
        index_sha256=sha256(a.index),checkpoint_sha256=sha256(a.checkpoint),association_sha256=sha256(a.association),
        source_sha256={str(path.relative_to(Path(__file__).resolve().parents[2])):sha256(path)
            for path in [*(Path(__file__).resolve().with_name(name) for name in ('clean_reference.py','tracking.py','metrics.py','readout.py','common.py','runtime.py')),
                         *(Path(__file__).resolve().parent.parent/'downstream_tracking'/name for name in ('inference.py','association.py','model.py','data.py'))]},
        family_ids=[f['family_id'] for f in families],generated_draw=0 if pair else None,
        generated_manifest_sha256=None if pair is None else sha256(Path(a.paired_predictions)/'manifest.json'),
        previous_readout_provenance=previous_readout,
        partial_corpus_diagnostic=panel is not None,registered_total_cells=registered_cells,
        availability_panel=None if panel is None else dict(path=str(Path(a.family_panel).resolve()),sha256=sha256(a.family_panel),
            source_sha256=sha256(Path(__file__).with_name('reader_availability_panel.py')),snapshot=panel),
        network_inputs='actual rendered/generated sensor pixels and calibration only; labels never enter the model or tracker',
        reference_used_for_scoring_only=True,threshold_selection_performed=False,
        reader_training_identity=saved['identity'],reader_training_sources=saved.get('source_hashes'),
        inference_only_amendment=True,formal_training_source_checks_relaxed=False,
        R3_protocol=dict(history_frames=6,future_frames=16,require_three_reference_visible_frames=True,
            history_anchor=5,effectively_invisible_at_anchor=True,projected_area_min_pixels=144,
            pre_ID_missing='primary failure, conditional metric undefined when all pre-IDs absent',
            no_reference_event='R3 undefined; fully visible/no reappearance/censored last-two-frame events are not eligible'),
        generated_readout_policy='rerun_retained_draw0_pixels' if a.rerun_generated else 'existing_readout',
        **reader_implementation())
    if (output/'manifest.json').exists() and read_json(output/'manifest.json')!=contract:raise ValueError('Use a new versioned diagnostic output; existing provenance differs')
    write_json(output/'manifest.json',contract);cells=[]
    def publish(complete):
        write_json(output/'report.json',dict(contract,complete=complete,cells=cells,
            clean=summarize_cells(cells,'clean'),generated=summarize_cells(cells,'generated'),
            elapsed_seconds=budget.elapsed_hours()*3600))
        write_json(output/'completion.json',dict(completed=complete,cells=len(cells),expected_cells=4*len(families),diagnostic_only=True,
            partial_corpus_diagnostic=panel is not None,registered_total_cells=registered_cells,formal_corpus_completed=False))
    for family in families:
        for branch in family['records']:
            w,c=branch['world'],branch['camera'];episode='%s_w%d_c%d'%(family['family_id'],w,c)
            folder=output/'cells'/episode;cache=folder/'metrics.json'
            if cache.exists():
                cached=read_json(cache)
                if cached.get('clean_prediction_sha256')!=sha256(folder/'prediction.jsonl') or cached.get('reference_sha256')!=branch['sha256']:
                    raise ValueError('Changed cached independent diagnostic or reference')
                if a.rerun_generated and cached.get('generated_prediction_sha256')!=sha256(folder/'generated_prediction.jsonl'):
                    raise ValueError('Changed cached generated diagnostic')
                cells.append(cached);continue
            if budget.expired():publish(False);return
            path=Path(index['root'])/branch['path']
            if sha256(path)!=branch['sha256']:raise ValueError('Changed audited reference branch')
            with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
            meta=metadata(family,episode);refs,events=native_reference(arrays,meta)
            predictions,trace=infer_clean(model,config,arrays,meta,refs,threshold)
            write_jsonl(folder/'prediction.jsonl',predictions)
            record=dict(family_id=family['family_id'],world=w,camera=c,reference_sha256=branch['sha256'],
                clean_prediction_sha256=sha256(folder/'prediction.jsonl'),clean=summarize(refs,predictions,events),trace=trace)
            if pair:
                if a.rerun_generated:
                    source=Path(a.paired_predictions)/family['family_id']/('world%d_camera%d_sample00.npz'%(w,c))
                    marker=read_json(source.parent/'draw_00.json')
                    if marker['files'][source.name]!=sha256(source):raise ValueError('Changed retained generated pixels')
                    with np.load(source,allow_pickle=False) as z:generated={k:z[k] for k in z.files}
                    gen_rows,gen_trace=infer_generated(model,config,arrays,generated,meta,refs,threshold)
                    write_jsonl(folder/'generated_prediction.jsonl',gen_rows)
                    record.update(generated_trace=gen_trace,generated_source_sha256=sha256(source),
                                  generated_prediction_sha256=sha256(folder/'generated_prediction.jsonl'))
                else:
                    source=Path(a.paired_readouts)/family['family_id']/('prediction_w%d_c%d_s00.jsonl'%(w,c))
                    marker=read_json(source.with_suffix('.complete.json'))
                    if marker['sha256']!=sha256(source):raise ValueError('Changed generated independent readout')
                    gen_rows=read_jsonl(source)
                record['generated']=summarize(refs,gen_rows,events)
            write_json(cache,record);cells.append(record)
            write_json(output/'status.json',dict(completed_cells=len(cells),total_cells=4*len(families),latest=episode,
                pair_forward_seconds=float(np.mean(trace['frame_pair_seconds'])),elapsed_seconds=budget.elapsed_hours()*3600))
            print(json.dumps(dict(cells=len(cells),episode=episode,clean_future_recall=record['clean']['detection']['future']['recall'],
                mean_frame_pair_seconds=float(np.mean(trace['frame_pair_seconds'])))),flush=True)
    publish(True)


if __name__=='__main__':main()
