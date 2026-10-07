"""Formal unfiltered generation evaluation; all eight samples retain failures."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import read_json,read_jsonl,write_json,sha256,prediction_manifest,require_reader_provenance,require_completed_corpus,VIEWS,NATIVE_CLASSES
from .metrics import aligned_state,depth_all,reference_correspondence_error,visibility_groups,load_detection_mask,ratio,exit_assignment,paired_camera_error,projected_exit_assignment
from .metrics import branch_diagnostics,evidence_effect,reappearance_timing,transition_errors,occlusion_violations
from .tracking import historical_identity_map,event_identity_metrics,evaluate_tracking,combine_tracking,coco_map
from .readout import metadata,evaluation_geometry,CALIBRATION_SOURCE


def region_key(family,native):
    k=family['target_index'];c=NATIVE_CLASSES[int(native['class_id'][k])]
    return '%s_class%d_receiver%d'%(family['cohort'],c,family['receiver'])


def corpus_detection_scores(readouts,families,draws):
    """COCO AP on complete future-frame corpora, one corpus per fixed draw.

    Keep only detector fields in memory; pixel masks and internal states cannot
    influence this metric. Conditioning history never enters the AP population.
    """
    def boxes(path):
        result=[]
        for row in read_jsonl(path):
            if row['frame']<6:continue
            result.append(dict(episode=row['episode'],frame=row['frame'],view=row['view'],canvas_size=row['canvas_size'],
                detections=[{k:v for k,v in d.items() if k in ('track_id','class_id','box_xyxy','score','visible','present')} for d in row['detections']]))
        return result
    refs=[];sources=[]
    for family in families:
        folder=Path(readouts)/family['family_id']
        for branch in family['records']:
            w,c=branch['world'],branch['camera'];refs.extend(boxes(folder/('reference_w%d_c%d.jsonl'%(w,c))))
            sources.append((folder,w,c))
    scores=[]
    for draw in range(draws):
        preds=[]
        for folder,w,c in sources:preds.extend(boxes(folder/('prediction_w%d_c%d_s%02d.jsonl'%(w,c,draw))))
        scores.append(dict(sample_index=draw,**coco_map(refs,preds)))
    return dict(per_draw=scores,mean={key:None if any(s[key] is None for s in scores) else float(np.mean([s[key] for s in scores])) for key in ('mAP_50_95','AP50')},
        population='all reference-visible instances in 16 future frames, all four cells; complete corpus per draw')


def aligned_visual(reference,prediction,native,base):
    """Identity map uses six conditioning frames only and remains fixed."""
    from .common import frame_index
    from .tracking import historical_identity_map
    from .metrics import metric_position
    ref,pred=frame_index(reference),frame_index(prediction)
    if set(ref)!=set(pred):raise ValueError('Reader frame export incomplete')
    mapping=historical_identity_map(reference,prediction);T,V,H,W=native['instance'].shape;N=len(native['instance_id'])
    state=np.full((T,V,N,11),np.nan);box_centers=np.full((T,V,N,2),np.nan);floor_projection=False;present=np.zeros((T,V,N),bool);masks=np.zeros((T,V,N,H,W),bool);embedding={}
    by_id={str(i):k for k,i in enumerate(native['instance_id'])}
    for row in prediction:
        t=int(row['frame']);v=VIEWS.index(row['view']);detections={str(d['track_id']):d for d in row['detections']}
        for gid,pid in mapping.items():
            if gid not in by_id or pid not in detections:continue
            k=by_id[gid];d=detections[pid];s=np.full(11,np.nan)
            box=np.asarray(d['box_xyxy']);box_centers[t,v,k]=(box[:2]+box[2:])/2
            floor_projection|=d.get('localization_scope')=='floor_projection'
            position=metric_position(d)
            if position is not None:s[:3]=position
            if d.get('velocity_world_mps') is not None:s[3:6]=d['velocity_world_mps']
            if d.get('yaw_rad') is not None:s[6]=d['yaw_rad']
            s[7]=0.
            if d.get('dimensions_m') is not None:s[8:11]=d['dimensions_m']
            state[t,v,k]=s;present[t,v,k]=True;masks[t,v,k]=load_detection_mask(d,base,(H,W))
            if 'identity_embedding' in d:embedding[t,v,k]=np.asarray(d['identity_embedding'],float)
    return dict(state=state,present=present,masks=masks,embedding=embedding,history_id_map=mapping,box_centers=box_centers,floor_projection=floor_projection)


def visual_metrics(reference,prediction,native,base,events):
    out=aligned_visual(reference,prediction,native,base);present=out['present'];masks=out['masks'];state=out['state']
    rawvis=(native['visible_fraction']>=.1)&(native['projection_area']>=144)&native['present'][:,None,:]
    historical=rawvis[:6].any((0,1));pop=native['present'][6:]&historical[None]
    refmask=native['instance'][:,:,None]==native['instance_id'][None,None,:,None,None]
    full=native['isolated_support'].astype(bool)
    scores={};counts={};per_reference=[]
    for view in range(2):
        p=state[6:,view];available=present[6:,view]&np.isfinite(p[:,:,:3]).all(-1)
        good=pop&available
        scores[VIEWS[view]+'_center_ADE_m']=ratio(np.linalg.norm(p[:,:,:3]-native['world_state'][6:,:,:3],axis=-1)[good].sum(),good.sum())
        scores[VIEWS[view]+'_center_coverage']=ratio(good.sum(),pop.sum())
        final=good[-1];scores[VIEWS[view]+'_center_FDE_m']=ratio(np.linalg.norm(p[-1,:,:3]-native['world_state'][-1,:,:3],axis=-1)[final].sum(),final.sum())
        scores[VIEWS[view]+'_FDE_coverage']=ratio(final.sum(),pop[-1].sum())
        counts[VIEWS[view]+'_reference_state']=int(pop.sum());counts[VIEWS[view]+'_matched_state']=int(good.sum())
        for t,k in np.argwhere(pop):
            per_reference.append(dict(frame=int(t)+6,view=VIEWS[view],reference_id=int(native['instance_id'][k]),
                matched=bool(good[t,k]),center_error_m=float(np.linalg.norm(p[t,k,:3]-native['world_state'][int(t)+6,k,:3])) if good[t,k] else None))
    fullyhidden=(~refmask[6:].any((-1,-2)))&full[6:].any((-1,-2))&historical[None,None,:]
    eligible=full[6:]&fullyhidden[...,None,None]
    scores['false_appearance_pixel_fraction']=ratio((masks[6:]&eligible).sum(),eligible.sum())
    counts['fully_occluded_projection_pixels']=int(eligible.sum());counts['false_appearance_pixels']=int((masks[6:]&eligible).sum())
    scores['false_appearance_frame_rate']=ratio((present[6:]&fullyhidden).sum(),fullyhidden.sum())
    counts['fully_occluded_target_frames']=int(fullyhidden.sum());counts['false_appearance_target_frames']=int((present[6:]&fullyhidden).sum())
    maskpop=np.broadcast_to(pop[:,None,:,None,None],masks[6:].shape)
    intersection=(masks[6:]&refmask[6:]&maskpop).sum();union=((masks[6:]|refmask[6:])&maskpop).sum()
    scores['visible_mask_IoU']=ratio(intersection,union);counts.update(mask_intersection=int(intersection),mask_union=int(union))
    # For fraction error the reference isolated projection is the fixed target
    # support. This is a diagnostic ratio, not a simulator-informed prediction.
    fraction=(masks&full).sum((-1,-2))/np.maximum(native['projection_area'],1)
    predicted_effective=(fraction>=.1)&(native['projection_area']>=144)&present
    vp=np.broadcast_to(pop[:,None,:],predicted_effective[6:].shape)
    from .metrics import binary_counts,binary_scores
    vc=binary_counts(predicted_effective[6:],rawvis[6:],vp);vs=binary_scores(vc)
    scores['visibility_F1']=vs['F1'];scores['visible_fraction_MAE']=ratio(np.abs(fraction[6:]-native['visible_fraction'][6:])[vp].sum(),vp.sum())
    counts.update({'visibility_'+k:v for k,v in vc.items()})
    evrows=[]
    for event in events:
        v=VIEWS.index(event['view']);k=int(event['target']);h=int(event['reappearance']);start=int(event['hidden_start'])
        if h<6 or not historical[k]:continue
        # Retain the first three-frame predicted run AFTER the same reference
        # hidden boundary, including early appearances during occlusion.
        timing=reappearance_timing(rawvis[:,v,k],predicted_effective[:,v,k],start=start,dt=.2)
        if timing['missing']:timing['absolute_seconds']=3.2
        early=transition_errors(rawvis[:,v,k],predicted_effective[:,v,k],h,'reappearance')
        late=transition_errors(rawvis[:,v,k],predicted_effective[:,v,k],start,'disappearance')
        evrows.append(dict(event,**timing,early_frames=early['false_visible_frames'],early_denominator=early['reference_hidden_frames'],
            late_frames=late['false_visible_frames'],late_denominator=late['reference_hidden_frames']))
    scores['reappearance_absolute_seconds']=None if not evrows else float(np.mean([e['absolute_seconds'] for e in evrows]))
    signed=[e['signed_seconds'] for e in evrows if e['signed_seconds'] is not None]
    scores['reappearance_signed_seconds']=None if not signed else float(np.mean(signed))
    scores['missing_reappearance_rate']=ratio(sum(e['missing'] for e in evrows),len(evrows));counts['reappearance_events']=len(evrows)
    for name in ('early','late'):
        counts[name+'_false_frames']=sum(e[name+'_frames'] for e in evrows);counts[name+'_reference_hidden_frames']=sum(e[name+'_denominator'] for e in evrows)
        scores[name+'_false_visible_fraction']=ratio(counts[name+'_false_frames'],counts[name+'_reference_hidden_frames'])
    identities=event_identity_metrics(reference,prediction,events,score_from_frame=6);scores.update({k:identities[k] for k in ('R3','cross_view_ID_agreement','joint_detection_coverage','cross_view_center_error_m')})
    counts.update({k:identities[k] for k in ('R3_success','R3_events','jointly_visible_count','jointly_detected_count','identity_agreement_count')})
    strata={}
    for name,mask in visibility_groups(rawvis[6:]).items():
        chosen=pop&mask
        values={}
        for v,view in enumerate(VIEWS):
            good=chosen&present[6:,v]&np.isfinite(state[6:,v,:,:3]).all(-1)
            values[view]=dict(ADE_m=ratio(np.linalg.norm(state[6:,v,:,:3]-native['world_state'][6:,:,:3],axis=-1)[good].sum(),good.sum()),
                coverage=ratio(good.sum(),chosen.sum()),reference_count=int(chosen.sum()),matched_count=int(good.sum()))
        strata[name]=values
    return dict(metrics=scores,counts=counts,events=evrows,visibility_strata=strata,history_id_map=out['history_id_map'],per_reference_centers=per_reference),out


def internal_state_metrics(generated,native):
    ids=np.asarray(generated['historical_reference_ids']);state=np.asarray(generated['internal_state']);p=np.asarray(generated['state_present'])
    N=len(native['instance_id']);aligned=np.full((16,N,11),np.nan);presence=np.zeros((16,N),bool)
    # historical_reference_ids are reference *indices* matched exclusively in
    # history by the generator export. They do not establish visual agreement.
    for k,reference in enumerate(ids.tolist()):
        if reference>=0:
            if reference>=N:raise ValueError('Invalid historical reference index')
            aligned[:,reference]=state[:,k];presence[:,reference]=p[:,k]
    vis=(native['visible_fraction']>=.1)&(native['projection_area']>=144)
    return aligned_state(aligned,presence,native['world_state'][6:],native['present'][6:],vis[:6].any((0,1)),vis[6:]),aligned,presence


def evaluate(index_path,predictions,readouts,regions_path,output,assets=None,device='cpu',development=False,limit=None,gpu_hours=24.,allow_oracle_diagnostics=False):
    from .runtime import Budget
    budget=Budget(gpu_hours)
    index=read_json(index_path);manifest=prediction_manifest(predictions,development,allow_oracle_diagnostics);reader=read_json(Path(readouts)/'manifest.json')
    diagnostic=bool(manifest.get('diagnostic_only'))
    schema='agcr-diagnostic-evaluation-1' if diagnostic else 'agcr-formal-evaluation-1'
    if diagnostic and (reader.get('diagnostic_only') is not True or reader.get('generator_conditions')!=manifest['conditions']):
        raise ValueError('Independent reader must preserve diagnostic condition provenance')
    require_reader_provenance(reader,development=development)
    if not development and reader.get('calibration_source')!=CALIBRATION_SOURCE:raise ValueError('Formal visual readouts require fixed audited reference camera calibration')
    if manifest['index_sha256']!=sha256(index_path) or reader['generator_manifest_sha256']!=sha256(Path(predictions)/'manifest.json'):raise ValueError('Evaluation provenance mismatch')
    regions=read_json(regions_path)
    if regions.get('split')!='val' or not regions.get('frozen'):raise ValueError('Exit regions must be frozen from validation, never test')
    if not development:
        require_completed_corpus(predictions);require_completed_corpus(readouts)
    if not development and assets is None:raise ValueError('Formal report requires verified LPIPS/I3D assets')
    video=None
    if assets:
        from .video import VideoMetrics
        video=VideoMetrics(assets,device)
    families=[r for r in index['families'] if r['split']==manifest['split']]
    if limit:
        if not development:raise ValueError('No formal partial-corpus summary')
        families=families[:limit]
    result=[];interventions=[];ref_features=[];gen_features=defaultdict(list)
    cache=Path(str(output)+'.cache');cache.mkdir(parents=True,exist_ok=True)
    cache_identity=dict(index=sha256(index_path),generator_manifest=sha256(Path(predictions)/'manifest.json'),reader_manifest=sha256(Path(readouts)/'manifest.json'),
        regions=sha256(regions_path),source={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},video_assets=None if assets is None else sha256(Path(assets)/'assets.json'))
    if (cache/'identity.json').exists() and read_json(cache/'identity.json')!=cache_identity:raise ValueError('Evaluation cache provenance changed; use a new output path')
    write_json(cache/'identity.json',cache_identity)
    for family in families:
        f=family['family_id'];cached=cache/(f+'.json')
        if cached.exists():
            saved=read_json(cached);result.extend(saved['rows']);interventions.extend(saved['interventions'])
            if video:
                with np.load(cache/(f+'.features.npz'),allow_pickle=False) as feature:
                    ref_features.extend(feature['reference'])
                    for draw in range(len(manifest['sample_seeds'])):gen_features[draw].extend(feature['generated'][draw])
            continue
        if budget.expired():
            write_json(str(output)+'.status.json',dict(completed_families=len(result)//4,budget_paused=True,elapsed_gpu_hours=budget.elapsed_hours()));return None
        row_start=len(result);diagnostic_start=len(interventions);feature_start=len(ref_features)
        records={};native_branches={};visuals={};states={};assignments={'internal':{},'visual':{},'visual_2d_exit':{}}
        for branch in family['records']:
            w,c=branch['world'],branch['camera'];ep='%s_w%d_c%d'%(f,w,c)
            native_path=Path(index['root'])/branch['path']
            if sha256(native_path)!=branch['sha256']:raise ValueError('Changed audited native branch')
            with np.load(native_path,allow_pickle=False) as z:native={k:z[k] for k in z.files}
            native_branches[w,c]=native;refs=read_jsonl(Path(readouts)/f/('reference_w%d_c%d.jsonl'%(w,c)))
            events=read_json(Path(readouts)/f/('events_w%d_c%d.json'%(w,c)));key=region_key(family,native)
            if key not in regions['regions']:raise ValueError('No validation-fixed exit regions for '+key)
            target=int(family['target_index']);history=((native['visible_fraction'][:6]>=.1)&(native['projection_area'][:6]>=144)).any((0,1))
            samples=[];assignments['internal'][w,c]=[];assignments['visual'][w,c]=[];assignments['visual_2d_exit'][w,c]=[]
            if video:ref_features.append(video.features(native['rgb'][6:].transpose(1,0,2,3,4)))
            for draw,seed in enumerate(manifest['sample_seeds']):
                source=Path(predictions)/f/('world%d_camera%d_sample%02d.npz'%(w,c,draw));predpath=Path(readouts)/f/('prediction_w%d_c%d_s%02d.jsonl'%(w,c,draw))
                marker=read_json(predpath.with_suffix('.complete.json'))
                if marker['sha256']!=sha256(predpath) or marker['source_sha256']!=sha256(source):raise ValueError('Changed independent prediction or generated sample')
                with np.load(source,allow_pickle=False) as z:generated={k:z[k] for k in z.files}
                measured_K,measured_poses=evaluation_geometry(generated,native['rgb'].shape,native['K'],native['camera_to_world'],native['time_s'])
                preds=read_jsonl(predpath);visual,out=visual_metrics(refs,preds,native,predpath.parent,events);visuals[w,c,draw]=out
                scores=dict(visual['metrics']);depth=None;correspondences=None
                if 'depth_mm' in generated:
                    depth=depth_all(generated['depth_mm'],generated['validity_probability'],native['depth_mm'][6:],native['valid'][6:])
                    scores.update({k:v for k,v in depth.items() if k not in ('sums','validity_counts')})
                    correspondences=reference_correspondence_error(generated['depth_mm'],native['depth_mm'][6:],native['valid'][6:],measured_K[6:],measured_poses[6:])
                    scores['surface_correspondence_error_m']=correspondences['surface_correspondence_error_m']
                internal=None
                if 'internal_state' in generated:
                    internal,aligned,presence=internal_state_metrics(generated,native);states[w,c,draw]=(aligned,presence)
                    assignments['internal'][w,c].append(exit_assignment(aligned[-1,target,:3] if presence[-1,target] else None,regions['regions'][key]))
                else:
                    assignments['internal'][w,c].append(None)
                    if 'per_view_internal_state' in generated:
                        estimates=[];presence_views=[];internal=dict(scope='independently sampled per-view internal states',shared_state_provided=False)
                        for vi,view in enumerate(VIEWS):
                            adapted=dict(generated,internal_state=generated['per_view_internal_state'][:,vi],state_present=generated['per_view_state_present'][:,vi])
                            internal[view],estimate,exists=internal_state_metrics(adapted,native)
                            estimates.append(estimate);presence_views.append(exists)
                        joint=presence_views[0]&presence_views[1]&history[None]
                        internal['jointly_present_target_frames']=int(joint.sum())
                        internal['reference_eligible_target_frames']=int(history.sum()*16)
                        internal['cross_view_center_disagreement_m']=None if not joint.any() else float(np.linalg.norm(estimates[0][...,:3]-estimates[1][...,:3],axis=-1)[joint].mean())
                receiver=int(family['receiver']);vstate=out['state'][-1,receiver,target,:3]
                rgb_only=reader['reader_identity']['modality']=='rgb'
                assignments['visual'][w,c].append(None if rgb_only else exit_assignment(vstate if out['present'][-1,receiver,target] else None,regions['regions'][key]))
                pixel=out['box_centers'][-1,receiver,target] if out['present'][-1,receiver,target] else None
                assignments['visual_2d_exit'][w,c].append(projected_exit_assignment(pixel,regions['regions'][key],measured_K[-1,receiver],measured_poses[-1,receiver]))
                if rgb_only:
                    # A floor-contact projection is not a 3-D object center.
                    # Preserve its dedicated scope and do not publish it under
                    # RGB-D center-error or camera-passivity metric names.
                    for name in list(scores):
                        if 'center_' in name or 'FDE_' in name:scores[name]=None
                    for item in visual['per_reference_centers']:item['center_error_m']=None;item['matched']=False
                    for stratum in visual['visibility_strata'].values():
                        for view in stratum.values():view['ADE_m']=None;view['coordinate_scope']='floor_projection_not_3D_center'
                future_refs=[r for r in refs if r['frame']>=6];future_preds=[r for r in preds if r['frame']>=6]
                tracked=list(evaluate_tracking(future_refs,future_preds).values())[0]
                scores.update({k:v for k,v in combine_tracking([tracked]).items() if k not in ('sufficient_statistics','per_view_HOTA')})
                if video:
                    scores.update(video.image_scores(generated['rgb'],native['rgb'][6:]));gen_features[draw].append(video.features(generated['rgb'].transpose(1,0,2,3,4)))
                samples.append(dict(sample_index=draw,sample_seed=seed,metrics=scores,counts=visual['counts'],visibility_strata=visual['visibility_strata'],events=visual['events'],
                    internal_state=internal,per_reference_centers=visual['per_reference_centers'],depth_statistics=None if depth is None else depth['sums'],correspondence_statistics=correspondences,tracking_statistics=tracked))
            metrics={k:None if any(s['metrics'][k] is None for s in samples) else float(np.mean([s['metrics'][k] for s in samples])) for k in samples[0]['metrics']}
            row=dict(metadata(family,ep),unit_id=ep,training_seed=manifest['training_seed'],family_id=f,world=w,camera=c,cohort=family['cohort'],receiver=family['receiver'],
                metrics=metrics,samples=samples,aggregation='mean_of_8' if len(samples)==8 else 'development_mean',best_of_8=None)
            if video and len(samples)==8:
                best=min(samples,key=lambda s:s['metrics']['paired_LPIPS']);row['best_of_8']=dict(sample_index=best['sample_index'],metrics=best['metrics'])
            if all((w,c,s) in states for s in range(len(samples))):
                probability=np.mean([states[w,c,s][1].astype(float) for s in range(len(samples))],axis=0)
                reference_present=native['present'][6:];population=np.broadcast_to(history,reference_present.shape)
                row['internal_presence_Brier']=ratio(np.square(probability-reference_present)[population].sum(),population.sum())
                row['internal_presence_sample_count']=len(samples)
            result.append(row)
        if len(manifest['sample_seeds'])==8:
            diag={}
            for scope in ('internal','visual','visual_2d_exit'):
                a,b=assignments[scope][0,0],assignments[scope][1,0]
                if any(x is None for x in a+b):diag[scope]=dict(status='output_not_provided');continue
                if family['cohort']=='evidence':diag[scope]=evidence_effect(np.array([a]),np.array([b]))
                else:diag[scope]=branch_diagnostics(np.array([a]),np.array([[.5,.5]]))
            camera_results=[]
            for w in range(2):
                v0=(native_branches[w,0]['visible_fraction'][6:]>=.1)&(native_branches[w,0]['projection_area'][6:]>=144)
                v1=(native_branches[w,1]['visible_fraction'][6:]>=.1)&(native_branches[w,1]['projection_area'][6:]>=144)
                for draw in range(8):
                    first=visuals[w,0,draw]['state'][6:,:,:,:3];second=visuals[w,1,draw]['state'][6:,:,:,:3]
                    camera_record=(dict(world=w,sample_index=draw,status='3D_center_not_provided') if reader['reader_identity']['modality']=='rgb' else
                        dict(world=w,sample_index=draw,**paired_camera_error(first,second,v0&v1)))
                    a=visuals[w,0,draw]['embedding'];b=visuals[w,1,draw]['embedding'];dist=[]
                    for t,v,k in np.argwhere(v0&v1):
                        key=(int(t)+6,int(v),int(k))
                        if key in a and key in b:
                            den=np.linalg.norm(a[key])*np.linalg.norm(b[key])
                            if den>0:dist.append(1-float(a[key]@b[key]/den))
                    camera_record.update(identity_cosine_distance=None if not dist else float(np.mean(dist)),identity_matched_count=len(dist))
                    camera_results.append(camera_record)
            interventions.append(dict(metadata(family,f),unit_id=f,training_seed=manifest['training_seed'],cohort=family['cohort'],diagnostics=diag,camera_passivity=camera_results,branch_assignments=assignments))
        if video:
            feature_path=cache/(f+'.features.npz');tmp=feature_path.with_suffix('.npz.tmp')
            with tmp.open('wb') as stream:np.savez(stream,reference=np.stack(ref_features[feature_start:]),generated=np.stack([np.stack(gen_features[d][feature_start:]) for d in range(len(manifest['sample_seeds']))]))
            tmp.replace(feature_path)
        write_json(cached,dict(rows=result[row_start:],interventions=interventions[diagnostic_start:]))
        write_json(str(output)+'.status.json',dict(completed_families=len(result)//4,total_families=len(families),elapsed_gpu_hours=budget.elapsed_hours()))
        write_json(output,dict(schema=schema,diagnostic_only=diagnostic,complete=False,development_only=development,generator_manifest=manifest,reader_manifest=reader,
            regions_sha256=sha256(regions_path),rows=result,interventions=interventions,results_scope='unfiltered independent image readout plus separately labelled internal-state diagnostics'))
    fvd=None
    if video:
        from .video import fvd_eight
        if len(manifest['sample_seeds'])==8 and len(ref_features)>=2:
            fvd=fvd_eight(np.stack(ref_features),np.stack([np.stack(gen_features[s]) for s in range(8)]))
    map_cache=cache/'corpus_coco.json'
    if map_cache.exists():detection_scores=read_json(map_cache)
    else:
        if budget.expired():
            write_json(str(output)+'.status.json',dict(completed_families=len(families),budget_paused=True,pending='corpus_coco'));return None
        detection_scores=corpus_detection_scores(readouts,families,len(manifest['sample_seeds']));write_json(map_cache,detection_scores)
    report=dict(schema=schema,diagnostic_only=diagnostic,complete=True,development_only=development,generator_manifest=manifest,reader_manifest=reader,
        regions_sha256=sha256(regions_path),rows=result,interventions=interventions,FVD=fvd,COCO=detection_scores,video_backend=None if video is None else video.provenance,
        results_scope='unfiltered independent image readout plus separately labelled internal-state diagnostics')
    write_json(output,report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('index','predictions','readouts','regions','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--assets');p.add_argument('--device',default='cpu');p.add_argument('--development',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--gpu-hours',type=float,default=24.)
    p.add_argument('--allow-oracle-diagnostics',action='store_true')
    a=p.parse_args();evaluate(a.index,a.predictions,a.readouts,a.regions,a.output,a.assets,a.device,a.development,a.limit,a.gpu_hours,a.allow_oracle_diagnostics)
if __name__=='__main__':main()
