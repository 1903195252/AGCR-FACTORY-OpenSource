"""CPU full-corpus official ranked-box AP, separate from fixed-gate tracking.

Native shared-frontend caches are evaluated once for all identity heads. Video
AP is computed over each complete draw, then averaged over the eight draws.
Missing sidecars never become an available-subset result. No model is loaded.
"""
import argparse
import contextlib
import io
import signal
import time
from pathlib import Path
import numpy as np

from .common import read_json,read_jsonl,write_json,sha256,SAMPLE_SEEDS,VIEWS
from .ranked_boxes import load_sidecar,ranked_records,ranking_contract,ranked_coco_map
from .tracking import visible_detections


def slim_reference(rows,start,frames,episode):
    expected={(episode,t,v) for t in range(start,frames) for v in VIEWS}
    selected=[r for r in rows if int(r['frame'])>=start]
    actual={(str(r['episode']),int(r['frame']),r['view']) for r in selected}
    if actual!=expected or len(selected)!=len(expected):raise ValueError('Reference future/full-episode frame population differs')
    slim=[]
    for r in selected:
        det=[dict(class_id=d['class_id'],box_xyxy=d['box_xyxy'],present=True,visible=True)
             for d in visible_detections(r)]
        slim.append(dict(episode=r['episode'],frame=r['frame'],view=r['view'],canvas_size=r['canvas_size'],detections=det))
    return slim


def native_tasks(args):
    index=read_json(args.index);root=Path(args.cache);manifest=read_json(root/'manifest.json')
    episodes=sorted({r['episode'] for r in index['windows'] if r['role']=='test'})
    amendment=Path(args.amendment_root);contract=read_json(amendment/'amendment.json');done=read_json(amendment/'reports_completion.json')
    if (len(episodes)!=250 or manifest.get('episodes')!=episodes or manifest.get('frames_per_episode')!=100 or
            manifest.get('index_sha256')!=sha256(args.index) or contract.get('index_sha256')!=sha256(args.index) or
            contract.get('episodes')!=episodes or Path(contract['cache_root']).resolve()!=root.resolve() or
            contract.get('frozen_frontend_sha256')!=manifest.get('frozen_frontend_sha256') or
            done.get('completed') is not True or done.get('episodes')!=250 or
            done.get('amendment_sha256')!=sha256(amendment/'amendment.json')):
        raise ValueError('Complete registered shared-frontend native corpus and reference reports required')
    if Path(args.reference_root).resolve() not in {Path(r['output']).resolve() for r in contract['readers'].values()}:
        raise ValueError('Reference rows must belong to the committed amended reports')
    jobs=[]
    for episode in episodes:
        folder=root/episode;marker=read_json(folder/'features.json');sidecar=folder/'ranked_boxes.npz'
        if marker.get('cache_contract_sha256')!=sha256(root/'manifest.json') or marker.get('ranked_sha256')!=sha256(sidecar):
            raise ValueError('Uncommitted or changed complete-query cache')
        source=Path(index['dataset_root'])/'episodes'/episode
        names={'frames.h5'}|{'rgb/%s_%04d.jpg'%(view,t) for t in range(100) for view in ('CAM_AIR','CAM_GROUND')}
        if set(marker['native_source_sha256'])!=names:raise ValueError('Native cache must bind H5 and all 200 RGB images')
        for name,digest in marker['native_source_sha256'].items():
            if sha256(source/name)!=digest:raise ValueError('Native RGB/reference/calibration changed: '+name)
        jobs.append(dict(episode=episode,sidecar=str(sidecar),reference=str(Path(args.reference_root)/episode/'reference.jsonl'),
            frames=100,score_from_frame=0,metadata_checks=dict(episode=episode,cache_contract_sha256=sha256(root/'manifest.json'),
                frozen_frontend_sha256=manifest['frozen_frontend_sha256'],frames=100,views=2,score_from_frame=0,reference_inputs_used=False)))
    return {0:jobs},dict(scope='shared_native_frontend',shared_across_identity_heads=True,episodes=episodes,
        cache_manifest_sha256=sha256(root/'manifest.json'),amendment_sha256=sha256(amendment/'amendment.json'),
        frozen_frontend_sha256=manifest['frozen_frontend_sha256'],draw_seeds=None)


def video_tasks(args):
    index=read_json(args.index);root=Path(args.readouts);manifest=read_json(root/'manifest.json')
    contract=read_json(root/'amendment.json');done=read_json(root/'amendment_completion.json');ranked=read_json(root/'ranked_completion.json')
    families=[f for f in index['families'] if f['split']=='test'];expected=len(families)*4*len(SAMPLE_SEEDS)
    if (not families or contract.get('index_sha256')!=sha256(args.index) or manifest.get('split')!='test' or
            done.get('completed') is not True or done.get('full_eight_draws') is not True or
            done.get('amendment_sha256')!=sha256(root/'amendment.json') or
            ranked.get('completed') is not True or ranked.get('clips')!=expected or ranked.get('frames_per_clip')!=22):
        raise ValueError('Complete original four-cell/eight-draw ranked corpus required, never partial diagnostics')
    jobs={d:[] for d in range(len(SAMPLE_SEEDS))}
    for family in families:
        if len(family['records'])!=4 or {(r['world'],r['camera']) for r in family['records']}!={(0,0),(0,1),(1,0),(1,1)}:
            raise ValueError('Original family must contain exactly four world/camera cells')
        fid=family['family_id'];folder=root/fid
        for record in family['records']:
            w,c=record['world'],record['camera'];episode='%s_w%d_c%d'%(fid,w,c)
            for draw,seed in enumerate(SAMPLE_SEEDS):
                prediction=folder/('prediction_w%d_c%d_s%02d.jsonl'%(w,c,draw));commit=read_json(prediction.with_suffix('.complete.json'))
                if commit.get('sha256')!=sha256(prediction):raise ValueError('Changed committed fixed-threshold readout')
                jobs[draw].append(dict(episode=episode,sidecar=str(folder/('ranked_w%d_c%d_s%02d.npz'%(w,c,draw))),
                    reference=str(folder/('reference_w%d_c%d.jsonl'%(w,c))),frames=22,score_from_frame=6,
                    metadata_checks=dict(family_id=fid,episode=episode,world=w,camera=c,draw=draw,seed=seed,
                        native_history_sha256=record['sha256'],generated_source_sha256=commit['source_sha256'],
                        checkpoint_sha256=manifest['checkpoint_sha256'],association_sha256=manifest['association_sha256'],
                        index_sha256=sha256(args.index),score_from_frame=6,frames=22,views=2,tracking_output_unchanged=True)))
    return jobs,dict(scope='full_eight_draw_video',shared_across_identity_heads=False,families=[f['family_id'] for f in families],
        reader_manifest_sha256=sha256(root/'manifest.json'),amendment_sha256=sha256(root/'amendment.json'),draw_seeds=list(SAMPLE_SEEDS))


def prepare_segment(job):
    arrays,marker=load_sidecar(job['sidecar'])
    for key,value in job['metadata_checks'].items():
        if marker['metadata'].get(key)!=value:raise ValueError('Sidecar population/provenance mismatch: '+key)
    refs=read_jsonl(job['reference']);episode=job['episode'];frames=job['frames'];start=job['score_from_frame']
    if len(arrays['boxes'])!=frames:raise ValueError('Incomplete raw frame sequence')
    timestamps={}
    for row in refs:
        if row['episode']!=episode:raise ValueError('Reference episode mismatch')
        t=int(row['frame']);value=float(row['time_s'])
        if t in timestamps and timestamps[t]!=value:raise ValueError('Unsynchronized reference timestamps')
        timestamps[t]=value
    if set(timestamps)!=set(range(frames)):raise ValueError('Missing reference timestamps')
    times=np.asarray([timestamps[t] for t in range(frames)])
    ref=slim_reference(refs,start,frames,episode)
    predictions=ranked_records(arrays['boxes'],arrays['logits'],dict(episode=episode),times,arrays['canvas_size'],start)
    return ref,predictions


def compute_coco(reference,prediction):
    # Reuse the existing official COCO implementation. Compact the prediction
    # rows before its temporary COCO-format conversion; masks, IDs and states
    # never occupy this full-corpus ranking population.
    for row in prediction:
        for d in row['detections']:d.pop('query_index',None)
    with contextlib.redirect_stdout(io.StringIO()):result=ranked_coco_map(reference,prediction)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    for name in ('native','video'):
        parser=sub.add_parser(name)
        for key in ('index','output'):parser.add_argument('--'+key,required=True)
        parser.add_argument('--cpu-hours',type=float,default=24.)
        parser.add_argument('--stop-file')
        for key in (('cache','reference-root','amendment-root') if name=='native' else ('readouts',)):
            parser.add_argument('--'+key,required=True)
    args=p.parse_args()
    if args.cpu_hours<=0:raise ValueError('Positive CPU wall-time budget required')
    import torch
    torch.set_num_threads(4);start=time.monotonic();stopping=[False]
    signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True));signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    expired=lambda:stopping[0] or time.monotonic()-start>=args.cpu_hours*3600 or bool(args.stop_file and Path(args.stop_file).exists())
    jobs,scope=(native_tasks(args) if args.mode=='native' else video_tasks(args));out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    # Freeze the complete source population before any score is calculated.
    sources={p:sha256(p) for tasks in jobs.values() for t in tasks for p in (t['sidecar'],str(Path(t['sidecar']).with_suffix('.complete.json')),t['reference'])}
    contract=dict(schema='agcr-complete-ranked-coco-1',index_sha256=sha256(args.index),ranking=ranking_contract(),source_sha256=sha256(__file__),
        source_files=sources,expected_segments={str(k):len(v) for k,v in jobs.items()},**scope)
    if (out/'manifest.json').exists() and read_json(out/'manifest.json')!=contract:raise ValueError('Changed ranked corpus; use a new output')
    write_json(out/'manifest.json',contract);scores=[]
    def finish(complete):
        write_json(out/'completion.json',dict(completed=complete,cpu_only=True,no_image_inference=True,draws=len(scores),expected_draws=len(jobs),
            report_sha256=sha256(out/'report.json') if complete else None,manifest_sha256=sha256(out/'manifest.json')))
    finish(False)
    for draw,tasks in jobs.items():
        result=out/('draw_%02d.json'%draw)
        if result.exists():
            prior=read_json(result)
            if prior.get('manifest_sha256')!=sha256(out/'manifest.json') or prior.get('segments')!=len(tasks):raise ValueError('Changed ranked draw cache')
            scores.append(prior);continue
        refs=[];preds=[]
        for i,job in enumerate(tasks):
            if expired():finish(False);return
            r,q=prepare_segment(job);refs.extend(r);preds.extend(q)
            write_json(out/'status.json',dict(state='building_full_corpus',draw=draw,segments=i+1,total_segments=len(tasks),scored_images=len(refs),elapsed_seconds=time.monotonic()-start))
        if expired():finish(False);return
        write_json(out/'status.json',dict(state='official_COCO_evaluation',draw=draw,scored_images=len(refs),elapsed_seconds=time.monotonic()-start))
        score=compute_coco(refs,preds);del refs,preds
        payload=dict(draw=draw,segments=len(tasks),manifest_sha256=sha256(out/'manifest.json'),metrics=score)
        write_json(result,payload);scores.append(payload)
    means={k:None if any(s['metrics'][k] is None for s in scores) else float(np.mean([s['metrics'][k] for s in scores])) for k in ('mAP_50_95','AP50')}
    write_json(out/'report.json',dict(contract,complete=True,cpu_only=True,new_image_inference=False,per_draw=scores,mean=means,
        confidence_threshold_selection_performed=False,old_tracking_outputs_changed=False,elapsed_seconds=time.monotonic()-start,
        status_of_confidence_curves='not_computed_by_this_AP_entrypoint',
        interpretation='Official score-ranked reference-box agreement. It remains affected by world-future prediction and reader-domain error; fixed-threshold tracking is reported separately.'))
    finish(True)


if __name__=='__main__':main()
