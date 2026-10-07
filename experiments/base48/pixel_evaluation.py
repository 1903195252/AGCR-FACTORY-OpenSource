"""Independent S2 pixel metrics: bounded LPIPS/I3D pass, then CPU-only report.

The existing metric definitions and every future frame are retained. Reader
outputs, target masks, internal predicted states, and old S1 scores are not
inputs. The output has its own manifest/cache/completion; old results stay intact.
"""
import argparse
from pathlib import Path
import signal
import time
import numpy as np

from experiments.formal_evaluation.common import read_json,write_json,sha256,SAMPLE_SEEDS
from experiments.formal_evaluation.video import VideoMetrics,fvd_eight
from experiments.formal_evaluation.metrics import depth_all,reference_correspondence_error
from experiments.formal_evaluation.readout import evaluation_geometry
from .reader_amendment import complete_pixels,immutable
from .sample import exclusive
from .train import BudgetLedger

SCHEMA='agcr-separate-S2-pixel-evaluation-1'
ORIGINAL_VIDEO_SHA='439b6fd3949900bfea109b2e024ddc618f0c3deac855319a983218b8ce079ef6'


def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:np.savez(f,**arrays)
    tmp.replace(path)


def lpips_only(video,prediction,reference):
    """The unchanged FP32/per-frame LPIPS portion of VideoMetrics.image_scores.

    PSNR/SSIM are computed by the original function on CPU after the GPU pass.
    No batch-size, normalization, resize, autocast, or sampling approximation.
    """
    if prediction.shape!=reference.shape or prediction.ndim!=5 or prediction.shape[1]!=2 or prediction.shape[-1]!=3:
        raise ValueError('Expected matching [16,2,H,W,3] RGB clips')
    torch=video.torch
    p=torch.from_numpy(prediction.copy()).to(video.device).permute(0,1,4,2,3).flatten(0,1).float()/127.5-1
    r=torch.from_numpy(reference.copy()).to(video.device).permute(0,1,4,2,3).flatten(0,1).float()/127.5-1
    with torch.no_grad():values=[video.lpips(p[i:i+1],r[i:i+1]).item() for i in range(len(p))]
    if len(values)!=32 or not np.isfinite(values).all():raise ValueError('Every 16 x 2 future LPIPS distance is required')
    return float(np.mean(values))


def generating_witness(plan_path,generator,predictions):
    """Bind the actual scheduler command, including a refined state checkpoint."""
    plan=read_json(plan_path);job=next((j for j in plan['jobs'] if j['id']=='sample_S2_test_shard0'),None)
    if job is None:raise ValueError('Actual S2 sampling checkpoint lineage missing')
    command=job['command'];checkpoints={};inputs={str(Path(plan_path).resolve()):sha256(plan_path)}
    for flag,key in (('--renderer-checkpoint','renderer_checkpoint_sha256'),('--state-checkpoint','state_checkpoint_sha256')):
        if command.count(flag)!=1:raise ValueError('Ambiguous original generator checkpoint')
        file=Path(command[command.index(flag)+1])
        if sha256(file)!=generator[key]:raise ValueError('Generating checkpoint bytes changed: '+flag)
        checkpoints[str(file)]=generator[key]
    if command.count('--output')!=1 or Path(command[command.index('--output')+1]).resolve()!=Path(predictions).resolve():
        raise ValueError('Scheduler points to another generated corpus')
    for i,flag in enumerate(command[:-1]):
        if flag in ('--protocol','--paths','--refit-protocol','--gate'):
            file=Path(command[i+1]);inputs[str(file)]=sha256(file)
    return checkpoints,inputs


def files_and_assets(a):
    index=read_json(a.index);generator=read_json(Path(a.predictions)/'manifest.json');population=complete_pixels(a.index,a.predictions)
    if generator.get('method')!='base48_evidence' or generator.get('state_identity',{}).get('variant')!='evidence':
        raise ValueError('This independent pixel pass is for the original S2 evidence model only')
    families=[f for f in index['families'] if f['split']=='test']
    if population['verified_source_files']!=len(families)*32:raise ValueError('Complete four-cell/eight-draw source corpus required')
    assets=Path(a.assets);evidence=read_json(assets/'assets.json');witness={str(assets/'assets.json'):sha256(assets/'assets.json')}
    expected={str(assets/'i3d_torchscript.pt'):evidence['i3d']['sha256'],**evidence['lpips']['weights']}
    for path,digest in expected.items():
        if sha256(path)!=digest:raise ValueError('Primary metric weight witness changed')
        witness[path]=digest
    checkpoints,files=generating_witness(a.plan,generator,a.predictions);branches=[]
    for family in families:
        if len(family['records'])!=4 or {(b['world'],b['camera']) for b in family['records']}!={(0,0),(0,1),(1,0),(1,1)}:
            raise ValueError('Each complete family needs exactly four world/camera cells')
        for branch in family['records']:
            native=Path(index['root'])/branch['path']
            if sha256(native)!=branch['sha256']:raise ValueError('Audited native future pixels/depth/calibration changed')
            files[str(native)]=branch['sha256'];fid=family['family_id'];w,c=branch['world'],branch['camera']
            sources=[]
            for draw,seed in enumerate(SAMPLE_SEEDS):
                marker=Path(a.predictions)/fid/('draw_%02d.json'%draw);m=read_json(marker)
                source=marker.parent/('world%d_camera%d_sample%02d.npz'%(w,c,draw))
                if m['seed']!=seed:raise ValueError('Original sample seed changed')
                files[str(marker)]=sha256(marker);files[str(source)]=m['files'][source.name]
                sources.append(str(source))
            branches.append(dict(family_id=fid,world=w,camera=c,unit_id='%s_w%d_c%d'%(fid,w,c),native=str(native),sources=sources))
    return index,generator,population,files,witness,checkpoints,branches


def preflight(a):
    output=Path(a.output).resolve()
    if output!=Path(a.run_root).resolve()/'diagnostics/s2_pixel_metrics/results':
        raise ValueError('Use the isolated diagnostics/s2_pixel_metrics/results destination')
    index,generator,population,files,assets,checkpoints,branches=files_and_assets(a)
    from experiments.formal_evaluation import video,metrics,readout
    from experiments.paper_experiments.agcr_exp import metrics as depth_backend
    if sha256(video.__file__)!=ORIGINAL_VIDEO_SHA:raise ValueError('Original LPIPS/FVD preprocessing changed; reassess this split evaluator')
    contract=dict(schema=SCHEMA,scope='S2_reader_independent_pixel_metrics',index_sha256=sha256(a.index),
        generator_manifest_sha256=sha256(Path(a.predictions)/'manifest.json'),scheduler_plan_sha256=sha256(a.plan),generator_checkpoint_files=checkpoints,
        generator_checkpoint_sha256=generator['checkpoint_sha256'],sample_seeds=list(SAMPLE_SEEDS),population=population,
        sources=files,assets=assets,source_sha256={str(p):sha256(p) for p in map(Path,(__file__,video.__file__,metrics.__file__,readout.__file__,depth_backend.__file__))},
        branches=branches,frames=16,views=2,reader_inputs_used=False,internal_state_inputs_used=False,
        original_outputs_modified=False,old_S1_metrics_used=False,
        execution='FP32 per-frame original LPIPS and original I3D on GPU; original PSNR/SSIM/depth and FVD covariance on CPU')
    if (output/'manifest.json').exists() and read_json(output/'manifest.json')!=contract:raise ValueError('Changed pixel-evaluation identity; cache cannot be reused')
    return output,contract


def read_native(path):
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in ('rgb','depth_mm','valid','K','camera_to_world','time_s')}
    if arrays['rgb'].dtype!=np.uint8 or arrays['rgb'].shape[:2]!=(22,2):raise ValueError('Native RGB must retain 22 paired uint8 frames')
    return arrays


def read_generated(path):
    with np.load(path,allow_pickle=False) as z:
        if not {'rgb','depth_mm','validity_probability','camera_to_world'}<=set(z.files):raise ValueError('S2 paired RGB-D outputs required')
        arrays={k:z[k] for k in ('rgb','depth_mm','validity_probability','camera_to_world')}
    if arrays['rgb'].dtype!=np.uint8 or arrays['rgb'].shape[:2]!=(16,2):raise ValueError('Generated RGB must retain 16 paired uint8 frames')
    return arrays


def cache_paths(root,branch,draw=None):
    folder=Path(root)/'clips'/branch['unit_id']
    path=folder/('reference.npz' if draw is None else 'draw_%02d.npz'%draw)
    return path,path.with_suffix('.json')


def verify_reference(root,branch,contract_sha):
    path,marker=cache_paths(root,branch);m=read_json(marker)
    if (m.get('manifest_sha256')!=contract_sha or m.get('sha256')!=sha256(path) or
            m.get('reference_source_sha256')!=sha256(branch['native'])):raise ValueError('Reference I3D feature identity changed')
    with np.load(path,allow_pickle=False) as z:features=z['features']
    if features.shape!=(2,400) or not np.isfinite(features).all():raise ValueError('Invalid paired reference features')
    return features


def verify_draw(root,branch,draw,contract_sha):
    path,marker=cache_paths(root,branch,draw);m=read_json(marker)
    if (m.get('manifest_sha256')!=contract_sha or m.get('sha256')!=sha256(path) or m.get('sample_seed')!=SAMPLE_SEEDS[draw] or
            m.get('source_sha256')!=sha256(branch['sources'][draw]) or m.get('reference_source_sha256')!=sha256(branch['native'])):
        raise ValueError('Generated LPIPS/I3D feature identity changed')
    with np.load(path,allow_pickle=False) as z:features=z['features'];lpips=float(z['paired_LPIPS'])
    if features.shape!=(2,400) or not np.isfinite(features).all() or not np.isfinite(lpips):raise ValueError('Invalid generated pixel cache')
    return features,lpips


def gpu_pass(a):
    import torch
    torch.set_num_threads(4)
    if a.device!='cuda:0' or not a.budget_ledger:raise ValueError('One allocated GPU and a persisted child ledger are required')
    ledger=BudgetLedger(a.budget_ledger,a.gpu_hours,1);done=False;completed=0;stop=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    expired=lambda:stop[0] or ledger.exhausted() or bool(a.stop_file and Path(a.stop_file).exists())
    output=None;backend_provenance=None
    try:
        output,contract=preflight(a);immutable(output/'manifest.json',contract);signature=sha256(output/'manifest.json')
        with exclusive(output/'.gpu.lock'):
            if expired():return
            backend=VideoMetrics(a.assets,a.device);backend_provenance=backend.provenance
            for branch in contract['branches']:
                if expired():return
                native=None;ref_path,ref_marker=cache_paths(output,branch)
                if ref_marker.exists():verify_reference(output,branch,signature)
                else:
                    native=read_native(branch['native']);features=backend.features(native['rgb'][6:].transpose(1,0,2,3,4))
                    atomic_npz(ref_path,features=features);write_json(ref_marker,dict(manifest_sha256=signature,sha256=sha256(ref_path),
                        reference_source_sha256=contract['sources'][branch['native']]))
                for draw,seed in enumerate(SAMPLE_SEEDS):
                    if expired():return
                    path,marker=cache_paths(output,branch,draw)
                    if marker.exists():verify_draw(output,branch,draw,signature)
                    else:
                        if native is None:native=read_native(branch['native'])
                        generated=read_generated(branch['sources'][draw])
                        evaluation_geometry(generated,native['rgb'].shape,native['K'],native['camera_to_world'],native['time_s'])
                        distance=lpips_only(backend,generated['rgb'],native['rgb'][6:])
                        features=backend.features(generated['rgb'].transpose(1,0,2,3,4))
                        atomic_npz(path,features=features,paired_LPIPS=np.asarray(distance))
                        write_json(marker,dict(manifest_sha256=signature,sha256=sha256(path),sample_seed=seed,
                            source_sha256=contract['sources'][branch['sources'][draw]],reference_source_sha256=contract['sources'][branch['native']]))
                    completed+=1;ledger.commit('S2_pixel_models',completed_clips=completed)
                    write_json(output/'status.json',dict(state='pixel_model_pass',completed_clips=completed,
                        expected_clips=len(contract['branches'])*8,used_gpu_hours=ledger.used))
            done=True
    finally:
        ledger.close('completed' if done else 'stopped')
        if output is not None:
            write_json(output/'gpu_completion.json',dict(completed=done,scope='LPIPS/I3D feature extraction only',completed_clips=completed,
                expected_clips=len(contract['branches'])*8,manifest_sha256=sha256(output/'manifest.json'),
                cpu_pixel_report_completed=False,used_gpu_hours=ledger.used,model_backend=backend_provenance,
                model_execution=dict(device=a.device,LPIPS=True,I3D=True,precision='float32',per_frame_LPIPS=True)))


def cpu_sample(backend,generated,native):
    K,C=evaluation_geometry(generated,native['rgb'].shape,native['K'],native['camera_to_world'],native['time_s'])
    metrics=backend.image_scores(generated['rgb'],native['rgb'][6:]);metrics.pop('paired_LPIPS')
    depth=depth_all(generated['depth_mm'],generated['validity_probability'],native['depth_mm'][6:],native['valid'][6:])
    metrics.update({k:v for k,v in depth.items() if k not in ('sums','validity_counts')})
    corresponding=reference_correspondence_error(generated['depth_mm'],native['depth_mm'][6:],native['valid'][6:],K[6:],C[6:])
    metrics['surface_correspondence_error_m']=corresponding['surface_correspondence_error_m']
    return dict(metrics=metrics,depth_statistics=depth['sums'],validity_counts=depth['validity_counts'],correspondence_statistics=corresponding)


def cpu_report(a):
    import torch
    torch.set_num_threads(4);started=time.monotonic();stop=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    expired=lambda:stop[0] or time.monotonic()-started>=a.cpu_hours*3600 or bool(a.stop_file and Path(a.stop_file).exists())
    output,contract=preflight(a);signature=sha256(output/'manifest.json');done=read_json(output/'gpu_completion.json')
    if done.get('completed') is not True or done.get('manifest_sha256')!=signature or done.get('completed_clips')!=len(contract['branches'])*8:
        raise ValueError('Complete exact LPIPS/I3D population required before CPU reporting')
    if done.get('model_backend')!=read_json(Path(a.assets)/'assets.json') or done.get('model_execution',{}).get('LPIPS') is not True or done.get('model_execution',{}).get('I3D') is not True:
        raise ValueError('Actual completed GPU metric backend provenance is required')
    backend=VideoMetrics(a.assets,'cpu',lpips=False,fvd=False);rows=[];references=[];generated_features=[[] for _ in range(8)]
    write_json(output/'completion.json',dict(completed=False,scope='S2 pixel metrics only',manifest_sha256=signature))
    with exclusive(output/'.report.lock'):
        for branch in contract['branches']:
            if expired():return
            references.append(verify_reference(output,branch,signature));native=None;samples=[]
            for draw,seed in enumerate(SAMPLE_SEEDS):
                if expired():return
                features,lpips=verify_draw(output,branch,draw,signature);generated_features[draw].append(features)
                file=output/'cpu'/branch['unit_id']/('draw_%02d.json'%draw)
                if file.exists():
                    sample=read_json(file)
                    if sample.get('manifest_sha256')!=signature or sample.get('sample_seed')!=seed:raise ValueError('Changed CPU pixel cache')
                else:
                    if native is None:native=read_native(branch['native'])
                    sample=cpu_sample(backend,read_generated(branch['sources'][draw]),native)
                    sample.update(manifest_sha256=signature,sample_index=draw,sample_seed=seed);sample['metrics']['paired_LPIPS']=lpips
                    write_json(file,sample)
                if sample['metrics']['paired_LPIPS']!=lpips:raise ValueError('CPU report LPIPS differs from actual model extraction')
                samples.append(sample)
                write_json(output/'status.json',dict(state='CPU_pixel_report',completed_branches=len(rows),expected_branches=len(contract['branches']),
                    draw=draw,elapsed_seconds=time.monotonic()-started))
            rows.append(dict(unit_id=branch['unit_id'],family_id=branch['family_id'],world=branch['world'],camera=branch['camera'],samples=samples,
                metrics={k:None if any(s['metrics'][k] is None for s in samples) else float(np.mean([s['metrics'][k] for s in samples])) for k in samples[0]['metrics']},aggregation='mean_of_8'))
        fvd=fvd_eight(np.stack(references),np.stack([np.stack(v) for v in generated_features]))
        report=dict(schema=SCHEMA,complete=True,scope='S2 reader-independent pixel metrics only',manifest_sha256=signature,rows=rows,FVD=fvd,
            video_backend=done['model_backend'],model_execution=done['model_execution'],
            cpu_execution=dict(device='cpu',LPIPS_model_loaded=False,I3D_model_loaded=False,source_gpu_completion_sha256=sha256(output/'gpu_completion.json')),
            reference_frame_scope='all 16 future frames; zero history anchors',
            FVD_population='complete four-cell corpus for each draw, then mean across eight draws and two views',
            reader_metrics_completed=False,old_S1_results_used=False,original_outputs_modified=False,cpu_seconds=time.monotonic()-started)
        write_json(output/'report.json',report);write_json(output/'completion.json',dict(completed=True,scope=report['scope'],manifest_sha256=signature,
            report_sha256=sha256(output/'report.json'),clips=len(rows)*8,draws=8,reader_metrics_completed=False,old_outputs_modified=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['check','gpu','report'])
    for key in ('run-root','plan','index','predictions','assets','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--gpu-hours',type=float,default=1.25)
    p.add_argument('--budget-ledger');p.add_argument('--stop-file');p.add_argument('--cpu-hours',type=float,default=24.);a=p.parse_args()
    if a.gpu_hours<=0 or a.cpu_hours<=0:raise ValueError('Positive explicit runtime budgets required')
    if a.action=='check':
        import json
        _,contract=preflight(a);print(json.dumps(dict(cpu_only=True,models_loaded=False,population=contract['population'],schema=SCHEMA)))
    elif a.action=='gpu':gpu_pass(a)
    else:cpu_report(a)


if __name__=='__main__':main()
