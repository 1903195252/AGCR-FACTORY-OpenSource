"""Run an independently trained visual reader through history and future.

Neither predicted internal states nor auxiliary generator reader tensors enter
this interface. Reference labels are exported separately after prediction.
"""
import argparse
from pathlib import Path
import numpy as np
from .common import read_json,write_json,write_jsonl,sha256,prediction_manifest,NATIVE_CLASSES,VIEWS
from .metrics import reappearance_events

CALIBRATION_SOURCE='audited_reference_camera_parameters'


def evaluation_geometry(generated,image_shape,K,camera_to_world,time_s):
    """Fixed measurement calibration, independent of generator input errors.

    Only sensor calibration, timestamps and image *shape* enter this check.
    It neither reads future reference pixels/state nor conditions a generator.
    Generated poses retain the possibly biased conditioning provenance; they
    must have the correct time/view axes but never calibrate the visual reader.
    """
    expected=tuple(image_shape)
    if len(expected)!=5 or expected[0]!=22 or expected[1]!=2 or expected[-1]!=3:
        raise ValueError('Reference must contain six history and sixteen future paired RGB frames')
    if np.shape(generated['rgb'])!=(16,)+expected[1:]:
        raise ValueError('Generated frame/view/resolution axes do not match the reference sensors')
    if np.shape(generated['camera_to_world'])!=(16,2,4,4):
        raise ValueError('Generated camera conditions have invalid frame/view axes')
    times=np.asarray(time_s);poses=np.asarray(camera_to_world);intrinsics=np.asarray(K)
    if times.shape!=(22,) or not np.isfinite(times).all() or not (np.diff(times)>0).all():
        raise ValueError('Reference timestamps must be finite and strictly ordered')
    if poses.shape!=(22,2,4,4) or not np.isfinite(poses).all():raise ValueError('Invalid reference camera poses')
    if not np.allclose(poses[...,3,:],[0.,0.,0.,1.]):raise ValueError('Invalid homogeneous reference camera poses')
    if intrinsics.shape==(3,3):intrinsics=np.broadcast_to(intrinsics,(22,2,3,3))
    elif intrinsics.shape==(2,3,3):intrinsics=np.broadcast_to(intrinsics,(22,2,3,3))
    if intrinsics.shape!=(22,2,3,3) or not np.isfinite(intrinsics).all():raise ValueError('Invalid reference camera intrinsics')
    if not ((intrinsics[...,0,0]>0)&(intrinsics[...,1,1]>0)).all():raise ValueError('Reference focal lengths must be positive')
    return intrinsics,poses


def mask_rle(mask):
    from pycocotools import mask as api
    r=api.encode(np.asfortranarray(np.asarray(mask,bool).astype(np.uint8)));r['counts']=r['counts'].decode('ascii')
    return r


def metadata(row,episode):
    parent=row['scene_parent_id'];layout=row.get('layout')
    if layout is None:
        import re
        m=re.search(r'layout_(\d+)',parent)
        if m is None:raise ValueError('Explicit layout missing from scene ancestry')
        layout=int(m.group(1))
    return dict(episode=episode,scene_parent_id=parent,layout=layout)


def infer_arrays(model,association,rgb,depth,valid,K,poses,times,meta,floor_height_m=0.):
    from experiments.downstream_tracking.inference import IndependentReader
    reader=IndependentReader(model,association,floor_height_m=floor_height_m);rows=[]
    for t in range(len(rgb)):
        k=K[t] if K.ndim==4 else K
        detected=reader.step(rgb[t],depth[t],valid[t],k,poses[t],float(times[t]))
        for view in VIEWS:
            packed=[]
            for det in detected[view]:
                d=dict(det);mask=d.pop('mask',None)
                if mask is None:raise ValueError('Reader inference omitted independently predicted pixel mask')
                d['mask_rle']=mask_rle(mask>=.5);packed.append(d)
            rows.append(dict(meta,frame=t,time_s=float(times[t]),view=view,canvas_size=list(rgb.shape[2:4]),detections=packed))
    return rows


def native_reference(arrays,meta,anchors=(5,)):
    rows=[];instance=arrays['instance'];ids=arrays['instance_id'];classes=arrays['class_id'];T,V,H,W=instance.shape
    visible=(arrays['visible_fraction']>=.1)&(arrays['projection_area']>=144)&arrays['present'][:,None,:]
    for t in range(T):
        for v,view in enumerate(VIEWS):
            targets=[]
            for k,identity in enumerate(ids):
                mask=instance[t,v]==identity;yy,xx=np.where(mask)
                box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else [0,0,0,0]
                s=arrays['world_state'][t,k]
                targets.append(dict(track_id=int(identity),class_id=NATIVE_CLASSES[int(classes[k])],score=1.,box_xyxy=box,
                    present=bool(arrays['present'][t,k]),visible=bool(visible[t,v,k]),visible_fraction=float(arrays['visible_fraction'][t,v,k]),
                    projection_area=int(arrays['projection_area'][t,v,k]),position_world_m=s[:3].tolist(),velocity_world_mps=s[3:6].tolist(),
                    yaw_rad=float(s[6]),dimensions_m=s[8:11].tolist(),mask_rle=mask_rle(mask)))
            rows.append(dict(meta,frame=t,time_s=float(arrays['time_s'][t]),view=view,canvas_size=[H,W],detections=targets))
    events=[]
    for e in reappearance_events(visible,arrays['projection_area']):
        v,k=e['view'],e['target'];h=e['reappearance']
        eligible=[anchor for anchor in anchors if anchor<h<=anchor+16 and e['hidden_start']<=anchor
            and not visible[anchor,v,k] and arrays['projection_area'][anchor,v,k]>=144
            and visible[max(0,anchor-5):anchor+1,v,k].any()]
        if eligible:
            events.append(dict(e,anchor=int(eligible[0]),episode=meta['episode'],view=VIEWS[v],track_id=int(ids[k])))
    return rows,events


def generator_readout(index_path,predictions,checkpoint,assets,association_path,output,device='cuda:0',development=False,limit=None,gpu_hours=24.,allow_oracle_diagnostics=False):
    from .runtime import Budget
    budget=Budget(gpu_hours)
    from experiments.downstream_tracking.inference import load_reader,reader_implementation
    index=read_json(index_path);manifest=prediction_manifest(predictions,development,allow_oracle_diagnostics)
    if index.get('development_only') and not development:raise ValueError('Development reference index')
    if manifest.get('index_sha256')!=sha256(index_path):raise ValueError('Reference/prediction index provenance mismatch')
    model,saved=load_reader(checkpoint,assets,device,required_role='video_evaluation',allow_development=development)
    if manifest.get('output_modality')=='rgb' and model.modality!='rgb':raise ValueError('RGB-only generators require the common RGB-only evaluation reader')
    identity=saved['identity'];association=read_json(association_path)
    if not development and (association.get('split')!='val' or association.get('checkpoint_sha256')!=sha256(checkpoint)):
        raise ValueError('Association thresholds must be fixed on validation for this independent reader')
    config=association.get('association',association)
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    provenance=dict(role='video_evaluation',checkpoint_sha256=sha256(checkpoint),
        training_partition_sha256=identity.get('training_partition_sha256',identity.get('partition_sha256')),
        generator_auxiliary_reader=False,future_reference_input=False,completed_training=bool(saved['completed']),
        development_only=bool(development),generator_manifest_sha256=sha256(Path(predictions)/'manifest.json'),
        association_sha256=sha256(association_path),reader_identity=identity,split=manifest['split'])
    provenance['calibration_source']=CALIBRATION_SOURCE
    provenance.update(reader_implementation())
    if manifest.get('diagnostic_only'):
        provenance.update(diagnostic_only=True,generator_conditions=manifest['conditions'],downstream_augmentation=False)
    if not provenance['training_partition_sha256']:raise ValueError('Independent training partition hash absent')
    if (root/'manifest.json').exists() and read_json(root/'manifest.json')!=provenance:raise ValueError('Refusing mixed reader output provenance')
    write_json(root/'manifest.json',provenance);families=[r for r in index['families'] if r['split']==manifest['split']]
    if limit:
        if not development:raise ValueError('Formal evaluation uses the complete registered split')
        families=families[:limit]
    completed=0
    for family in families:
        f=family['family_id'];folder=root/f;folder.mkdir(exist_ok=True)
        for branch in family['records']:
            world,camera=branch['world'],branch['camera'];ep='%s_w%d_c%d'%(f,world,camera);meta=metadata(family,ep)
            native_path=Path(index['root'])/branch['path']
            if sha256(native_path)!=branch['sha256']:raise ValueError('Changed audited native branch')
            with np.load(native_path,allow_pickle=False) as raw:arrays={k:raw[k] for k in raw.files}
            refs,events=native_reference(arrays,meta)
            write_jsonl(folder/('reference_w%d_c%d.jsonl'%(world,camera)),refs)
            write_json(folder/('events_w%d_c%d.json'%(world,camera)),events)
            for draw,seed in enumerate(manifest['sample_seeds']):
                dest=folder/('prediction_w%d_c%d_s%02d.jsonl'%(world,camera,draw));marker=dest.with_suffix('.complete.json')
                if marker.exists():
                    if read_json(marker).get('sha256')!=sha256(dest):raise ValueError('Changed completed reader output')
                    completed+=1;continue
                if budget.expired():
                    write_json(root/'status.json',dict(completed_branches=completed,budget_paused=True,elapsed_gpu_hours=budget.elapsed_hours()));return False
                source=Path(predictions)/f/('world%d_camera%d_sample%02d.npz'%(world,camera,draw))
                draw_marker=read_json(Path(predictions)/f/('draw_%02d.json'%draw))
                if draw_marker['seed']!=seed or draw_marker['files'][source.name]!=sha256(source):raise ValueError('Changed generated draw')
                with np.load(source,allow_pickle=False) as generated:
                    # Explicitly select only image/sensor channels. Future
                    # reference imagery/state/IDs never reach IndependentReader.
                    rgb=np.concatenate([arrays['rgb'][:6],generated['rgb']])
                    if 'depth_mm' in generated:
                        future_depth=generated['depth_mm'];future_valid=generated['validity_probability']>=.5
                    else:
                        if model.modality!='rgb':raise ValueError('Missing depth cannot enter an RGB-D evaluation reader')
                        future_depth=np.zeros_like(arrays['depth_mm'][6:]);future_valid=np.zeros_like(arrays['valid'][6:],dtype=bool)
                    depth=np.concatenate([arrays['depth_mm'][:6],future_depth])
                    valid=np.concatenate([arrays['valid'][:6],future_valid])
                    K,poses=evaluation_geometry(generated,arrays['rgb'].shape,arrays['K'],arrays['camera_to_world'],arrays['time_s'])
                floor=float(arrays['world_from_factory'][2,3]) if 'world_from_factory' in arrays else 0.
                rows=infer_arrays(model,config,rgb,depth,valid,K,poses,arrays['time_s'],meta,floor_height_m=floor)
                write_jsonl(dest,rows);write_json(marker,dict(sha256=sha256(dest),source_sha256=sha256(source),seed=seed));completed+=1
                write_json(root/'status.json',dict(completed_branches=completed,total_branches=len(families)*4*len(manifest['sample_seeds'])))
    write_json(root/'completion.json',dict(completed=True,development_only=development,branches=completed))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',required=True);p.add_argument('--predictions',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--assets',required=True);p.add_argument('--association',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--development',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--gpu-hours',type=float,default=24.)
    p.add_argument('--allow-oracle-diagnostics',action='store_true',help='Explicitly score diagnostic oracle samples; never primary evidence or augmentation')
    a=p.parse_args();generator_readout(a.index,a.predictions,a.checkpoint,a.assets,a.association,a.output,a.device,a.development,a.limit,a.gpu_hours,a.allow_oracle_diagnostics)
if __name__=='__main__':main()
