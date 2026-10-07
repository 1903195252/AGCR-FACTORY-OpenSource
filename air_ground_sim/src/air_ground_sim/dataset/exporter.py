"""Resumable scene-level nuScenes exporter for a complete native factory."""
import hashlib
import json
import time
import uuid
import shutil
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image,ImageDraw
import mujoco
from .. import Simulator
from ..paths import CONFIG
from ..scene.actors import DynamicActorController
from ..scene.registry import sha256
from .catalog import ObjectCatalog,CATEGORIES,PALETTE
from .geometry import yaw_rotation,quaternion,box_corners,projected_edges
from .rig import FiveCameraRig,CHANNELS

TABLES=('category','attribute','visibility','instance','sensor','calibrated_sensor','ego_pose',
        'log','scene','sample','sample_data','sample_annotation','map')


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n')
    tmp.replace(path)


def token(*parts):return uuid.uuid5(uuid.NAMESPACE_URL,'agcr-factory/'+ '/'.join(map(str,parts))).hex


def scene_plan(config):
    layout=json.loads((CONFIG/'factory_industrial_layout.json').read_text())
    benches=[m for m in layout['modules'] if m['kind']=='assembly']
    if not benches:
        benches=[m for m in layout['modules'] if m['name'].startswith('industrial_assembly_')]
    rng=np.random.default_rng(config['seed']);plans=[]
    for i in range(config['scenes']):
        if i%50<40:
            bench=np.asarray(benches[i%40]['position'],float)
            center=bench+np.array([0,-2.,3.4])
            # Short paths on the operator side, below 4.8 m utility bridges.
            axis=np.array([1.,0,0]);length=5.
            region='assembly';yaw=float(rng.uniform(-np.pi,np.pi))
        else:
            j=i%50-40
            center=np.array([-155+j%5*70, -82 if j<5 else -110, 7.2])
            axis=np.array([1.,0,0]);length=16.;region='logistics';yaw=float(rng.uniform(-np.pi,np.pi))
        plans.append(dict(name=f'factory-{i:04d}',index=i,center=center.tolist(),axis=axis.tolist(),
            length_m=length,yaw=yaw,region=region,split='train' if i%50<28 or 40<=i%50<47 else ('val' if i%50<34 or i%50==47 else 'test')))
    return plans


def pose(plan,fraction):
    p=np.array(plan['center'])+np.array(plan['axis'])*plan['length_m']*(fraction-.5)
    # Smooth yaw change; virtual rig is not the physical UAV dynamics.
    yaw=plan['yaw']+.25*np.sin(np.pi*fraction)
    return p,yaw_rotation(yaw)


def link_records(records,key):
    previous={}
    for record in records:
        k=key(record);record['prev']='';record['next']=''
        if k in previous:
            previous[k]['next']=record['token'];record['prev']=previous[k]['token']
        previous[k]=record


def save_images(root,image_file,semantic_file,instance_file,rgb,semantic,instance,quality):
    Image.fromarray(rgb).save(root/image_file,quality=quality,subsampling=0)
    Image.fromarray(semantic).save(root/semantic_file,compress_level=1)
    Image.fromarray(instance).save(root/instance_file,compress_level=1)


def save_preview(root,name,views,boxes,catalog,offset,K):
    w,h=views[0]['rgb'].shape[1],views[0]['rgb'].shape[0]
    sheet=Image.new('RGB',(w*3,(h+28)*len(views)),(24,29,35));draw=ImageDraw.Draw(sheet)
    for row,v in enumerate(views):
        rgb=Image.fromarray(v['rgb']);overlay=rgb.copy();d=ImageDraw.Draw(overlay)
        for oid in v['visible']:
            b=boxes[oid];color=tuple(PALETTE[catalog.by_id[oid]['category_id']].tolist())
            corners=box_corners(b['translation'],b['size'],b['rotation'])-offset
            for line in projected_edges(corners,v['position'],v['rotation'],K):
                if np.max(np.abs(line))<1e5:d.line([tuple(p) for p in line],fill=color,width=2)
        semantic=Image.fromarray(PALETTE[v['semantic']])
        y=row*(h+28)
        for col,(im,label) in enumerate([(rgb,'RGB'),(overlay,'projected 3D boxes'),(semantic,'semantic IDs')]):
            sheet.paste(im,(col*w,y+28));draw.text((col*w+8,y+8),v['channel']+' | '+label,fill='white')
    (root/'previews').mkdir(exist_ok=True)
    sheet.save(root/'previews'/f'{name}.jpg',quality=88)


def base_tables(config,rig,identity):
    t={n:[] for n in TABLES}
    t['category']=[dict(token=token('category',n),name=n,description=d,index=i) for i,(n,d,_) in enumerate(CATEGORIES)]
    t['visibility']=[dict(token=str(i+1),level=l,description=d) for i,(l,d) in enumerate([
        ('v0-40','0-40% visible'),('v40-60','40-60% visible'),('v60-80','60-80% visible'),('v80-100','80-100% visible')])]
    # Visibility is deliberately blank on annotations: no fake occlusion fraction.
    for c in rig.calibration():
        st=token(identity,'sensor',c['channel']);ct=token(identity,'calibration',c['channel'])
        t['sensor'].append(dict(token=st,channel=c['channel'],modality='camera'))
        t['calibrated_sensor'].append(dict(token=ct,sensor_token=st,translation=c['translation'],rotation=c['rotation'],camera_intrinsic=c['camera_intrinsic']))
    log=token(identity,'log')
    t['log']=[dict(token=log,logfile=config['name'],vehicle='virtual-uav-survey-rig',date_captured='2026-09-22',location='agcr-factory-synthetic')]
    t['map']=[dict(token=token(identity,'map'),category='semantic_prior',filename='maps/factory_floor.png',log_tokens=[log])]
    return t


def export_dataset(config,output,resume=False):
    root=Path(output)
    if root.exists() and not resume:raise FileExistsError(root)
    if config['trajectory_mode']!='kinematic_virtual_survey':raise ValueError('Unsupported trajectory mode')
    if config['profile']!='factory_industrial':raise ValueError('This semantic catalog is bound to factory_industrial')
    if config['scenes']<1 or config['samples_per_scene']<2 or config['fps']<=0:raise ValueError('Invalid sequence dimensions')
    if config['width']<16 or config['height']<16 or not 1<config['horizontal_fov_deg']<170:raise ValueError('Invalid camera configuration')
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    root.mkdir(parents=True,exist_ok=True)
    cfg_file=root/'capture_config.json'
    if resume:
        if json.loads(cfg_file.read_text())!=config:raise ValueError('Resume configuration differs')
    else:write_json(cfg_file,config)
    for folder in ('samples','semantic','instance_masks','extensions','chunks','maps'):(root/folder).mkdir(exist_ok=True)
    print('Loading complete factory model...',flush=True)
    started=time.monotonic();offset=np.asarray(config['global_offset_m'])
    with Simulator(config['profile'],dynamic_actors=False) as sim:
        model,data=sim.model,sim.data
        source=sim.model_spec['model_sha256']
        pipeline_files=list(Path(__file__).parent.glob('*.py'))+[CONFIG/n for n in (
            'factory_semantic_sources.json','factory_industrial_layout.json','appearance_industrial.json')]
        pipeline_hashes={p.name:sha256(p) for p in sorted(pipeline_files)}
        pipeline_digest=hashlib.sha256(json.dumps(pipeline_hashes,sort_keys=True).encode()).hexdigest()
        identity=source+'/'+digest+'/'+pipeline_digest
        if resume and (root/'extensions/pipeline.json').exists():
            if json.loads((root/'extensions/pipeline.json').read_text())['sha256']!=pipeline_digest:
                raise ValueError('Pipeline changed since capture; start a new dataset')
        write_json(root/'extensions/pipeline.json',dict(sha256=pipeline_digest,files=pipeline_hashes))
        (root/'source').mkdir(exist_ok=True)
        for file in pipeline_files:shutil.copy2(file,root/'source'/file.name)
        catalog=ObjectCatalog(model,data)
        rig=FiveCameraRig(model,config['width'],config['height'],config['horizontal_fov_deg'],sim.appearance,config['shadow_size'])
        rig.option.geomgroup[4]=int(config['roof_visible'])
        actor=DynamicActorController(model,'industrial')
        write_json(root/'extensions/object_catalog.json',catalog.manifest())
        write_json(root/'extensions/semantic_sources.json',json.loads((CONFIG/'factory_semantic_sources.json').read_text()))
        plans=scene_plan(config);write_json(root/'extensions/trajectories.json',plans)
        tables=base_tables(config,rig,identity)
        # Georeferenced floor-footprint prior, explicitly NOT traversability.
        mask=Image.new('L',(4100,3100),0);ImageDraw.Draw(mask).rectangle((50,50,4050,3050),fill=255)
        mask.save(root/'maps/factory_floor.png')
        write_json(root/'extensions/map_metadata.json',dict(resolution_m=.1,global_offset_m=offset.tolist(),meaning='Factory floor footprint only; obstacles not subtracted, not a traversability map'))
        state_boxes=None
        writers=ThreadPoolExecutor(max_workers=4)
        pending=[]
        try:
            for plan in plans:
                sid=plan['index'];chunk=root/'chunks'/f'{sid:04d}.json'
                if resume and chunk.exists():
                    saved=json.loads(chunk.read_text())
                    if saved['identity']!=identity:raise ValueError('Checkpoint identity mismatch')
                    print('Resuming: completed '+plan['name'],flush=True);continue
                records={n:[] for n in ('scene','sample','sample_data','sample_annotation','ego_pose','instance')}
                extensions=[];instances={};scene_token=token(identity,'scene',sid);count=config['samples_per_scene']
                scene_started=time.monotonic();pixel_counts=Counter();annotation_count=0
                for frame in range(count):
                    timestamp=config['epoch_us']+round((sid*count+frame)*1e6/config['fps'])
                    sample_token=token(identity,'sample',sid,frame);ego_token=token(identity,'ego',sid,frame)
                    position,ego_rotation=pose(plan,frame/(count-1))
                    data.time=(sid*count+frame)/config['fps']
                    actor.update(data,data.time)
                    mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
                    if state_boxes is None:state_boxes=catalog.boxes(data,offset)
                    boxes=dict(state_boxes)
                    for obj in catalog.objects:
                        if obj['dynamic']:
                            b=obj['body'];r=data.xmat[b].reshape(3,3)
                            boxes[obj['id']]=dict(translation=(data.xpos[b]+r@obj['local_center']+offset).tolist(),size=obj['extent_xyz'][[1,0,2]].tolist(),rotation=quaternion(r))
                    frozen=(data.time,data.qpos.copy(),data.mocap_pos.copy(),data.mocap_quat.copy())
                    rig.update(data,position)
                    records['sample'].append(dict(token=sample_token,timestamp=timestamp,scene_token=scene_token))
                    records['ego_pose'].append(dict(token=ego_token,timestamp=timestamp,translation=(position+offset).tolist(),rotation=quaternion(ego_rotation)))
                    seen=set();previews=[]
                    for cam_index,channel in enumerate(CHANNELS):
                        rgb,seg,cam_pos,cam_rot=rig.view(cam_index,position,ego_rotation)
                        semantic,instance=catalog.masks(seg)
                        values,counts=np.unique(instance,return_counts=True)
                        visible=[int(i) for i,n in zip(values,counts) if i and n>=config['min_instance_pixels']]
                        seen.update(visible)
                        for i,n in zip(*np.unique(semantic,return_counts=True)):pixel_counts[int(i)]+=int(n)
                        sd=token(identity,'image',sid,frame,channel);stem=f'{plan["name"]}__{channel}__{timestamp}'
                        image_file=f'samples/{channel}/{stem}.jpg';semantic_file=f'semantic/{channel}/{stem}.png';instance_file=f'instance_masks/{channel}/{stem}.png'
                        for file in (image_file,semantic_file,instance_file):(root/file).parent.mkdir(exist_ok=True)
                        pending.append(writers.submit(save_images,root,image_file,semantic_file,instance_file,rgb,semantic,instance,config['jpeg_quality']))
                        if len(pending)>10:pending.pop(0).result()
                        records['sample_data'].append(dict(token=sd,sample_token=sample_token,ego_pose_token=ego_token,
                            calibrated_sensor_token=token(identity,'calibration',channel),filename=image_file,fileformat='jpg',width=config['width'],height=config['height'],timestamp=timestamp,is_key_frame=True))
                        observations=[]
                        for oid in visible:
                            yy,xx=np.nonzero(instance==oid)
                            observations.append(dict(instance_id=oid,instance_token=token(identity,'instance',sid,oid),annotation_token=token(identity,'annotation',sid,frame,oid),
                                visible_pixels=int(len(xx)),bbox_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]))
                        extensions.append(dict(sample_data_token=sd,semantic_filename=semantic_file,instance_filename=instance_file,
                            camera_to_global_translation=(cam_pos+offset).tolist(),camera_to_global_rotation=quaternion(cam_rot),objects=observations))
                        if frame in (0,count//2,count-1):
                            previews.append(dict(channel=channel,rgb=rgb,semantic=semantic,visible=visible,position=cam_pos,rotation=cam_rot))
                    assert data.time==frozen[0] and np.array_equal(data.qpos,frozen[1]) and np.array_equal(data.mocap_pos,frozen[2]) and np.array_equal(data.mocap_quat,frozen[3]),'Scene changed during synchronous camera capture'
                    for oid in sorted(seen):
                        obj=catalog.by_id[oid];it=token(identity,'instance',sid,oid);at=token(identity,'annotation',sid,frame,oid)
                        if it not in instances:instances[it]=dict(token=it,category_token=token('category',CATEGORIES[obj['category_id']][0]),nbr_annotations=0,first_annotation_token=at,last_annotation_token=at)
                        instances[it]['nbr_annotations']+=1;instances[it]['last_annotation_token']=at
                        records['sample_annotation'].append(dict(token=at,sample_token=sample_token,instance_token=it,visibility_token='',attribute_tokens=[],
                            **boxes[oid],num_lidar_pts=0,num_radar_pts=0))
                        annotation_count+=1
                    if previews:save_preview(root,f'{plan["name"]}_{frame:04d}',previews,boxes,catalog,offset,rig.K)
                    if frame%20==0:print(f'{plan["name"]}: {frame+1}/{count} frames, {time.monotonic()-scene_started:.1f}s',flush=True)
                for job in pending:job.result()
                pending.clear()
                records['instance']=list(instances.values())
                link_records(records['sample'],lambda x:0)
                link_records(records['sample_data'],lambda x:x['calibrated_sensor_token'])
                link_records(records['sample_annotation'],lambda x:x['instance_token'])
                records['scene']=[dict(token=scene_token,name=plan['name'],description=f'{plan["region"]}; scripted virtual five-camera UAV survey; native MuJoCo geometry',
                    log_token=tables['log'][0]['token'],nbr_samples=count,first_sample_token=records['sample'][0]['token'],last_sample_token=records['sample'][-1]['token'])]
                write_json(root/'extensions'/f'{plan["name"]}.json',extensions)
                write_json(chunk,dict(identity=identity,tables=records,pixel_counts=dict(pixel_counts),annotations=annotation_count,seconds=time.monotonic()-scene_started))
                write_json(root/'progress.json',dict(completed_scenes=sid+1,total_scenes=len(plans),completed_samples=(sid+1)*count,elapsed_s=time.monotonic()-started,status='running'))
                print('Completed '+plan['name'],flush=True)
        finally:
            writers.shutdown(wait=True)
            rig.close()
        totals=Counter()
        for plan in plans:
            chunk=json.loads((root/'chunks'/f'{plan["index"]:04d}.json').read_text())
            for name,rows in chunk['tables'].items():tables[name].extend(rows)
            totals.update({int(k):v for k,v in chunk['pixel_counts'].items()})
        for name in TABLES:write_json(root/config['version']/(name+'.json'),tables[name])
        write_json(root/'splits.json',{split:[p['name'] for p in plans if p['split']==split] for split in ('train','val','test')})
        manifest=dict(status='complete',profile=sim.profile,model_sha256=source,config_sha256=digest,
            pipeline_sha256=pipeline_digest,samples=len(tables['sample']),images=len(tables['sample_data']),annotations=len(tables['sample_annotation']),
            instances=len(tables['instance']),scenes=len(plans),elapsed_s=time.monotonic()-started,channels=list(CHANNELS),
            semantic_pixels={CATEGORIES[k][0]:v for k,v in sorted(totals.items())},ignore_fraction=totals[0]/sum(totals.values()),
            capture_mode=config['trajectory_mode'],synchronized=True,physics_integrated=False,
            conventions=dict(global_axes='MuJoCo XYZ, translated by global_offset_m',ego_axes='X forward Y left Z up',camera_axes='X right Y down Z forward',
                quaternion='wxyz',box_size='width length height',timestamp='microseconds; synthetic Unix epoch'),
            limitations=['Factory-specific ontology; not official nuScenes detection classes or splits',
                'No LiDAR/radar: point counts are zero, do not apply nonzero-point filtering',
                'Visibility fraction not estimated; visibility_token is empty',
                'Mixed/unverified legacy batches are semantic ignore; see object catalog',
                'Virtual survey rig and prescribed mocap actors; no integrated flight dynamics',
                'Scene-disjoint splits in one fixed factory, not independent-world generalization'])
        write_json(root/'manifest.json',manifest);write_json(root/'progress.json',dict(status='complete',completed_samples=manifest['samples'],completed_scenes=len(plans)))
        print(json.dumps(manifest,indent=2),flush=True)
        return manifest
