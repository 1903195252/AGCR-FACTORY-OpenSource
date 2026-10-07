"""Synchronous native RGB/optical-Z/segmentation and isolated-surface capture."""
import json
import math
import os
import shutil
import time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import air_ground_sim
import h5py
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from .common import atomic_json, bbox, digest, file_hash, intrinsics, quaternion, rotation_z, CaptureQualityError, ScriptValidationError
from .planning import actor_state, camera_poses, episode_spec, CLASS_IDS
from .events import fit_fixtures, classify, audit_event, frustum_presence

SIGNS=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])

def local_bounds(model,data,gids,body):
    points=[]
    for g in gids:
        kind=model.geom_type[g];s=model.geom_size[g]
        if kind==mujoco.mjtGeom.mjGEOM_MESH:
            k=model.geom_dataid[g];lo=model.mesh_vertadr[k];count=model.mesh_vertnum[k]
            v=model.mesh_vert[lo:lo+count]
        else:
            extent=s.copy()
            if kind==mujoco.mjtGeom.mjGEOM_SPHERE: extent[:]=s[0]
            elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE: extent=np.array([s[0],s[0],s[0]+s[1]])
            elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER: extent=np.array([s[0],s[0],s[1]])
            v=SIGNS*extent
        world=v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        points.append((world-data.xpos[body])@data.xmat[body].reshape(3,3))
    v=np.concatenate(points);lo,hi=v.min(0),v.max(0)
    return (lo+hi)/2,hi-lo

class SceneCapture:
    def __init__(self,xml,config):
        self.c=config;self.m=mujoco.MjModel.from_xml_path(str(xml));self.d=mujoco.MjData(self.m)
        self.m.vis.map.znear=config['near_m']/self.m.stat.extent
        self.m.vis.map.zfar=config['far_m']/self.m.stat.extent
        self.m.vis.quality.offsamples=0
        self.m.vis.quality.shadowsize=config['shadowsize']
        mujoco.mj_forward(self.m,self.d)
        self.r=mujoco.Renderer(self.m,height=config['height'],width=config['width'],max_geom=self.m.ngeom+100)
        self.K=intrinsics(config);self.targets={}
        for k in range(config['max_targets']):
            for kind in CLASS_IDS:
                b=self.m.body(kind+'_'+str(k)).id
                descendants={b}
                for child in range(b+1,self.m.nbody):
                    if int(self.m.body_parentid[child]) in descendants:descendants.add(child)
                gids=np.flatnonzero(np.isin(self.m.geom_bodyid,list(descendants)))
                center,dim=local_bounds(self.m,self.d,gids,b)
                self.targets[(kind,k)]=dict(body=b,gids=gids,mid=int(self.m.body_mocapid[b]),center=center,dimensions=dim)
        self.static=[]
        for gid in np.flatnonzero(self.m.geom_group==0):
            if self.m.geom_type[gid]==mujoco.mjtGeom.mjGEOM_PLANE:continue
            center,dim=local_bounds(self.m,self.d,[gid],0)
            lo,hi=center-dim/2,center+dim/2
            if hi[2]>.06 and lo[2]<3.5:self.static.append((lo,hi,gid))
        self.static_lo=np.array([x[0] for x in self.static]);self.static_hi=np.array([x[1] for x in self.static])
        self.base_rgba=self.m.geom_rgba.copy()
        self.base_geom_pos=self.m.geom_pos.copy();self.base_geom_size=self.m.geom_size.copy()
        self.option=mujoco.MjvOption();self.option.sitegroup[:]=0
        self.option.geomgroup[:]=0;self.option.geomgroup[[0,1]]=1

    def collision_free(self,spec):
        # 50 Hz geometry check, independent of image quality or test outcomes.
        times=np.arange(0,spec['duration']+.001,.02)
        self.set_state(spec,0.)
        extra=[]
        for f in spec['fixtures']:
            b=self.m.body(f['name']).id
            for gid in np.flatnonzero(self.m.geom_bodyid==b):
                ctr,dim=local_bounds(self.m,self.d,[gid],0);extra.append((ctr-dim/2,ctr+dim/2))
        low=np.concatenate([self.static_lo,np.array([a for a,b in extra])]) if extra else self.static_lo
        high=np.concatenate([self.static_hi,np.array([b for a,b in extra])]) if extra else self.static_hi
        trajectories=[]
        for a in spec['actors']:
            obj=self.targets[(a['kind'],a['slot'])]
            p,u,yaw,rate=actor_state(a,times,spec['duration'])
            bounds=[]
            for j in range(len(times)):
                R=rotation_z(yaw[j]);center=p[j]+R@obj['center'];extent=np.abs(R)@(obj['dimensions']/2)
                lo=center-extent;hi=center+extent
                hits=np.all(hi[None,:] > low+.012,axis=1)&np.all(lo[None,:]<high-.012,axis=1)
                if hits.any():
                    self.collision_reason=dict(actor=a['kind'],slot=a['slot'],time=float(times[j]),target_lo=lo.tolist(),target_hi=hi.tolist(),obstacle_lo=low[np.flatnonzero(hits)[0]].tolist(),obstacle_hi=high[np.flatnonzero(hits)[0]].tolist())
                    return False
                bounds.append((lo,hi))
            trajectories.append(np.array(bounds))
        for i,A in enumerate(trajectories):
            for B in trajectories[:i]:
                if (np.all(A[:,1]>B[:,0]+.012,axis=1)&np.all(A[:,0]<B[:,1]-.012,axis=1)).any():return False
        return True

    def configure_fixtures(self,spec):
        self.m.geom_pos[:]=self.base_geom_pos;self.m.geom_size[:]=self.base_geom_size
        for f in spec['fixtures']:
            body=self.m.body(f['name']).id;gids=np.flatnonzero(self.m.geom_bodyid==body)
            scale=np.array(f['scale'])
            self.m.geom_pos[gids]*=scale;self.m.geom_size[gids]*=scale
            # Suspensions reach the actual ceiling; manipulator support reaches floor.
            for gid in gids:
                name=self.m.geom(gid).name
                if '_link' in name:
                    self.m.geom_pos[gid,0]*=f.get('support_side',1.)
                if '_cable' in name:
                    top=15.7-f['position'][2];bottom=f['scale'][2]
                    self.m.geom_pos[gid,2]=(top+bottom)/2
                    self.m.geom_size[gid]=[.015,.015,(top-bottom)/2]
                if '_link2' in name:
                    low=-f['position'][2];high=-f['scale'][2]*.6
                    self.m.geom_pos[gid,2]=(low+high)/2
                    self.m.geom_size[gid,2]=max(.05,(high-low)/2)
            self.m.geom_rbound[gids]=np.linalg.norm(self.m.geom_size[gids],axis=1)

    def spec(self,plan,attempts=None):
        failures=[]
        for attempt in (range(3) if attempts is None else attempts):
            s=fit_fixtures(self,episode_spec(plan,self.c,attempt),plan)
            self.configure_fixtures(s)
            if not self.collision_free(s):
                failures.append(dict(collision=getattr(self,'collision_reason','actor overlap')));continue
            frames=(list(range(s['before_frames'][0],s['after_frames'][-1]+1)) if s['before_pair']==s['after_pair'] else sorted(set(s['before_frames']+s['after_frames'])))
            fractions=np.zeros((self.c['frames_per_episode'],2,len(s['actors'])))
            areas=np.zeros_like(fractions,dtype=np.int32)
            for frame in frames:
                t=frame/self.c['fps'];self.set_state(s,t)
                for v,(C,_,_) in enumerate(camera_poses(s,t)):
                    surfaces=self.view(C,s)[-1]
                    for k,q in enumerate(surfaces):
                        fractions[frame,v,k]=q['visible']/max(1,q['area']);areas[frame,v,k]=q['area']
            audit=audit_event(classify(fractions,areas,self.c),areas,s,self.c)
            if audit['passed']:
                s['event_preflight']=audit;return s
            failures.append(audit)
        raise ScriptValidationError('Script validation failed for '+plan['name']+': '+json.dumps(failures))

    def set_state(self,spec,t):
        self.d.mocap_pos[:]=[0,0,-100]
        self.d.mocap_quat[:]=[1,0,0,0]
        self.d.time=t
        for f in spec['fixtures']:
            self.d.mocap_pos[self.m.body_mocapid[self.m.body(f['name']).id]]=f['position']
        states=[]
        for a in spec['actors']:
            obj=self.targets[(a['kind'],a['slot'])]
            p,u,yaw,rate=actor_state(a,t,spec['duration']);R=rotation_z(yaw)
            self.d.mocap_pos[obj['mid']]=p
            self.d.mocap_quat[obj['mid']]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
            delta=R@obj['center'];center=p+delta
            velocity=u+float(rate)*np.array([-delta[1],delta[0],0])
            states.append(dict(center=center.tolist(),velocity=velocity.tolist(),yaw=float(yaw),
                               yaw_rate=float(rate),dimensions=obj['dimensions'].tolist(),origin=p.tolist(),present=True))
        # No integration between views or passes; all products share this exact state.
        mujoco.mj_forward(self.m,self.d)
        return states

    def update(self,C,groups):
        self.option.geomgroup[:]=0;self.option.geomgroup[groups]=1
        self.r.update_scene(self.d,scene_option=self.option)
        near,far=self.c['near_m'],self.c['far_m']
        half=near*self.c['height']/(2*self.K[1,1])
        for cam in self.r.scene.camera:
            cam.pos[:]=C[:3,3];cam.forward[:]=C[:3,2];cam.up[:]=-C[:3,1]
            cam.frustum_near=near;cam.frustum_far=far;cam.frustum_top=half;cam.frustum_bottom=-half
            cam.frustum_width=0;cam.frustum_center=0;cam.orthographic=0

    def depth(self):
        self.r.enable_depth_rendering()
        try: d=self.r.render().copy()*1000
        finally:self.r.disable_depth_rendering()
        v=np.isfinite(d)&(d>=self.c['depth_min_mm'])&(d<=self.c['depth_max_mm'])
        return np.where(v,d,0).astype(np.float32),v

    def view(self,C,spec):
        self.update(C,[0,1]);rgb=self.r.render().copy();depth,valid=self.depth()
        self.r.enable_segmentation_rendering()
        try:seg=self.r.render().copy()
        finally:self.r.disable_segmentation_rendering()
        geometry=(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&(seg[:,:,0]>=0)
        ids=seg[:,:,0].clip(0)
        semantic=np.where(geometry,self.m.geom_user[ids,0],0).astype(np.uint8)
        instance=np.zeros(geometry.shape,np.uint16)
        for k,a in enumerate(spec['actors']):
            gids=self.targets[(a['kind'],a['slot'])]['gids']
            instance[geometry & np.isin(ids,gids)]=k+1
        self.update(C,[0]);static,static_valid=self.depth()
        surfaces=[]
        for k,a in enumerate(spec['actors']):
            gids=self.targets[(a['kind'],a['slot'])]['gids']
            self.m.geom_group[gids]=2
            try:
                self.update(C,[2]);isolated,support=self.depth()
            finally:self.m.geom_group[gids]=1
            mask=instance==k+1
            if np.any(mask & ~support):
                raise CaptureQualityError('Visible target outside isolated target silhouette')
            rect=bbox(support);x0,y0,x1,y1=rect
            crop=isolated[y0:y1,x0:x1]
            behind=support & ~mask & valid & (depth<isolated-1.)
            contributors=np.bincount(semantic[behind].ravel(),minlength=18)[:18]
            surfaces.append(dict(box=rect,depth=crop,area=int(support.sum()),visible=int(mask.sum()),bbox_visible=bbox(mask),
                                 occluder_pixels=contributors))
        return rgb,depth,valid,semantic,instance,static,static_valid,surfaces

    def audit_depth(self,C,depth,instance,seed):
        rng=np.random.default_rng(seed);h,w=depth.shape
        valid=(depth>0)
        valid[:2]=False;valid[-2:]=False;valid[:,:2]=False;valid[:,-2:]=False
        # Avoid raster/surface boundaries; disagreement there is scored separately by masks.
        for axis in [0,1]:
            valid &= np.abs(depth-np.roll(depth,1,axis))<50
            valid &= np.abs(depth-np.roll(depth,-1,axis))<50
            valid &= instance==np.roll(instance,1,axis)
            valid &= instance==np.roll(instance,-1,axis)
        yy,xx=np.nonzero(valid)
        if not len(xx):raise CaptureQualityError('No depth audit support')
        errors=[]
        for j in rng.choice(len(xx),min(32,len(xx)),replace=False):
            ray=np.array([(xx[j]-self.K[0,2])/self.K[0,0],(yy[j]-self.K[1,2])/self.K[1,1],1.])
            direction=C[:3,:3]@ray;norm=np.linalg.norm(direction);direction/=norm
            gid=np.array([-1],np.int32)
            distance=mujoco.mj_ray(self.m,self.d,C[:3,3].copy(),direction,np.array([1,1,0,0,0,0],np.uint8),1,-1,gid)
            if distance>0:errors.append(abs(distance/norm*1000-float(depth[yy[j],xx[j]])))
        if not errors:raise CaptureQualityError('No ray intersections')
        return errors

    def close(self):self.r.close()

def ds(h,name,shape,dtype,config,chunks=None):
    return h.create_dataset(name,shape=shape,dtype=dtype,chunks=chunks or True,
                            compression='gzip',compression_opts=config['compression_level'],shuffle=True)

def preview(path,rgbs,depths,instances):
    import colorsys
    W,H=rgbs[0].shape[1],rgbs[0].shape[0]
    im=Image.new('RGB',(W*2,H*2+48),'white');draw=ImageDraw.Draw(im)
    for v in range(2):
        im.paste(Image.fromarray(rgbs[v]),(v*W,24))
        D=depths[v];x=np.clip((np.log(np.maximum(D,100))-np.log(1000))/(np.log(60000)-np.log(1000)),0,1)
        lut=np.array([colorsys.hsv_to_rgb(i/255*2/3,.9,.95) for i in range(256)])
        color=(lut[(x*255).astype(np.uint8)]*255).astype(np.uint8);color[D==0]=[210,210,210]
        im.paste(Image.fromarray(color),(v*W,H+48))
        draw.text((v*W+8,5),['CAM_AIR','CAM_GROUND'][v]+' | synchronized RGB',fill='black')
        draw.text((v*W+8,H+29),'Native optical-Z (mm): red near / blue far; 1000--60000 log scale',fill='black')
    im.thumbnail((1440,1008));im.save(path)

def collect_episode(scene,plan,config,out,identity,frame_limit=None):
    root=Path(out);name=plan['name'];final=root/'episodes'/name
    if final.exists():
        marker=final/'complete.json'
        if marker.exists() and json.loads(marker.read_text())['identity']==identity:
            return json.loads(marker.read_text())
        raise ValueError('Incomplete or incompatible final episode: '+str(final))
    stage=root/'staging'/name
    if stage.exists():shutil.rmtree(stage)
    stage.mkdir(parents=True);(stage/'rgb').mkdir()
    started=time.monotonic();spec=scene.spec(plan);N=len(spec['actors'])
    T=frame_limit or config['frames_per_episode'];W,H=config['width'],config['height']
    if shutil.disk_usage(root).free < config['disk_reserve_gib']*1024**3:
        raise RuntimeError('Dataset disk reserve reached')
    # Per-instance appearance stays constant over the whole episode.
    scene.m.geom_rgba[:]=scene.base_rgba
    rng=np.random.default_rng(np.random.SeedSequence(plan['camera_seed']+[703]))
    for a in spec['actors']:
        gids=scene.targets[(a['kind'],a['slot'])]['gids']
        for gid in gids:
            mid=int(scene.m.geom_matid[gid])
            if mid>=0:scene.m.geom_rgba[gid]=scene.m.mat_rgba[mid]
        scene.m.geom_rgba[gids,:3]*=rng.uniform(.90,1.,3)
    times=np.arange(T,dtype=np.float64)/config['fps']
    timestamps=config['epoch_us']+np.rint((plan['index']*config['frames_per_episode']/config['fps']+times)*1e6).astype(np.int64)
    camera=np.empty((T,2,4,4));ego=np.empty_like(camera);mount=np.empty_like(camera)
    states=np.empty((T,N,11),np.float64)
    visibility=np.zeros((T,2,N),np.float32);area=np.zeros((T,2,N),np.int32);visible_area=area.copy()
    boxes=np.zeros((T,2,N,4),np.int32);boxes_vis=boxes.copy();offsets=np.zeros((T,2,N,2),np.int64)
    checksums=[];ray_errors=[]
    contributor_counts=np.zeros((T,2,N,18),np.int32)
    with h5py.File(stage/'frames.h5','w') as h:
        h.attrs.update(schema_version=config['schema_version'],identity=identity,depth_unit='mm',world_unit='m',
                       depth_definition='optical-axis Z from native MuJoCo depth buffer',camera_axes='x right, y down, z forward',
                       kinematics='prescribed mocap; no closed-loop robot physics',mujoco_version=mujoco.__version__)
        ds_depth=ds(h,'depth_mm',(T,2,H,W),'f4',config,(1,1,H,W))
        ds_valid=ds(h,'valid',(T,2,H,W),'u1',config,(1,1,H,W))
        ds_sem=ds(h,'semantic',(T,2,H,W),'u1',config,(1,1,H,W))
        ds_inst=ds(h,'instance',(T,2,H,W),'u2',config,(1,1,H,W))
        ds_static=ds(h,'static_depth_mm',(T,2,H,W),'f4',config,(1,1,H,W))
        packed=h.create_dataset('isolated/depth_mm',shape=(0,),maxshape=(None,),chunks=(65536,),dtype='f4',
                                compression='gzip',compression_opts=config['compression_level'],shuffle=True)
        cursor=0
        for frame,t in enumerate(times):
            labels=scene.set_state(spec,t)
            before=(scene.d.time,scene.d.qpos.copy(),scene.d.mocap_pos.copy(),scene.d.mocap_quat.copy())
            for k,label in enumerate(labels):
                states[frame,k]=label['center']+label['velocity']+[label['yaw'],label['yaw_rate']]+label['dimensions']
            rgbs=[];depths=[];insts=[]
            for v,(C,E,X) in enumerate(camera_poses(spec,t)):
                camera[frame,v]=C;ego[frame,v]=E;mount[frame,v]=X
                rgb,D,V,semantic,instance,static,static_valid,surfaces=scene.view(C,spec)
                file=stage/'rgb'/('%s_%04d.jpg'%(config['camera_channels'][v],frame))
                Image.fromarray(rgb).save(file,quality=config['jpeg_quality'],subsampling=0)
                checksums.append(file_hash(file))
                ds_depth[frame,v]=D;ds_valid[frame,v]=V;ds_sem[frame,v]=semantic
                ds_inst[frame,v]=instance;ds_static[frame,v]=static
                for k,s in enumerate(surfaces):
                    boxes[frame,v,k]=s['box'];boxes_vis[frame,v,k]=s['bbox_visible']
                    contributor_counts[frame,v,k]=s['occluder_pixels']
                    area[frame,v,k]=s['area'];visible_area[frame,v,k]=s['visible']
                    visibility[frame,v,k]=s['visible']/max(1,s['area'])
                    raw=s['depth'].ravel();count=len(raw);offsets[frame,v,k]=[cursor,count]
                    if count:packed.resize((cursor+count,));packed[cursor:cursor+count]=raw;cursor+=count
                if frame in (0,T//2,T-1):
                    scene.m.geom_group[scene.m.geom_group==2]=1
                    ray_errors.extend(scene.audit_depth(C,D,instance,frame*2+v+11))
                    rgbs.append(rgb);depths.append(D);insts.append(instance)
            if (scene.d.time!=before[0] or not np.array_equal(scene.d.qpos,before[1]) or
                not np.array_equal(scene.d.mocap_pos,before[2]) or not np.array_equal(scene.d.mocap_quat,before[3])):
                raise CaptureQualityError('Simulator state changed between synchronized views')
            if rgbs:preview(stage/('preview_%04d.png'%frame),rgbs,depths,insts)
            if frame%25==0:print(json.dumps(dict(episode=name,frame=frame,total=T,elapsed_s=round(time.monotonic()-started,1))),flush=True)
        for key,value in dict(K=scene.K,camera_to_world=camera,ego_to_world=ego,camera_to_ego=mount,
            time_s=times,timestamp_us=timestamps,state=states,class_id=np.array([CLASS_IDS[a['kind']] for a in spec['actors']]),
            occluder_pixel_counts=contributor_counts,visible_fraction=visibility,full_area=area,visible_area=visible_area,bbox_full=boxes,bbox_visible=boxes_vis,
            rgb_sha256=np.array(checksums,dtype='S64').reshape(T,2)).items():h.create_dataset(key,data=value)
        h.create_dataset('isolated/boxes',data=boxes);h.create_dataset('isolated/offsets',data=offsets)
        h.attrs['state_fields']='center_x,center_y,center_z,velocity_x,velocity_y,velocity_z,yaw,yaw_rate,length,width,height'
        h.attrs['static_valid_definition']='static_depth_mm > 0'
        h.attrs['presence_definition']='All enumerated scripted targets exist throughout the episode, including outside the camera frustum.'
    metrics=dict(ray_depth_mm_p50=float(np.median(ray_errors)),ray_depth_mm_p95=float(np.percentile(ray_errors,95)),
                 ray_depth_mm_max=float(max(ray_errors)),ray_samples=len(ray_errors),
                 visible_pixels=int(visible_area.sum()),projected_pixels=int(area.sum()))
    if metrics['ray_depth_mm_p95']>2.0:raise CaptureQualityError('Optical-depth audit failed: '+str(metrics))
    b=(visibility>=config['visibility_fraction_min'])&(area>=config['projection_area_min_px'])
    pair=(2*b[:,0]+b[:,1]).astype(np.uint8)
    metrics['visibility_pair_counts']={str(i):int((pair==i).sum()) for i in range(4)}
    metrics['effective_target_frames']=int(b.sum())
    metrics['focal_causal_occluder_pixels']={str(k):int(contributor_counts[:,:,0,k].sum()) for k in range(18) if contributor_counts[:,:,0,k].sum()}
    metrics['measured_motion_events']=measured_events(b,area,states,config)
    metrics['script_event']=audit_event(pair,area,spec,config)
    if not metrics['script_event']['passed']:raise CaptureQualityError('Full sequence failed scripted event audit')
    if spec['actors'][0]['mode']=='new_target_entry' and b[:31,:,0].any():
        raise CaptureQualityError('New-entry focal target was observed before the forecast anchor')
    annotations=inspection_labels(states,b,area,config)
    atomic_json(stage/'events.json',dict(script=metrics['script_event'],measured=metrics['measured_motion_events'],inspection=annotations))
    result=dict(**plan,identity=identity,frames=T,targets=N,seconds=time.monotonic()-started,
                data_bytes=sum(p.stat().st_size for p in stage.rglob('*') if p.is_file()),metrics=metrics,
                frames_sha256=file_hash(stage/'frames.h5'),spec=spec)
    atomic_json(stage/'episode.json',result)
    atomic_json(stage/'complete.json',{k:v for k,v in result.items() if k!='spec'})
    final.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,final)
    return result


def measured_events(visible,area,state,c):
    result=[];run=c['event_min_visible_run'];T=len(visible)
    pair=(2*visible[:,0]+visible[:,1]).astype(np.uint8)
    for k in range(pair.shape[1]):
        for t in range(run,T-run+1):
            before=pair[t-1,k];after=pair[t,k]
            if before!=after and np.all(pair[t-run:t,k]==before) and np.all(pair[t:t+run,k]==after):
                result.append(dict(kind='visibility_transition',instance=k+1,frame=t,before=int(before),after=int(after)))
        for v in range(2):
            in_frame=frustum_presence(area[:,v,k])
            for t in range(1,T):
                if in_frame[t]!=in_frame[t-1]:result.append(dict(kind='fov_entry' if in_frame[t] else 'fov_exit',instance=k+1,view=v,frame=t))
        seen=np.flatnonzero(visible[:,:,k].any(axis=1))
        if len(seen) and seen[0]>0:result.append(dict(kind='first_observation',instance=k+1,frame=int(seen[0])))
    return result


def inspection_labels(states,visible,areas,c):
    """Ground-centre region entry and history-qualified directional reappearance."""
    entries=[];recoveries=[];hidden=[]
    for k in range(states.shape[1]):
        # Region definition is stored in layouts.json and invariant under rendering.
        center=states[:,k,:2];inside=(center[:,0]>=-1)&(center[:,0]<=3)&(center[:,1]>=1.6)&(center[:,1]<=4.4)
        for t in range(1,len(states)-1):
            if not inside[t-1] and inside[t] and inside[t+1]:
                entries.append(dict(instance=k+1,zone='restricted_aisle_0',frame=t,position_m=center[t].tolist(),
                                    visible_pair=(2*visible[t,0,k]+visible[t,1,k]).item()))
        for start in c['evaluation_window_starts']:
            anchor=start+c['history_frames']-1;end=anchor+c['future_frames']
            if not visible[anchor,:,k].any() and np.all(areas[anchor,:,k]>=c['projection_area_min_px']):
                hidden.append(dict(instance=k+1,start=start,historical_evidence=bool(visible[start:anchor+1,:,k].any())))
            for v in range(2):
                if not visible[:anchor,v,k].any() or visible[anchor,v,k] or areas[anchor,v,k]<c['projection_area_min_px']:continue
                for t in range(anchor+1,min(end,len(states)-3)+1):
                    if visible[t:t+3,v,k].all():
                        recoveries.append(dict(instance=k+1,start=start,view=v,reappearance=t,partner_visible_at_anchor=bool(visible[anchor,1-v,k])))
                        break
    return dict(zone_entries=entries,reappearances=recoveries,neither_visible_windows=hidden)
