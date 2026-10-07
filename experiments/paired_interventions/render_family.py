"""Render an explicit paired replay using the existing factory SceneCapture.

No random episode sampling, image copying, or physics integration between views.
The input is a reviewed event script, not a pre-existing batch relabeled as pairs.
Usage: python3 -m experiments.paired_interventions.render_family plan.json outdir
See experiments/README.md for the dense trajectory contract and planning prerequisites.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def render(plan_path, output):
    # Delay OpenGL initialization until the explicit rendering command.
    from air_ground_sim.tools.paper_dataset.capture import SceneCapture, local_bounds
    from air_ground_sim.tools.paper_dataset.common import rotation_z
    from air_ground_sim.tools.paper_dataset.planning import CLASS_IDS
    import mujoco
    from .audit import audit_manifest

    plan_path, output = Path(plan_path), Path(output)
    plan = json.loads(plan_path.read_text())
    base = plan_path.parent
    resolve = lambda name: (base / plan[name]).resolve()
    if output.exists():
        raise FileExistsError('Use a new family directory; existing captures are never overwritten')
    output.mkdir(parents=True)
    config = json.loads(resolve('config').read_text())
    spec = json.loads(resolve('spec').read_text())
    scene = SceneCapture(resolve('xml'), config)
    try:
        with np.load(resolve('trajectory'), allow_pickle=False) as file:
            replay = {key: file[key] for key in ('time_s','origins','yaw','camera_to_world','ground_body_to_world_at_t')}
        times, origins, yaw, cameras = (replay[key] for key in ('time_s','origins','yaw','camera_to_world'))
        actors = spec['actors']; count = len(actors)
        if origins.shape != (2,len(times),count,3) or yaw.shape != (2,len(times),count):
            raise ValueError('Dense origins/yaw must cover two worlds and all actors')
        if cameras.shape != (2,22,2,4,4):
            raise ValueError('Need two passive camera paths at all 6+16 paired timestamps')
        if len(times)<211 or not np.allclose(np.diff(times),.02,atol=1e-8,rtol=0) or not np.isclose(times[0],-1) or not np.isclose(times[-1],3.2):
            raise ValueError('Dense world scripts must cover -1..3.2 seconds at 50 Hz')
        if not all(np.isfinite(value).all() for value in replay.values()):
            raise ValueError('Nonfinite trajectory')
        body=replay['ground_body_to_world_at_t']
        if body.shape!=(4,4) or not np.allclose(body[3],[0,0,0,1]) or not np.allclose(body[:3,:3].T@body[:3,:3],np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(body[:3,:3]),1):
            raise ValueError('Ground-body reference pose must be a rigid transform')
        heading=np.arctan2(body[1,0],body[0,0]); local_R=rotation_z(heading).T
        world_from_factory=np.eye(4);world_from_factory[:3,:3]=local_R
        world_from_factory[:3,3]=-local_R@body[:3,3]
        scene.configure_fixtures(spec)
        low,high = list(scene.static_lo),list(scene.static_hi)
        for fixture in spec['fixtures']:
            body = scene.m.body(fixture['name']).id
            scene.d.mocap_pos[scene.m.body_mocapid[body]] = fixture['position']
        mujoco.mj_forward(scene.m,scene.d)
        for fixture in spec['fixtures']:
            body=scene.m.body(fixture['name']).id
            for gid in np.flatnonzero(scene.m.geom_bodyid==body):
                center,extent=local_bounds(scene.m,scene.d,[gid],0)
                low.append(center-extent/2); high.append(center+extent/2)
        low,high=np.asarray(low),np.asarray(high)
        # The same conservative 50-Hz AABB clearance convention as capture.py.
        for world in range(2):
            for tick in range(len(times)):
                boxes=[]
                for k,actor in enumerate(actors):
                    target=scene.targets[(actor['kind'],actor['slot'])]
                    R=rotation_z(yaw[world,tick,k]); center=origins[world,tick,k]+R@target['center']
                    extent=np.abs(R)@(target['dimensions']/2); lo,hi=center-extent,center+extent
                    hit=np.flatnonzero(np.all(hi>low+.012,axis=1)&np.all(lo<high-.012,axis=1))
                    if len(hit):
                        raise ValueError('Scripted trajectory intersects factory/fixture: '+json.dumps(dict(
                            world=world,tick=tick,actor=k,time=float(times[tick]),box=[lo.tolist(),hi.tolist()],
                            static_count=len(scene.static_lo),obstacles=[dict(index=int(i),low=low[i].tolist(),high=high[i].tolist()) for i in hit])))
                    if any(np.all(hi>a+.012)&np.all(lo<b-.012) for a,b in boxes):
                        raise ValueError('Scripted actor collision at '+str((world,tick,k)))
                    boxes.append((lo,hi))
        velocity=np.gradient(origins,.02,axis=1)
        rate=np.gradient(np.unwrap(yaw,axis=1),.02,axis=1)
        stamps=(np.arange(22)-5)*.2
        indices=np.rint((stamps-times[0])/.02).astype(int)
        records=[]; ray_errors=[]
        # MuJoCo XML assets and rendering configuration are common to all cells.
        env_hash=hashlib.sha256((resolve('xml').read_bytes()+resolve('config').read_bytes()+resolve('spec').read_bytes())).hexdigest()
        for world in range(2):
            for path in range(2):
                collected={key:[] for key in ('rgb','depth_mm','valid','instance','semantic','static_depth_mm','static_valid',
                                              'isolated_depth_mm','isolated_support','visible_fraction','projection_area','world_state')}
                for frame,index in enumerate(indices):
                    scene.d.mocap_pos[:]=[0,0,-100]; scene.d.mocap_quat[:]=[1,0,0,0]
                    for fixture in spec['fixtures']:
                        scene.d.mocap_pos[scene.m.body_mocapid[scene.m.body(fixture['name']).id]]=fixture['position']
                    states=[]
                    for k,actor in enumerate(actors):
                        target=scene.targets[(actor['kind'],actor['slot'])]; angle=yaw[world,index,k]
                        R=rotation_z(angle); offset=R@target['center']
                        scene.d.mocap_pos[target['mid']]=origins[world,index,k]
                        scene.d.mocap_quat[target['mid']]=[np.cos(angle/2),0,0,np.sin(angle/2)]
                        u=velocity[world,index,k]+rate[world,index,k]*np.array([-offset[1],offset[0],0])
                        states.append(np.r_[origins[world,index,k]+offset,u,angle,rate[world,index,k],target['dimensions']])
                    scene.d.time=float(stamps[frame]); mujoco.mj_forward(scene.m,scene.d)
                    views=[scene.view(cameras[path,frame,v],spec) for v in range(2)]
                    if frame in (0,5,6,13,21):
                        for v,view in enumerate(views):
                            ray_errors.extend(scene.audit_depth(cameras[path,frame,v],view[1],view[4],frame*2+v+11))
                    for name,position in (('rgb',0),('depth_mm',1),('valid',2),('semantic',3),('instance',4),('static_depth_mm',5),('static_valid',6)):
                        collected[name].append(np.stack([view[position] for view in views]))
                    isolated=np.zeros((2,count,config['height'],config['width']),np.float32)
                    fractions=np.zeros((2,count),np.float32); areas=np.zeros((2,count),np.int32)
                    for v,view in enumerate(views):
                        for k,surface in enumerate(view[-1]):
                            x0,y0,x1,y1=surface['box']; isolated[v,k,y0:y1,x0:x1]=surface['depth']
                            fractions[v,k]=surface['visible']/max(1,surface['area']); areas[v,k]=surface['area']
                    collected['isolated_depth_mm'].append(isolated)
                    collected['isolated_support'].append(isolated>0)
                    collected['visible_fraction'].append(fractions); collected['projection_area'].append(areas)
                    collected['world_state'].append(states)
                arrays={key:np.asarray(value) for key,value in collected.items()}
                # Match the paper/old loader: fixed frame at ground body at t,
                # x along its horizontal heading, z up; never reset per frame.
                arrays['world_state'][...,:3]=arrays['world_state'][...,:3]@local_R.T+world_from_factory[:3,3]
                arrays['world_state'][...,3:6]=arrays['world_state'][...,3:6]@local_R.T
                arrays['world_state'][...,6]=(arrays['world_state'][...,6]-heading+np.pi)%(2*np.pi)-np.pi
                arrays.update(time_s=stamps,camera_to_world=world_from_factory@cameras[path],
                              K=np.broadcast_to(scene.K,(22,2,3,3)).copy(),present=np.ones((22,count),bool),
                              class_id=np.array([CLASS_IDS[a['kind']] for a in actors],np.int64),
                              instance_id=np.arange(1,count+1,dtype=np.int64),world_from_factory=world_from_factory)
                name='world%d_camera%d.npz'%(world,path)
                np.savez_compressed(output/name,**arrays)
                record={key:plan[key] for key in ('family_id','scene_parent_id','split','cohort','receiver','target_index','nuisance_seed')}
                record.update(path=name,world=world,camera=path,environment_sha256=env_hash,passive_camera=True)
                records.append(record)
        (output/'manifest.json').write_text(json.dumps(dict(records=records),indent=2)+'\n')
        ray_report=dict(samples=len(ray_errors),p95_mm=float(np.percentile(ray_errors,95)),max_mm=float(max(ray_errors)))
        (output/'depth_audit.json').write_text(json.dumps(ray_report,indent=2)+'\n')
        if ray_report['p95_mm']>2.:
            raise ValueError('Native optical-depth ray audit exceeds 2 mm at p95')
        result=audit_manifest(output/'manifest.json')
        (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
        (output/'complete.json').write_text(json.dumps(dict(passed=True,family_id=plan['family_id']))+'\n')
        return result
    finally:
        scene.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan'); parser.add_argument('output')
    args=parser.parse_args()
    result=render(args.plan,args.output)
    print(json.dumps(dict(passed=result['passed'],families=len(result['families']))))
