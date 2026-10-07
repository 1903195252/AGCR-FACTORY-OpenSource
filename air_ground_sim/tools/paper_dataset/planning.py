"""Balanced, prescribed visibility events; random draws only vary nuisance factors."""
import numpy as np
from .common import camera_basis, rotation_z

EVENT_PAIRS = {'both_visible':(3,3), 'aerial_only':(2,2), 'ground_only':(1,1),
 'neither_visible':(0,0), 'visible_to_occluded':(3,2), 'occluded_to_visible':(2,3),
 'aerial_only_to_ground_only':(2,1), 'neither_to_visible':(0,3)}
CLASS_IDS={'forklift':4,'worker':3,'agv':13,'uav':14,'loaded_forklift':4}

def plans(c):
    out=[]; counters={}
    for layout in range(c['train_layouts']+c['val_layouts']+c['test_layouts']):
        split='train' if layout<c['train_layouts'] else ('val' if layout<c['train_layouts']+c['val_layouts'] else 'test')
        groups=[('main',c['paths_per_layout'])]
        if split=='train':groups.append(('augmentation',c['augmentation_paths_per_train_layout']))
        if split=='test':groups.append(('open_entry',c['open_entry_paths_per_test_layout']))
        offset=0
        for pool,count in groups:
            key=(split,pool)
            for path in range(offset,offset+count):
                j=counters.get(key,0);counters[key]=j+1
                event=c['event_groups'][j%8] if pool!='open_entry' else 'neither_to_visible'
                cls=c['target_classes'][(j//8+j%8)%4]
                occluder=c['occluder_types'][(j//8)%5]
                motion=c['motion_modes'][(j//8+2*(j%8))%len(c['motion_modes'])]
                transition_view='air' if (j//8)%2 else 'ground'
                if occluder=='vehicle' and (event=='ground_only' or (event in ('visible_to_occluded','occluded_to_visible') and transition_view=='air')):occluder='suspended_load'
                # Visibility transitions require relative motion. Stop means a dwell
                # before/after crossing, not an impossible stationary edge crossing.
                if motion=='new_target_entry' and event!='neither_to_visible':motion='fov_entry_exit'
                if pool=='open_entry':
                    motion='new_target_entry'
                    if occluder=='vehicle':occluder='rack'
                elif motion=='new_target_entry' and occluder=='vehicle':motion='fov_entry_exit'
                for repeat in range(c['motion_repeats']):
                    out.append(dict(index=len(out),name='l%02d_p%03d_r%d'%(layout,path,repeat),layout=layout,
                        split=split,pool=pool,path=path,repeat=repeat,event=event,occluder=occluder,
                        focal_class=cls,motion=motion,transition_view=transition_view,direction=(-1 if (j//8)%2 else 1),camera_seed=[c['seed'],layout,path,61],
                        motion_seed=[c['seed'],layout,path,repeat,97]))
            offset+=count
    return out

def episode_spec(plan,c,attempt=0):
    rng=np.random.default_rng(np.random.SeedSequence(plan['motion_seed']))
    cr=np.random.default_rng(np.random.SeedSequence(plan['camera_seed']))
    duration=(c['frames_per_episode']-1)/c['fps']
    # Camera paths are identical across repeats; target motion varies within the event constraints.
    speed=1.55
    actors=[dict(kind=plan['focal_class'],slot=0,lane=3.,start=-speed*7.3,speed=speed,
                mode=plan['motion'],direction=plan['direction'],z=3.2 if plan['focal_class']=='uav' else 0.,
                future_delta=float(rng.uniform(-.10,.10)),history_shift=(-.12 if plan['repeat']==0 else .12),role='event_target')]
    for k,kind in enumerate(['worker','agv'],1):
        actors.append(dict(kind=kind,slot=k,lane=8.+k*5,start=-8.+k*3,speed=.45+.15*k,
                           mode='straight',z=0.,future_delta=0.,role='background_target'))
    camera=dict(start_x=float(cr.uniform(-1.,1.)),vx=float(cr.uniform(-.07,.07)),
        air_y=float(cr.uniform(7.5,8.5)),air_z=float(cr.uniform(10.5,12.5)),ground_y=float(cr.uniform(-14.,-12.)),ground_z=float(cr.uniform(.85,1.15)),
        air_pitch=-1.09,ground_pitch=.04)
    if plan['focal_class']=='uav':
        camera.update(start_x=float(cr.uniform(-.2,.2)),vx=float(cr.uniform(-.025,.025)),air_y=6.,air_z=7.,ground_y=-3.,ground_pitch=.35,air_pitch=-.90)
        if plan.get('geometry_revision') in ('event-camera-v3','event-camera-v4','event-camera-v6'):
            # Keep the small target above the existing 144-pixel threshold and
            # inside the aerial frustum through the prescribed event. Both
            # motion realizations use the same revised camera path.
            camera.update(air_z=8., ground_y=-2.,
                          ground_pitch=float(np.arctan2(3.2-camera['ground_z'], 5.)))
    before,after=EVENT_PAIRS[plan['event']]
    if plan['transition_view']=='air':
        if plan['event']=='visible_to_occluded':after=1
        if plan['event']=='occluded_to_visible':before=1
    return dict(actors=actors,camera=camera,duration=duration,physical_attempt=attempt,
        trajectory_mode=c['trajectory_mode'],event=plan['event'],occluder=plan['occluder'],
        before_pair=before,after_pair=after,
        before_frames=c['event_before_frames'],after_frames=c['event_after_frames'],fixtures=[],
        geometry_revision=plan.get('geometry_revision'))

def actor_state(a,t,duration):
    t=np.asarray(t,dtype=float);v=a['speed'];mode=a['mode']
    # Brake before the occupied production cells at the aisle ends.
    real_t=t.copy();brake=np.clip(t-12.,0,2.)
    time_rate=np.where(t<12.,1.,np.where(t<14.,1-brake/2,0.))
    time_acc=np.where((t>=12.)&(t<14.),-.5,0.)
    t=np.minimum(t,12.)+brake-brake*brake/4
    x=a['start']+a.get('history_shift',0.)+v*t;vx=np.full_like(t,v);ax=np.zeros_like(t)
    y=np.full_like(t,a['lane']);vy=np.zeros_like(t);ay=np.zeros_like(t)
    # Continuous position and velocity. Abrupt changes have finite acceleration.
    if mode in ('sudden_acceleration','sudden_deceleration'):
        sign=1 if mode=='sudden_acceleration' else -1
        h=t-7.4;delta=sign*.55
        x=x+delta*.3*(np.logaddexp(0,h/.3)-np.logaddexp(0,-7.4/.3))
        sig=1/(1+np.exp(-np.clip(h/.3,-100,100)))
        vx=v+delta*sig;ax=delta/.3*sig*(1-sig)
    elif mode=='stop':
        # Brake within the forecast, then dwell during the post-event phase.
        h=np.clip(t-7.8,0,.8)
        x=a['start']+a.get('history_shift',0.)+v*(np.minimum(t,7.8)+h-h*h/1.6)
        vx=np.where(t<7.8,v,np.where(t<8.6,v*(1-h/.8),0.));ax=np.where((t>=7.8)&(t<8.6),-v/.8,0.)
    elif mode=='turn':
        bend=np.tanh((t-8.)/.6);y=y-.65*bend
        vy=-.65/.6*(1-bend*bend);ay=2*.65/.6**2*bend*(1-bend*bend)
    # Two futures branch only after t=6 s, the anchor of the paired-history task.
    h=np.maximum(t-6.,0);d=0. if mode=='stop' else a.get('future_delta',0.)
    x=x+d*h*h/(1+h);vx=vx+d*h*(2+h)/(1+h)**2;ax=ax+2*d/(1+h)**3
    direction=a.get('direction',1);x=x*direction;vx=vx*direction;ax=ax*direction
    heading=np.where(vx*vx+vy*vy>1e-10,np.arctan2(vy,vx),np.pi if direction<0 else 0.)
    ax=ax*time_rate**2+vx*time_acc;ay=ay*time_rate**2+vy*time_acc
    vx=vx*time_rate;vy=vy*time_rate
    yaw=np.full_like(t,a['heading']) if 'heading' in a else heading
    rate=(vx*ay-vy*ax)/np.maximum(vx*vx+vy*vy,1e-8)
    z=np.full_like(t,a.get('z',0.));zero=np.zeros_like(t)
    return np.stack([x,y,z],-1),np.stack([vx,vy,zero],-1),yaw,rate

def camera_poses(spec,t):
    c=spec['camera'];out=[]
    for view in ['air','ground']:
        yaw=-np.pi/2 if view=='air' else np.pi/2
        pos=[c['start_x']+c['vx']*t,c[view+'_y'],c[view+'_z']]
        E=np.eye(4);E[:3,:3]=rotation_z(yaw);E[:3,3]=pos
        E[2,3]-=.18 if view=='air' else .65
        C=np.eye(4);C[:3,:3]=camera_basis(yaw,c[view+'_pitch']);C[:3,3]=pos
        out.append((C,E,np.linalg.inv(E)@C))
    return out
