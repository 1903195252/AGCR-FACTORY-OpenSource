"""Construct occluders from prescribed ray intersections and audit actual masks."""
import numpy as np
from .planning import actor_state, camera_poses
from .common import rotation_z
SIGNS=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])

def fit_fixtures(scene,spec,plan):
    focal=spec['actors'][0];obj=scene.targets[(focal['kind'],0)]
    # Fit geometry to the canonical branch; repeats cannot change past geometry.
    canonical=dict(focal,future_delta=0.,history_shift=0.)
    for view,bit in [(0,2),(1,1)]:
        hidden=[not bool(spec['before_pair']&bit),not bool(spec['after_pair']&bit)]
        if not any(hidden):continue
        yy=(spec['camera']['air_y']+3.)/2 if view==0 else -.5
        envelopes=[]
        fit_frames = [list(range(28,46))]*2 if hidden==[True,True] else [spec['before_frames'],spec['after_frames']]
        if canonical['mode']=='new_target_entry':fit_frames=[list(range(31)),spec['after_frames']]
        for frames in fit_frames:
            values=[]
            for frame in frames:
                t=frame/scene.c['fps'];C=camera_poses(spec,t)[view][0]
                p,_,yaw,_=actor_state(canonical,t,spec['duration'])
                corners=(obj['center']+SIGNS*obj['dimensions']/2)@rotation_z(yaw).T+p
                q=(yy-C[1,3])/(corners[:,1]-C[1,3])
                values.append(C[:3,3]+q[:,None]*(corners-C[:3,3]))
            q=np.concatenate(values);envelopes.append((q.min(0),q.max(0)))
        # Transition is caused by traversal of a permanent occluding edge.
        indices=[i for i,h in enumerate(hidden) if h]
        lo=np.min([envelopes[i][0] for i in indices],axis=0)
        hi=np.max([envelopes[i][1] for i in indices],axis=0)
        margin=.10+spec['physical_attempt']*.06
        x0,x1=lo[0]-margin,hi[0]+margin
        kind=plan['occluder']
        # Vehicles remain on the floor. A separate high fixture is needed for
        # aerial occlusion; record both contributors rather than levitating a vehicle.
        if kind=='vehicle' and view==1:
            slot=len(spec['actors']);vehicle=scene.targets[('loaded_forklift',slot)]
            # A near-camera parked vehicle magnifies its projected silhouette.
            C=camera_poses(spec,6.)[1][0];cy=float(C[1,3]+.18*(3.-C[1,3]))
            vehicle_bounds=[]
            for frames in [spec['before_frames'],spec['after_frames']]:
                samples=[]
                for frame in frames:
                    t=frame/scene.c['fps'];cam=camera_poses(spec,t)[1][0]
                    pp,_,yaw,_=actor_state(canonical,t,spec['duration'])
                    corners=(obj['center']+SIGNS*obj['dimensions']/2)@rotation_z(yaw).T+pp
                    q=(cy-cam[1,3])/(corners[:,1]-cam[1,3])
                    samples.extend((cam[0,3]+q*(corners[:,0]-cam[0,3])).tolist())
                vehicle_bounds.append((min(samples),max(samples)))
            heading=0.
            if hidden[0]!=hidden[1]:
                hidden_index=0 if hidden[0] else 1;visible_index=1-hidden_index
                # The former 4-cm overlap could keep a stopping target hidden
                # after the transition. Move the parked vehicle's edge away
                # from the visible phase, without changing visibility labels.
                edge=.04
                if spec.get('geometry_revision')=='event-camera-v3':edge=-.08
                if spec.get('geometry_revision') in ('event-camera-v4','event-camera-v6'):
                    # Search physical overlap instead of repeating the same
                    # vehicle placement. Labels and visibility thresholds stay fixed.
                    edge=-.08+.06*spec['physical_attempt']
                if np.mean(vehicle_bounds[hidden_index])<np.mean(vehicle_bounds[visible_index]):
                    cx=vehicle_bounds[hidden_index][1]+edge-.95
                else:
                    cx=vehicle_bounds[hidden_index][0]-edge+.95;heading=np.pi
            else:cx=(min(v[0] for v in vehicle_bounds)+max(v[1] for v in vehicle_bounds))/2
            spec['actors'].append(dict(kind='loaded_forklift',slot=slot,lane=cy,start=float(cx-1.5*np.cos(heading)),
                speed=0.,heading=float(heading),mode='straight',z=float(-vehicle['center'][2]+vehicle['dimensions'][2]/2),
                future_delta=0.,role='vehicle_occluder'))
            continue
        if kind=='vehicle':kind='suspended_load'
        low=max(.15,float(lo[2])-.35);high=min(14.,float(hi[2])+.4)
        if kind in ('rack','ground_structure'):low=0.
        support_side=1.
        if kind=='robot_arm' and spec.get('geometry_revision')=='event-camera-v6' and hidden[0]!=hidden[1]:
            # Keep the arm/support on the hidden side of the workpiece. A long
            # historical envelope otherwise scales the support into the exit
            # rays, occluding the target even after it clears the workpiece.
            hidden_index=0 if hidden[0] else 1
            hidden_x=(envelopes[hidden_index][0][0]+envelopes[hidden_index][1][0])/2
            visible_x=(envelopes[1-hidden_index][0][0]+envelopes[1-hidden_index][1][0])/2
            support_side=-1. if hidden_x<visible_x else 1.
        spec['fixtures'].append(dict(name='occluder_'+['air','ground'][view]+'_'+kind,
            kind=kind,view=view,position=[float((x0+x1)/2),yy,float((low+high)/2)],
            scale=[float((x1-x0)/2),1.,float((high-low)/2)],
            expected_hidden=hidden, support_side=support_side))
    return spec

def classify(fractions,areas,c):
    b=(fractions>=c['visibility_fraction_min'])&(areas>=c['projection_area_min_px'])
    return (2*b[:,0]+b[:,1]).astype(np.uint8)

def audit_event(pair,areas,spec,c):
    """Only the nominated target defines quota membership; all targets keep labels."""
    records=[];ok=True
    for name,frames,expected in [('before',spec['before_frames'],spec['before_pair']),
                                 ('after',spec['after_frames'],spec['after_pair'])]:
        actual=pair[frames,0]
        # Occlusion requires full projection in the image: out-of-FOV is not hidden.
        in_frame=np.all(areas[frames,:,0]>=c['projection_area_min_px'])
        passed=bool(np.all(actual==expected) and in_frame)
        records.append(dict(phase=name,frames=frames,expected_pair=expected,actual_pairs=actual.tolist(),
                            projected_in_both=bool(in_frame),passed=passed));ok &= passed
    if spec['before_pair']==spec['after_pair']:
        frames=list(range(spec['before_frames'][0],spec['after_frames'][-1]+1))
        passed=bool(np.all(pair[frames,0]==spec['before_pair']) and np.all(areas[frames,:,0]>=c['projection_area_min_px']))
        records.append(dict(phase='sustained',frames=frames,passed=passed));ok &= passed
    return dict(passed=bool(ok),focal_instance=1,checks=records,
                pair_codes={'0':'neither_visible','1':'ground_only','2':'aerial_only','3':'both_visible'})


def frustum_presence(projected_area):
    """Rasterized target support exists, independently of the usable-size threshold."""
    return np.asarray(projected_area)>0
