"""Prescribed native 2-world x 2-camera factory interventions.

Layout ancestry follows the existing train/val/test layout partition. Motion
and fixture placement are deterministic; only nuisance offsets use a seed.
Every cell is independently rendered and must pass the pixel-level audit.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def trajectory(times, cohort, speed=2., bend=.65, direction=1):
    """Two smooth turns start inside history, or strictly after ambiguity time."""
    start = -.28 if cohort == 'evidence' else .04
    h = np.maximum(times-start, 0.)
    lateral = bend*h*h/(.35+h)
    lateral_velocity = bend*h*(.7+h)/(.35+h)**2
    centers = np.zeros((2, len(times), 3), np.float64)
    angles = np.zeros((2, len(times)), np.float64)
    for world, sign in enumerate((-1., 1.)):
        centers[world, :, 0] = direction*speed*times
        centers[world, :, 1] = 3.+sign*lateral
        angles[world] = np.arctan2(sign*lateral_velocity, direction*speed)
    return centers, angles


def make_plan(dataset_root, output, layout, split, cohort, receiver, serial,
              target_class='forklift', occluder='rack', margin=.3):
    from air_ground_sim.tools.paper_dataset.capture import SceneCapture
    from air_ground_sim.tools.paper_dataset.common import camera_basis, rotation_z

    dataset_root, output = Path(dataset_root).resolve(), Path(output).resolve()
    config = json.loads((dataset_root/'config.json').read_text())
    expected = ('train' if layout < config['train_layouts'] else
                'val' if layout < config['train_layouts']+config['val_layouts'] else 'test')
    if split != expected:
        raise ValueError('Layout ancestry disagrees with the existing dataset split')
    family = 'l%02d_%s_r%d_%s_%s_%03d' % (layout, cohort, receiver, target_class, occluder, serial)
    directory = output/family
    directory.mkdir(parents=True, exist_ok=False)
    xml = dataset_root/'layouts'/('%02d.xml' % layout)
    scene = SceneCapture(xml, config)
    try:
        times = np.linspace(-1., 3.2, 211)
        stamps = (np.arange(22)-5)*.2
        indices = np.rint((stamps+1.)/.02).astype(int)
        speed = dict(forklift=2., worker=1.5, agv=1.8, uav=1.6)[target_class]
        centers, angles = trajectory(times, cohort, speed=speed)
        nuisance = np.random.default_rng(np.random.SeedSequence([20261003, layout, serial]))
        centers[..., 0] += float(nuisance.uniform(-.15, .15))
        target = scene.targets[(target_class, 0)]
        centers[..., 2] = target['dimensions'][2]/2 + (3.0 if target_class == 'uav' else .025)
        background = scene.targets[('worker', 1)]
        actors = [dict(kind=target_class, slot=0, role='intervention_target'),
                  dict(kind='worker', slot=1, role='background_target')]
        origins = np.empty((2, len(times), 2, 3), np.float64)
        yaw = np.zeros((2, len(times), 2), np.float64)
        for world in range(2):
            yaw[world, :, 0] = angles[world]
            for tick in range(len(times)):
                origins[world, tick, 0] = centers[world, tick]-rotation_z(angles[world, tick])@target['center']
            origins[world, :, 1] = np.array([5., 7., background['dimensions'][2]/2+.025])-background['center']
        camera_base = []
        # The native UAV is small: the original 15 m ground stand-off leaves
        # only ~30 pixels, below the fixed 144-pixel evidence threshold. Move
        # the sensors closer for this class instead of weakening acceptance
        # or scaling the physical target. Intrinsics remain unchanged.
        mounts=([([0.,7.,6.],-np.pi/2,-.61),([0.,-2.,1.],np.pi/2,.35)] if target_class=='uav' else
                [([0.,7.,10.],-np.pi/2,-1.09),([0.,-12.,1.],np.pi/2,.04)])
        for pos, azimuth, pitch in mounts:
            C = np.eye(4)
            C[:3, :3] = camera_basis(azimuth, pitch)
            C[:3, 3] = pos
            camera_base.append(C)
        cameras = np.broadcast_to(np.stack(camera_base), (2, 22, 2, 4, 4)).copy()
        # Smooth passive translation; all historical matrices are untouched.
        phase = np.maximum(stamps, 0.)/3.2
        shift = .9*(3*phase**2-2*phase**3)
        cameras[1, :, receiver, 0, 3] += shift
        # Fit common permanent occluders to BOTH worlds' historical envelopes.
        # They are never switched on/off, and camera alternatives use them unchanged.
        signs = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        fixtures = []
        hidden_views = (receiver,) if cohort == 'evidence' else (0, 1)
        for view in hidden_views:
            # Keep the suspended screen above the vehicle's entire swept box.
            # At y=5 the historical viewing cone reaches down into the turn.
            plane_y = -.5 if view == 1 else 6.0
            rays = []
            for world in range(2):
                for frame in (4, 5):
                    tick = indices[frame]
                    corners = (signs*target['dimensions']/2)@rotation_z(angles[world, tick]).T+centers[world, tick]
                    C = camera_base[view]
                    ratio = (plane_y-C[1,3])/(corners[:,1]-C[1,3])
                    rays.append(C[:3,3]+ratio[:,None]*(corners-C[:3,3]))
            ray = np.concatenate(rays)
            lo, hi = ray.min(0), ray.max(0)
            # Near the aerial camera, a small world-space margin subtends a
            # large angle. Scale it with screen distance to retain pre-event
            # visibility, while the exact receiver-history audit guards leaks.
            local_margin=(.04 if target_class=='uav' else margin) if view==1 else margin*.2
            x0, x1 = lo[0]-local_margin, hi[0]+local_margin
            low, high = max(.1, lo[2]-.4), min(14., hi[2]+.4)
            kind = occluder if view == 1 else 'suspended_load'
            if kind in ('rack', 'ground_structure'):
                low = 0.
            fixtures.append(dict(name='occluder_'+('air' if view==0 else 'ground')+'_'+kind,
                kind=kind, view=view, position=[float((x0+x1)/2), plane_y, float((low+high)/2)],
                scale=[float((x1-x0)/2),1. if view==1 else .1,float((high-low)/2)], support_side=1.))
        body = np.eye(4)
        body[:3,:3] = rotation_z(np.pi/2)
        body[:3,3] = [0.,camera_base[1][1,3],.35]
        spec = dict(actors=actors, fixtures=fixtures, duration=4.2,
                    trajectory_mode='prescribed_counterfactual_replay', cohort=cohort)
        (directory/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        (directory/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
        np.savez(directory/'trajectory.npz', time_s=times, origins=origins, yaw=yaw,
                 camera_to_world=cameras, ground_body_to_world_at_t=body)
        plan = dict(xml=str(xml), config='config.json', spec='spec.json', trajectory='trajectory.npz',
            family_id=family, scene_parent_id='native_layout_%02d_intervention_%03d'%(layout,serial),
            layout=layout, split=split, cohort=cohort, receiver=receiver, target_index=0,
            nuisance_seed=[20261003,layout,serial], target_class=target_class, occluder=occluder)
        plan['planner_revision']=2
        (directory/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        return directory/'plan.json'
    finally:
        scene.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--layout',type=int,required=True)
    p.add_argument('--split',choices=['train','val','test'],required=True)
    p.add_argument('--cohort',choices=['evidence','ambiguous'],required=True)
    p.add_argument('--receiver',type=int,choices=[0,1],default=1)
    p.add_argument('--serial',type=int,default=0)
    p.add_argument('--target-class',choices=['forklift','worker','agv','uav'],default='forklift')
    p.add_argument('--occluder',choices=['rack','robot_arm','suspended_load','ground_structure'],default='rack')
    p.add_argument('--margin',type=float,default=.3)
    a=p.parse_args()
    print(make_plan(a.dataset_root,a.output,a.layout,a.split,a.cohort,a.receiver,a.serial,
                    a.target_class,a.occluder,a.margin))


if __name__=='__main__':
    main()
