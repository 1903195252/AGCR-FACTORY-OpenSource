import argparse
import json
import time
from pathlib import Path
import sys
from . import _native_runtime


def main():
    parser=argparse.ArgumentParser(description='Independent full-factory simulation; no learned policy required.')
    parser.add_argument('action',choices=['inspect','demo','sensors','verify','export-scene'])
    from .scene.registry import profiles
    parser.add_argument('--profile',choices=list(profiles()),default='factory_industrial')
    parser.add_argument('--seconds',type=float,default=2.)
    parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--appearance',choices=['industrial','legacy'],default='industrial')
    parser.add_argument('--output',type=Path,default=Path('outputs/demo'))
    parser.add_argument('--component',choices=['all','factory','uav','dog'],default='all')
    args=parser.parse_args()
    _native_runtime()
    if args.action=='verify':
        from .scene.registry import verify_authoring
        print(json.dumps(verify_authoring(args.profile),indent=2));return
    if args.action=='export-scene':
        from .scene.export import export_scene
        print(export_scene(args.profile,args.component,args.output));return
    from . import Simulator
    from .io import export_interface,export_observations,save_physics_state,jsonable
    import numpy as np
    from PIL import Image
    if not np.isfinite(args.seconds) or args.seconds<=0:parser.error('--seconds must be finite and positive')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    with Simulator(args.profile,appearance=args.appearance) as sim:
        export_interface(sim,args.output/'interfaces.json')
        initial=sim.state();samples=[]
        if args.action=='demo':
            air=initial['air']['position_m'].copy();air[2]=1.5
            ground=initial['ground']['position_m'][:2]+[.5,0.]
            sim.command(air=np.r_[air,0.],ground=np.r_[ground,0.],duration=4.)
            viewer=sim.launch_viewer() if args.viewer else None
            ticks=round(args.seconds/sim.timestep)
            for i in range(ticks):
                state=sim.step()
                if i%100==0 or i==ticks-1:
                    samples.append([sim.time,*state['air']['position_m'],*state['ground']['position_m']])
                    if viewer:
                        if not viewer.is_running():break
                        viewer.sync()
            np.savetxt(args.output/'trajectory.csv',np.asarray(samples),delimiter=',',
                       header='time_s,air_x,air_y,air_z,ground_x,ground_y,ground_z',comments='')
        if args.action in ('demo','sensors'):
            export_observations(sim.observe(width=160,height=120,rings=4,azimuth_samples=60),args.output/'sensors')
            Image.fromarray(sim.render()).save(args.output/'scene.png')
        save_physics_state(sim,args.output/'physics_state.npz')
        state=sim.state()
        report=dict(profile=args.profile,seconds_simulated=sim.time,physics_steps=sim.steps,
            appearance=sim.appearance_name,
            model_sha256=sim.model_spec['model_sha256'],mujoco_geoms=sim.model.ngeom,mujoco_meshes=sim.model.nmesh,
            warnings=[int(w.number) for w in sim.data.warning],
            air_displacement_m=state['air']['position_m']-initial['air']['position_m'],
            ground_displacement_m=state['ground']['position_m']-initial['ground']['position_m'],
            elapsed_wall_s=time.perf_counter()-started,policy_loaded=False,
            torch_imported='torch' in sys.modules,scope='Bottom-level smoke run, not full mission acceptance')
        (args.output/'report.json').write_text(json.dumps(jsonable(report),indent=2)+'\n')
        print(json.dumps(jsonable(report),indent=2))


if __name__=='__main__':main()
