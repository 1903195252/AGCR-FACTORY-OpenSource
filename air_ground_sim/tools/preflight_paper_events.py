#!/usr/bin/env python3
"""Audit prescribed events before any production collection; retain failure evidence."""
import argparse,json,os,time
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'1')
from paper_dataset.common import atomic_json
from paper_dataset.capture import SceneCapture,preview
from paper_dataset.planning import camera_poses

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--count',type=int,default=40)
    a=p.parse_args();root=Path(a.root);c=json.loads((root/'config.json').read_text())
    jobs=[j for j in json.loads((root/'plans.json').read_text()) if j['pool']=='main' and j['repeat']==0][:a.count]
    report=[];scene=None;layout=None;start=time.monotonic()
    try:
        for job in jobs:
            if layout!=job['layout']:
                if scene:scene.close()
                scene=SceneCapture(root/'layouts'/('%02d.xml'%job['layout']),c);layout=job['layout']
                print(json.dumps({'dimensions':{kind:scene.targets[(kind,0)]['dimensions'].tolist() for kind in ['worker','forklift','agv','uav']}}),flush=True)
            try:
                spec=scene.spec(job);record=dict(plan=job,passed=True,audit=spec['event_preflight'])
                for frame in [30,44]:
                    t=frame/c['fps'];scene.set_state(spec,t);views=[scene.view(C,spec) for C,_,_ in camera_poses(spec,t)]
                    target=root/'preflight'/('%s_%03d.png'%(job['name'],frame));target.parent.mkdir(exist_ok=True)
                    preview(target,[v[0] for v in views],[v[1] for v in views],[v[4] for v in views])
            except Exception as e:record=dict(plan=job,passed=False,error=str(e))
            report.append(record);atomic_json(root/'preflight.json',report)
            print(json.dumps(dict(name=job['name'],event=job['event'],occluder=job['occluder'],target=job['focal_class'],passed=record['passed'],error=record.get('error',''),elapsed=time.monotonic()-start)),flush=True)
    finally:
        if scene:scene.close()
if __name__=='__main__':main()
