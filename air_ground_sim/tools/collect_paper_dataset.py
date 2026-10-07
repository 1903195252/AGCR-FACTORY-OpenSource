#!/usr/bin/env python3
"""Collect event-qualified air-ground RGB-D episodes (separate from UAV-only legacy)."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import zipfile
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')
os.environ.setdefault('MUJOCO_GL','egl')
from paper_dataset.common import load_config, atomic_json, digest, file_hash, pipeline_hash
from paper_dataset.planning import plans
from paper_dataset.scene import make_layout, build_scene

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--assets',required=True);p.add_argument('--out',required=True)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--limit',type=int);p.add_argument('--indices',help='Comma-separated plan indices; pilot only')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--pilot',action='store_true')
    args=p.parse_args();c=load_config(args.config);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.workers<1 or not 0<=args.worker<args.workers:raise ValueError('Invalid worker assignment')
    identity=digest(dict(config=c,assets=file_hash(args.assets),pipeline=pipeline_hash(),purpose='pilot' if args.pilot else 'formal'))
    with (out/'prepare.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest=out/'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text())['identity']!=identity:raise ValueError('Dataset identity mismatch; use a new output directory')
        else:
            assetdir=out/'assets';assetdir.mkdir(exist_ok=True)
            with zipfile.ZipFile(args.assets) as z:
                for name in z.namelist():
                    if not (assetdir/name).resolve().is_relative_to(assetdir.resolve()):raise ValueError('Unsafe archive path')
                z.extractall(assetdir)
            assignments=plans(c)
            atomic_json(out/'plans.json',assignments);atomic_json(out/'config.json',c)
            layouts=[]
            for i in range(c['train_layouts']+c['val_layouts']+c['test_layouts']):
                layout=make_layout(i,c);layouts.append(layout)
                build_scene(assetdir,layout,out/'layouts'/('%02d.xml'%i),c)
            atomic_json(out/'layouts.json',layouts)
            atomic_json(manifest,dict(identity=identity,purpose='pilot' if args.pilot else 'formal',
                 planned_episodes=len(assignments),created_unix=time.time(),pipeline_sha256=pipeline_hash(),
                 asset_sha256=file_hash(args.assets),config=c))
    if args.prepare_only:return
    from paper_dataset.capture import SceneCapture,collect_episode
    jobs=json.loads((out/'plans.json').read_text())
    if args.indices:
        wanted={int(i) for i in args.indices.split(',')};jobs=[j for j in jobs if j['index'] in wanted]
    jobs=[j for j in jobs if j['index']%args.workers==args.worker]
    if args.limit:jobs=jobs[:args.limit]
    with (out/('worker_%d.lock'%args.worker)).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scene=None;layout=None;failed=0
        try:
            for job in jobs:
                if (out/'STOP').exists():break
                if layout!=job['layout']:
                    if scene:scene.close()
                    scene=SceneCapture(out/'layouts'/('%02d.xml'%job['layout']),c);layout=job['layout']
                try:
                    result=collect_episode(scene,job,c,out,identity)
                    print(json.dumps(dict(status='complete',name=job['name'],seconds=result['seconds'],metrics=result['metrics'])),flush=True)
                except Exception as exc:
                    failed+=1
                    atomic_json(out/'rejected'/(job['name']+'.json'),dict(plan=job,error=str(exc),time=time.time()))
                    print(json.dumps(dict(status='rejected',name=job['name'],error=str(exc))),flush=True)
                    if args.pilot or 'disk reserve' in str(exc) or isinstance(exc,OSError):raise
                atomic_json(out/('worker_%d.status.json'%args.worker),dict(last=job['name'],failed=failed,updated=time.time()))
        finally:
            if scene:scene.close()
    if failed:sys.exit(2)

if __name__=='__main__':main()
