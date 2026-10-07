"""Execute registered internal factorial runs; never substitute external methods."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .pipeline import stages
from .train import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--paths',required=True);p.add_argument('--output',required=True)
    p.add_argument('--variants',nargs='+',default=['full']);p.add_argument('--seeds',nargs='+',type=int,default=[17,29,43])
    p.add_argument('--world-size',type=int,default=2);p.add_argument('--gpu-hours',type=float,required=True)
    p.add_argument('--run',action='store_true');a=p.parse_args()
    protocol=json.loads(Path(__file__).with_name('protocol.json').read_text())
    allowed={v['id'] for v in protocol['essential_variants']}
    if set(a.variants)-allowed: raise ValueError('No complete registered implementation for: '+str(set(a.variants)-allowed))
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True);jobs=[]
    for variant in a.variants:
        for seed in a.seeds:
            run=root/(variant+'_seed'+str(seed))
            jobs.append(dict(variant=variant,seed=seed,run=str(run),stages=stages(a.paths,run,variant,seed,a.world_size)))
    plan=dict(schema='agcr-intervention-matrix-1',jobs=jobs,results=None,
        scope='registered_internal_factorial_training_and_paired_generation',
        external_adaptations='SynCamMaster and ShareVerse require their complete adapters; not relabelled internal controls',
        downstream='independent tracker training and final external visual metrics are separate jobs')
    atomic_json(root/'matrix.json',plan)
    if not a.run:
        print(json.dumps(dict(plan=str(root/'matrix.json'),training_runs=len(jobs),started=False)));return
    paths=json.loads(Path(a.paths).read_text())
    if Path(paths['pause_marker']).exists(): raise RuntimeError('Formal experiment queue is user-paused')
    start=time.monotonic()
    for job in jobs:
        remaining=a.gpu_hours-(time.monotonic()-start)*a.world_size/3600
        if remaining<=0: break
        command=[sys.executable,'-m','experiments.paired_interventions.pipeline','--paths',a.paths,'--output',job['run'],
            '--variant',job['variant'],'--seed',str(job['seed']),'--world-size',str(a.world_size),'--gpu-hours',str(remaining),'--run']
        subprocess.run(command,check=True)
        status=json.loads((Path(job['run'])/'status.json').read_text())
        if status['state']!='training_complete': break
        for split in ('val','test'):
            remaining=a.gpu_hours-(time.monotonic()-start)*a.world_size/3600
            if remaining<=0: return
            command=[sys.executable,'-m','experiments.paired_interventions.sample','--paths',a.paths,
                '--checkpoint',str(Path(job['run'])/('video_seed%d'%job['seed'])/'best.pt'),
                '--output',str(Path(job['run'])/('predictions_'+split)),'--split',split,'--gpu-hours',str(remaining)]
            subprocess.run(command,check=True)


if __name__=='__main__': main()
