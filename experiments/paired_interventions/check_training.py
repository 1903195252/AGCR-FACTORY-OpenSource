"""Native short-update and DDP resume acceptance, in independent run folders."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from .data import file_sha256
from .train import atomic_json


def compare(left,right):
    a=torch.load(left,map_location='cpu',weights_only=False);b=torch.load(right,map_location='cpu',weights_only=False)
    error=max(float((a['model'][k].float()-v.float()).abs().max()) for k,v in b['model'].items() if v.numel())
    optimizer=max(float((v-b['optimizer']['state'][k][name]).abs().max())
        for k,row in a['optimizer']['state'].items() for name,v in row.items() if isinstance(v,torch.Tensor))
    rng=[all(torch.equal(x[k],y[k]) for k in ('torch','cuda','generator')) for x,y in zip(a['rank_rng'],b['rank_rng'])]
    same_sampler=all(a[k]==b[k] for k in ('family_sampler','ordinary_sampler','informative_sampler'))
    # BF16 CUDA kernels can change accumulation order; this is a short-update
    # tolerance for parameters/Adam moments, not an image-quality threshold.
    return dict(passed=error<=1e-6 and optimizer<=1e-5 and all(rng) and same_sampler,
        parameter_max_abs=error,optimizer_max_abs=optimizer,parameter_tolerance=1e-6,optimizer_tolerance=1e-5,
        rank_rng_equal=rng,sampler_equal=same_sampler)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--paths',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    paths=json.loads(Path(a.paths).read_text());parent=file_sha256(paths['upstream'])
    pause=file_sha256(paths['pause_marker'])
    def run(name,args,world=1):
        prefix=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node='+str(world),'--module'] if world>1 else [sys.executable,'-m']
        command=prefix+args
        with (root/(name+'.log')).open('w') as log:
            print('Running '+name,flush=True);subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    run('unit',['unittest','experiments.paired_interventions.test_interventions','experiments.paired_interventions.test_training_contracts','-v'])
    run('native',['experiments.paired_interventions.check_native','--paths',a.paths,'--output',str(root/'native'),'--decode'])
    def training(name,phase,world=2,init=None,extra=()):
        args=['experiments.paired_interventions.train','--paths',a.paths,'--output',str(root/name),
            '--phase',phase,'--steps','2','--batch',str(world),'--development','--families-only','--gpu-hours','0.6']
        if init: args+=['--init',str(init)]
        return args+list(extra)
    run('dynamics',training('dynamics','dynamics',extra=['--validate-every','1','--validation-panel','4']),2)
    run('depth',training('depth','depth',init=root/'dynamics/last.pt'),2)
    first=training('resume','video',init=root/'depth/last.pt',extra=['--force-feedback','--stop-after','1'])
    run('video_split',first,2)
    second=training('resume','video',extra=['--force-feedback','--resume',str(root/'resume/last.pt')])
    run('video_resume',second,2)
    run('video_continuous',training('continuous','video',init=root/'depth/last.pt',extra=['--force-feedback']),2)
    comparison=compare(root/'resume/last.pt',root/'continuous/last.pt')
    atomic_json(root/'resume_comparison.json',comparison)
    statuses=[json.loads((root/name/'status.json').read_text()) for name in ('dynamics','depth','resume','continuous')]
    gradient_keys=['decoded_gradient_predicted_future_state','decoded_gradient_history_track_tokens','decoded_gradient_history_scene','decoded_gradient_history_initial_state']
    gradient_ok=all(statuses[-1]['metrics'].get(k,0)>0 for k in gradient_keys)
    unchanged=parent==file_sha256(paths['upstream']) and pause==file_sha256(paths['pause_marker'])
    native=json.loads((root/'native/report.json').read_text())
    report=dict(scope='native_joint_training_acceptance',passed=comparison['passed'] and gradient_ok and unchanged and native['passed'],
        development_only=True,benchmark_results=False,source_sha256={p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')},
        native=native,resume=comparison,decoded_feedback_reaches_history=gradient_ok,parent_and_pause_unchanged=unchanged,
        parent_sha256=parent,statuses=statuses)
    atomic_json(root/'acceptance.json',report)
    if not report['passed']: raise SystemExit('Native training acceptance did not pass')
    print(json.dumps(dict(passed=True,report=str(root/'acceptance.json'))),flush=True)


if __name__=='__main__': main()
