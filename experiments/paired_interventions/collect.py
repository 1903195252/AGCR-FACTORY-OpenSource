"""Collect controlled families with bounded workers and per-family failures."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
import os
import multiprocessing
import signal
from pathlib import Path
import time
import traceback


def initialize_renderer_worker(gpus):
    # EGL initializes a process-global display: changing devices per job does
    # not move that display. Bind each worker once, before importing MuJoCo.
    identity = multiprocessing.current_process()._identity[-1]
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str((identity - 1) % gpus)
    os.environ['AGCR_RENDER_DEVICE_BOUND'] = '1'


def bounded_capture(jobs, workers, gpus, stopping):
    iterator = iter(jobs)
    with ProcessPoolExecutor(workers, initializer=initialize_renderer_worker,
                             initargs=(gpus,)) as pool:
        pending = set()
        def replenish():
            while len(pending) < workers and not stopping[0]:
                job = next(iterator, None)
                if job is None:
                    break
                pending.add(pool.submit(capture_one, job))
        replenish()
        while pending:
            done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
            replenish()


def capture_one(job):
    # Import the renderer only after selecting the EGL device for this worker.
    if os.environ.get('AGCR_RENDER_DEVICE_BOUND') != '1':
        os.environ['MUJOCO_EGL_DEVICE_ID']=str(job['gpu'])
    from .plan_families import make_plan
    from .render_family import render
    root=Path(job['output'])
    started=time.monotonic()
    try:
        name='l%02d_%s_r%d_%s_%s_%03d'%(job['layout'],job['cohort'],job['receiver'],
                job['target_class'],job['occluder'],job['serial'])
        destination=root/'families'/name
        accepted=[p.parent for p in (root/'families').glob(name+'*/complete.json')
                  if json.loads(p.read_text()).get('family_id')==name]
        if len(accepted)>1: raise ValueError('Duplicate accepted family: '+name)
        if accepted:
            return dict(family=name,path=str(accepted[0]),passed=True,reused=True,seconds=0.)
        plan=root/'plans'/name/'plan.json'
        if not plan.is_file():
            if plan.parent.exists():
                # Interrupted planning can leave a directory before its JSON
                # is committed. Preserve it and regenerate the identical plan.
                archive=root/'plan_archive'/('partial_%d'%time.time_ns())
                archive.mkdir(parents=True,exist_ok=False)
                plan.parent.rename(archive/name)
            plan=make_plan(job['dataset_root'],root/'plans',job['layout'],job['split'],
                           job['cohort'],job['receiver'],job['serial'],job['target_class'],job['occluder'],job['margin'])
        # Failed captures stay inspectable. A retry gets a distinct location.
        if destination.exists():
            destination=destination.with_name(name+'_retry_%d'%time.time_ns())
        render(plan,destination)
        return dict(family=name,path=str(destination),passed=True,reused=False,seconds=time.monotonic()-started,
                    gpu=int(os.environ['MUJOCO_EGL_DEVICE_ID']),worker_pid=os.getpid())
    except Exception as exc:
        return dict(family=name,passed=False,error=str(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--layouts',nargs='+',type=int,required=True)
    p.add_argument('--cohorts',nargs='+',choices=['evidence','ambiguous'],default=['evidence','ambiguous'])
    p.add_argument('--receivers',nargs='+',type=int,choices=[0,1],default=[0,1])
    p.add_argument('--families-per-cell',type=int,default=1)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--gpus',type=int,default=2)
    p.add_argument('--margin',type=float,default=.3)
    p.add_argument('--target-classes',nargs='+',default=['forklift'])
    p.add_argument('--occluders',nargs='+',default=['rack'])
    a=p.parse_args()
    config=json.loads((Path(a.dataset_root)/'config.json').read_text())
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for layout in a.layouts:
        split='train' if layout<config['train_layouts'] else 'val' if layout<config['train_layouts']+config['val_layouts'] else 'test'
        for cohort in a.cohorts:
            for receiver in a.receivers:
                for serial in range(a.families_per_cell):
                    jobs.append(dict(dataset_root=str(Path(a.dataset_root).resolve()),output=str(root),layout=layout,
                        split=split,cohort=cohort,receiver=receiver,serial=serial,margin=a.margin,
                        target_class=a.target_classes[serial%len(a.target_classes)],
                        occluder=a.occluders[(serial//len(a.target_classes))%len(a.occluders)],gpu=len(jobs)%a.gpus))
    (root/'collection_plan.json').write_text(json.dumps(jobs,indent=2)+'\n')
    if a.workers < 1 or a.gpus < 1: raise ValueError('Positive worker and GPU counts required')
    results=[]; stopping=[False]
    def stop(*_): stopping[0]=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    for row in bounded_capture(jobs,a.workers,a.gpus,stopping):
            results.append(row)
            report=dict(updated_utc=datetime.now(timezone.utc).isoformat(),total=len(jobs),completed=len(results),
                        passed=sum(x['passed'] for x in results),failed=sum(not x['passed'] for x in results),results=results)
            temp=root/'collection_status.tmp';temp.write_text(json.dumps(report,indent=2)+'\n')
            temp.replace(root/'collection_status.json')
            print(json.dumps({k:v for k,v in row.items() if k!='traceback'}),flush=True)
    if stopping[0]: raise SystemExit(130)
    if any(not row['passed'] for row in results):
        raise SystemExit(1)


if __name__=='__main__':
    main()
